"""
高分辨率谷歌影像预测脚本
基于PanoOptiNet模型，对大幅面高分辨率谷歌卫星影像进行语义分割预测。
支持GeoTIFF格式输入，保留地理坐标信息，采用滑窗重叠拼接策略避免边缘效应。
"""

import os
import sys
import math
import argparse
import time
import numpy as np
import cv2
from osgeo import gdal
from tqdm import tqdm

# 将项目根目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mmseg.apis import init_segmentor, inference_segmentor


def gdal_to_opencv(gdal_data):
    """将GDAL读取的影像数据(bands, h, w)转换为OpenCV格式(h, w, bands-BGR)"""
    bands, h, w = gdal_data.shape
    if 'int8' in gdal_data.dtype.name:
        cv_data = np.zeros((h, w, bands), np.uint8)
    elif 'int16' in gdal_data.dtype.name:
        cv_data = np.zeros((h, w, bands), np.uint16)
    else:
        cv_data = np.zeros((h, w, bands), np.float32)

    # GDAL波段顺序(R,G,B) -> OpenCV顺序(B,G,R)
    for i in range(bands):
        cv_data[:, :, i] = gdal_data[bands - 1 - i, :, :]
    return cv_data


def predict_image(config_file, checkpoint_file, input_path, output_path,
                  target_size=512, overlap_rate=0.25, device='cuda:0',
                  morph_kernel=5, min_area=200):
    """
    对单张高分辨率谷歌影像进行预测（内存优化版）

    针对超大影像采用分块策略，避免加载整幅影像到内存。
    使用中心区域优先策略：只保存每个块的中心部分结果，重叠区域采用简单拼接。

    Args:
        config_file: 模型配置文件路径
        checkpoint_file: 模型权重文件路径
        input_path: 输入影像路径 (GeoTIFF)
        output_path: 输出结果路径 (GeoTIFF)
        target_size: 滑窗大小，默认512
        overlap_rate: 重叠率，默认0.25 (25%)
        device: 推理设备，默认 'cuda:0'
        morph_kernel: 形态学开运算核大小(像素)，去除小斑块噪声，0=不使用
        min_area: 最小连通域面积(像素数)，小于此面积的斑块被移除，0=不过滤
    """
    # ========== 1. 加载模型 ==========
    print(f'[1/5] 加载模型...')
    print(f'      配置文件: {config_file}')
    print(f'      权重文件: {checkpoint_file}')
    model = init_segmentor(config_file, checkpoint_file, device=device)
    print(f'      模型加载完成，设备: {device}')

    # ========== 2. 打开影像 ==========
    print(f'[2/5] 打开影像: {input_path}')
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f'错误: 无法打开影像文件 {input_path}')
        sys.exit(1)

    img_w = ds.RasterXSize
    img_h = ds.RasterYSize
    img_bands = ds.RasterCount
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()

    print(f'      影像尺寸: {img_w} x {img_h}, 波段数: {img_bands}')
    if geotrans:
        print(f'      空间分辨率: {abs(geotrans[1]):.6f} x {abs(geotrans[5]):.6f}')
    if proj:
        print(f'      坐标系: {proj[:80]}...')

    # ========== 3. 计算分块参数 ==========
    print(f'[3/5] 计算分块参数...')
    overlap = int(target_size * overlap_rate)
    stride = target_size - overlap

    # 计算行列方向的块数
    cols = max(1, math.ceil((img_w - overlap) / stride))
    rows = max(1, math.ceil((img_h - overlap) / stride))
    total_blocks = rows * cols

    print(f'      滑窗大小: {target_size}x{target_size}')
    print(f'      重叠率: {overlap_rate*100:.0f}% ({overlap}px)')
    print(f'      步长: {stride}px')
    print(f'      分块数: {cols} x {rows} = {total_blocks} 块')

    # ========== 4. 创建输出影像 ==========
    print(f'[4/5] 创建输出影像...')
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_path, img_w, img_h, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
    if proj:
        dst_ds.SetProjection(proj)
    if geotrans:
        dst_ds.SetGeoTransform(geotrans)

    # ========== 5. 分块预测（内存优化版） ==========
    print(f'[5/5] 开始分块预测...')
    t0 = time.time()

    pbar = tqdm(total=total_blocks, desc='预测进度', unit='block')
    
    for row in range(rows):
        for col in range(cols):
            # 计算当前块的起始坐标
            x_start = col * stride
            y_start = row * stride

            # 边界修正：确保不超出影像范围
            if x_start + target_size > img_w:
                x_start = img_w - target_size
            if y_start + target_size > img_h:
                y_start = img_h - target_size

            # 确保起始坐标不为负
            x_start = max(0, x_start)
            y_start = max(0, y_start)

            # 计算实际读取尺寸
            read_w = min(target_size, img_w - x_start)
            read_h = min(target_size, img_h - y_start)

            # 读取影像块
            block_data = ds.ReadAsArray(x_start, y_start, read_w, read_h)
            if block_data is None:
                pbar.update(1)
                continue

            # 处理不足target_size的块：填充
            if read_w < target_size or read_h < target_size:
                if block_data.ndim == 2:
                    padded = np.zeros((target_size, target_size), dtype=block_data.dtype)
                    padded[:read_h, :read_w] = block_data
                else:
                    padded = np.zeros((block_data.shape[0], target_size, target_size),
                                     dtype=block_data.dtype)
                    padded[:, :read_h, :read_w] = block_data
                block_data = padded

            # 转换为OpenCV格式 (h, w, c)
            if block_data.ndim == 2:
                img_block = np.stack([block_data] * 3, axis=-1)
            else:
                img_block = gdal_to_opencv(block_data)

            # 模型推理
            pred = inference_segmentor(model, img_block)[0]

            # 计算要写入的区域 - 只取中心部分，避免重叠区域的边界效应
            # 对于非边界块，只保留中间的 stride x stride 区域
            margin = overlap // 2
            
            write_y_start = y_start + (margin if row > 0 else 0)
            write_y_end = y_start + read_h - (margin if row < rows - 1 else 0)
            write_x_start = x_start + (margin if col > 0 else 0)
            write_x_end = x_start + read_w - (margin if col < cols - 1 else 0)
            
            crop_y_start = margin if row > 0 else 0
            crop_y_end = read_h - (margin if row < rows - 1 else 0)
            crop_x_start = margin if col > 0 else 0
            crop_x_end = read_w - (margin if col < cols - 1 else 0)
            
            # 裁剪预测结果
            pred_crop = pred[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            pred_u8 = np.uint8(pred_crop * 255)
            
            # 写入输出影像
            dst_ds.GetRasterBand(1).WriteArray(pred_u8, write_x_start, write_y_start)

            pbar.update(1)

    pbar.close()

    dst_ds.GetRasterBand(1).SetNoDataValue(0)
    dst_ds.FlushCache()
    dst_ds = None
    ds = None

    # ========== 6. 后处理：去除斑块噪声 ==========
    if morph_kernel > 0 or min_area > 0:
        print(f'[后处理] 去除斑块噪声...')
        post_process(output_path, morph_kernel=morph_kernel, min_area=min_area)

    elapsed = time.time() - t0
    print(f'\n预测完成!')
    print(f'  耗时: {elapsed/60:.2f} 分钟')
    print(f'  输出: {output_path}')
    print(f'  影像尺寸: {img_w} x {img_h}')


def post_process(tif_path, morph_kernel=5, min_area=200):
    """
    对预测结果进行后处理，去除斑块噪声。
    采用分块读写策略处理超大影像，避免内存溢出。

    处理步骤:
      1. 形态学开运算+闭运算 (分块处理，带边缘扩展避免块边界伪影)
      2. 连通域面积过滤 (分块处理，使用向量化查找表加速)

    Args:
        tif_path: 预测结果GeoTIFF路径
        morph_kernel: 形态学操作核大小(像素)，值越大去噪越强，0=跳过
        min_area: 最小保留面积(像素数)，小于此值的连通域被移除，0=跳过
    """
    ds = gdal.Open(tif_path, gdal.GA_Update)
    if ds is None:
        print(f'警告: 无法打开文件进行后处理 {tif_path}')
        return

    img_w = ds.RasterXSize
    img_h = ds.RasterYSize
    band = ds.GetRasterBand(1)

    # 分块大小: 每块 4096x4096，对于超大影像分块处理
    BLOCK = 4096
    # 形态学操作需要的边缘扩展量 (避免块边界产生伪影)
    pad = max(morph_kernel * 2, 16) if morph_kernel > 0 else 0

    total_removed = 0

    rows = math.ceil(img_h / BLOCK)
    cols = math.ceil(img_w / BLOCK)
    total_tiles = rows * cols

    print(f'  影像尺寸: {img_w}x{img_h}, 分{cols}x{rows}={total_tiles}块后处理')

    pbar = tqdm(total=total_tiles, desc='  后处理进度', unit='tile')

    for tr in range(rows):
        for tc in range(cols):
            # 计算当前块的范围（含扩展边缘）
            y0 = tr * BLOCK
            x0 = tc * BLOCK
            y1 = min(y0 + BLOCK, img_h)
            x1 = min(x0 + BLOCK, img_w)

            # 带边缘扩展的读取范围
            ey0 = max(0, y0 - pad)
            ex0 = max(0, x0 - pad)
            ey1 = min(img_h, y1 + pad)
            ex1 = min(img_w, x1 + pad)

            # 读取扩展区域
            tile = band.ReadAsArray(ex0, ey0, ex1 - ex0, ey1 - ey0)
            binary = np.where(tile > 127, 255, 0).astype(np.uint8)

            before_count = int(np.sum(binary > 0))

            # 步骤1: 形态学开运算+闭运算
            if morph_kernel > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 步骤2: 连通域面积过滤 (向量化查找表，不逐个循环)
            if min_area > 0:
                binary = _filter_small_components(binary, min_area)

            after_count = int(np.sum(binary > 0))
            total_removed += abs(before_count - after_count)

            # 裁剪掉扩展边缘，只写回核心区域
            cy0 = y0 - ey0
            cx0 = x0 - ex0
            core = binary[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
            band.WriteArray(core, x0, y0)

            pbar.update(1)

    pbar.close()
    ds.FlushCache()
    ds = None

    total_pixels = img_w * img_h
    print(f'  后处理完成: 共修正 {total_removed} 像素 '
          f'({total_removed / total_pixels * 100:.3f}%)')


def _filter_small_components(binary, min_area):
    """
    使用向量化查找表快速过滤小面积连通域。
    替代逐连通域循环，速度提升数十倍。
    """
    # 过滤前景小斑块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if num_labels > 1:
        areas = stats[:, cv2.CC_STAT_AREA]
        # 构建查找表: keep[label_id] = 该连通域是否保留
        keep = (areas >= min_area).astype(np.uint8)
        keep[0] = 0  # 背景始终为0
        # 向量化映射: 直接用labels作为索引查表
        binary = keep[labels] * 255

    # 过滤背景中的小孔洞
    bg = (255 - binary).astype(np.uint8)
    num_labels_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(
        bg, connectivity=8)
    if num_labels_bg > 1:
        areas_bg = stats_bg[:, cv2.CC_STAT_AREA]
        keep_bg = (areas_bg >= min_area).astype(np.uint8)
        keep_bg[0] = 1  # 背景区域的"背景"=真正的前景，保留
        # 小孔洞(keep_bg=0)填充为前景
        fill_mask = (keep_bg[labels_bg] == 0)
        binary[fill_mask] = 255

    return binary.astype(np.uint8)


def _build_weight_map(size):
    """
    构建高斯权重矩阵，中心权重高、边缘权重低。
    用于重叠区域的加权平均，实现无缝拼接。
    """
    sigma = size / 4.0
    center = size / 2.0
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    weight = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * sigma**2))
    # 归一化到 [0.1, 1.0]，避免边缘权重过低
    weight = weight / weight.max() * 0.9 + 0.1
    return weight


def batch_predict(config_file, checkpoint_file, input_dir, output_dir,
                  target_size=512, overlap_rate=0.25, img_pattern='*.tif',
                  device='cuda:0', morph_kernel=5, min_area=200):
    """
    批量预测文件夹中的所有影像

    Args:
        input_dir: 输入影像文件夹
        output_dir: 输出结果文件夹
        img_pattern: 影像文件匹配模式
        其他参数同 predict_image
    """
    import fnmatch

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有匹配的影像文件
    all_files = os.listdir(input_dir)
    tif_files = fnmatch.filter(all_files, img_pattern)

    if not tif_files:
        print(f'未找到匹配 "{img_pattern}" 的影像文件: {input_dir}')
        return

    print(f'找到 {len(tif_files)} 张待预测影像')
    print('=' * 60)

    for i, filename in enumerate(tif_files):
        print(f'\n[{i+1}/{len(tif_files)}] 处理: {filename}')
        print('-' * 60)

        input_path = os.path.join(input_dir, filename)
        name_no_ext = os.path.splitext(filename)[0]
        output_name = f'{name_no_ext}_pred.tif'
        output_path = os.path.join(output_dir, output_name)

        predict_image(
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            input_path=input_path,
            output_path=output_path,
            target_size=target_size,
            overlap_rate=overlap_rate,
            device=device,
            morph_kernel=morph_kernel,
            min_area=min_area
        )

    print('\n' + '=' * 60)
    print(f'全部预测完成！共处理 {len(tif_files)} 张影像')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PanoOptiNet 高分辨率谷歌影像预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预测单张影像
  python tools/predict_googleearthmap.py \\
    --input "D:/google_imagery/area1.tif" \\
    --output "D:/results/area1_pred.tif" \\
    --config "work_dirs/segformer_mit-b3_512x512_160k_mygreenland.py" \\
    --checkpoint "work_dirs/iter_160000.pth"

  # 批量预测文件夹
  python tools/predict_googleearthmap.py \\
    --input_dir "E:/GoogleEarth/15_16" \\
    --output_dir "E:/GoogleEarth/newpredict" \\
    --config "work_dirs/panooptinet_mit_b4_ndwi/segformer_mit-b4_greenland-0331.py" \\
    --checkpoint "work_dirs/panooptinet_mit_b4_ndwi/iter_576000.pth" \\
    --overlap 0.2 --block_size 512
        """)

    # 模型参数
    parser.add_argument('--config', type=str, required=True,
                        help='模型配置文件路径 (.py)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径 (.pth)')

    # 输入输出 - 单张模式
    parser.add_argument('--input', type=str, default=None,
                        help='单张影像路径 (GeoTIFF)')
    parser.add_argument('--output', type=str, default=None,
                        help='单张影像输出路径 (GeoTIFF)')

    # 输入输出 - 批量模式
    parser.add_argument('--input_dir', type=str, default=None,
                        help='批量预测输入文件夹')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='批量预测输出文件夹')

    # 预测参数
    parser.add_argument('--block_size', type=int, default=512,
                        help='滑窗大小 (默认: 512)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='重叠率 0~1 (默认: 0.25)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备 (默认: cuda:0)')
    parser.add_argument('--img_type', type=str, default='*.tif',
                        help='影像文件匹配模式 (默认: *.tif)')

    # 后处理参数
    parser.add_argument('--morph_kernel', type=int, default=5,
                        help='形态学去噪核大小(像素)，越大去噪越强，0=不使用 (默认: 5)')
    parser.add_argument('--min_area', type=int, default=200,
                        help='最小连通域面积(像素数)，小于此值的斑块被移除，0=不过滤 (默认: 200)')

    args = parser.parse_args()

    # 参数校验
    if args.input and args.input_dir:
        print('错误: --input 和 --input_dir 不能同时指定')
        sys.exit(1)

    if not args.input and not args.input_dir:
        print('错误: 请指定 --input (单张) 或 --input_dir (批量)')
        sys.exit(1)

    # 单张预测模式
    if args.input:
        if not args.output:
            # 自动生成输出路径
            base, ext = os.path.splitext(args.input)
            args.output = f'{base}_pred{ext}'

        predict_image(
            config_file=args.config,
            checkpoint_file=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            target_size=args.block_size,
            overlap_rate=args.overlap,
            device=args.device,
            morph_kernel=args.morph_kernel,
            min_area=args.min_area
        )

    # 批量预测模式
    elif args.input_dir:
        if not args.output_dir:
            args.output_dir = os.path.join(args.input_dir, 'predict_results')

        batch_predict(
            config_file=args.config,
            checkpoint_file=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=args.block_size,
            overlap_rate=args.overlap,
            img_pattern=args.img_type,
            device=args.device,
            morph_kernel=args.morph_kernel,
            min_area=args.min_area
        )
