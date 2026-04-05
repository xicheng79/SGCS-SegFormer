import os
import sys
import fnmatch
import time
import argparse
import numpy as np
import torch
from osgeo import gdal
from tqdm import tqdm


def gdal_to_numpy(data):
    return data if data.ndim == 2 else np.transpose(data, (1, 2, 0))


def build_weight_window(h, w, mode='hann'):
    if mode == 'gaussian':
        y = np.linspace(-1, 1, h, dtype=np.float32)
        x = np.linspace(-1, 1, w, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        win = np.exp(-(xx ** 2 + yy ** 2) / 0.5).astype(np.float32)
    else:
        wy = np.hanning(h) if h > 1 else np.ones(1, dtype=np.float32)
        wx = np.hanning(w) if w > 1 else np.ones(1, dtype=np.float32)
        win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 1e-6)


def compute_positions(full_size, crop_size, stride):
    if full_size <= crop_size:
        return [0]
    pos = list(range(0, full_size - crop_size + 1, stride))
    if pos[-1] != full_size - crop_size:
        pos.append(full_size - crop_size)
    return pos


class UnifiedMMSegSolver:
    def __init__(self, config_file, model_file, device='cuda:0', mmseg_version='1x'):
        self.config_file = config_file
        self.checkpoint_file = model_file
        self.mmseg_version = mmseg_version.lower()
        if self.mmseg_version == '0x':
            from mmseg.apis import init_segmentor
            self.model = init_segmentor(config_file, model_file, device=device)
        elif self.mmseg_version == '1x':
            from mmseg.apis import init_model
            self.model = init_model(config_file, model_file, device=device)
        else:
            raise ValueError("mmseg_version 仅支持 '0x' 或 '1x'")

    def _to_scores(self, pred, class_number):
        pred = np.asarray(pred)
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[0] == class_number:
            return pred.astype(np.float32)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 2:
            h, w = pred.shape
            scores = np.zeros((class_number, h, w), dtype=np.float32)
            pred = pred.astype(np.int64)
            valid = (pred >= 0) & (pred < class_number)
            for k in range(class_number):
                scores[k][valid & (pred == k)] = 1.0
            return scores
        raise ValueError(f'无法识别的预测输出形状: {pred.shape}')

    def predict_scores(self, img, class_number):
        """直接前向推理获取 logits/概率，绕过模型内部的滑窗机制。

        外层 SlidingPredictor 已经完成滑窗裁块，这里输入的 img 是单个小块，
        不应再经过 inference_segmentor/inference_model 触发模型内部的
        slide_inference，否则会导致 encode_decode 参数不匹配。
        """
        if self.mmseg_version == '0x':
            from mmcv.parallel import collate, scatter
            from mmseg.datasets.pipelines import Compose
            from mmseg.apis.inference import LoadImage
            cfg = self.model.cfg
            device = next(self.model.parameters()).device
            test_pipeline = Compose([LoadImage()] + cfg.data.test.pipeline[1:])
            data = test_pipeline(dict(img=img))
            data = collate([data], samples_per_gpu=1)
            if next(self.model.parameters()).is_cuda:
                data = scatter(data, [device])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']]
            with torch.no_grad():
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0]
                if hasattr(self.model, 'overlap'):
                    # PanoOptiNet_EncoderDecoder: encode_decode 需要 overlap 参数
                    seg_logit = self.model.encode_decode(img_tensor, img_metas, self.model.overlap)
                else:
                    seg_logit = self.model.encode_decode(img_tensor, img_metas)
                pred = seg_logit.cpu().numpy()
            return self._to_scores(pred, class_number)
        from mmseg.apis import inference_model
        result = inference_model(self.model, img)
        if hasattr(result, 'seg_logits') and result.seg_logits is not None:
            pred = result.seg_logits.data.cpu().numpy()
        elif hasattr(result, 'pred_sem_seg') and result.pred_sem_seg is not None:
            pred = result.pred_sem_seg.data.cpu().numpy()
        else:
            raise ValueError('mmseg 1.x 输出中未找到 seg_logits 或 pred_sem_seg')
        return self._to_scores(pred, class_number)


class SlidingPredictor:
    def __init__(self, target_size, overlap_rate, class_number, weight_mode='hann'):
        if target_size <= 0:
            raise ValueError('target_size 必须大于 0')
        if not (0 <= overlap_rate < 1):
            raise ValueError('overlap_rate 必须满足 0 <= overlap_rate < 1')
        self.target_size = target_size
        self.overlap_rate = overlap_rate
        self.class_number = class_number
        self.stride = target_size - int(target_size * overlap_rate)
        self.weight_mode = weight_mode
        if self.stride <= 0:
            raise ValueError('stride 必须大于 0，请减小 overlap_rate 或增大 target_size')

    def read_block(self, ds, x, y):
        rw = min(self.target_size, ds.RasterXSize - x)
        rh = min(self.target_size, ds.RasterYSize - y)
        block = ds.ReadAsArray(x, y, rw, rh)
        if block is None:
            raise RuntimeError(f'读取影像块失败: x={x}, y={y}, w={rw}, h={rh}')
        pad = ((0, self.target_size - rh), (0, self.target_size - rw)) if block.ndim == 2 else ((0, 0), (0, self.target_size - rh), (0, self.target_size - rw))
        return np.pad(block, pad, mode='edge'), rh, rw

    def predict_large_image(self, ds, solver, out_path):
        """流式滑窗预测，边预测边写入 GDAL 文件，内存占用仅为活跃窗口行。"""
        h, w = ds.RasterYSize, ds.RasterXSize
        xs = compute_positions(w, self.target_size, self.stride)
        ys = compute_positions(h, self.target_size, self.stride)
        full_weight = build_weight_window(self.target_size, self.target_size, self.weight_mode)

        # 创建输出文件
        dst = gdal.GetDriverByName('GTiff').Create(out_path, w, h, 1, gdal.GDT_Byte)
        dst.SetGeoTransform(ds.GetGeoTransform())
        dst.SetProjection(ds.GetProjection())

        # 缓冲区：只保留 [buf_y0, buf_y0 + buf_rows) 范围内的行
        buf_y0 = ys[0]
        buf_rows = min(self.target_size, h - buf_y0)
        score_buf = np.zeros((self.class_number, buf_rows, w), dtype=np.float32)
        weight_buf = np.zeros((buf_rows, w), dtype=np.float32)
        flushed_to = 0  # 已写入到输出文件的行号

        t0 = time.time()
        with tqdm(total=len(xs) * len(ys), desc='滑窗预测', unit='patch') as pbar:
            for yi, y in enumerate(ys):
                # 在处理新 y 行之前，把已经不会被后续 patch 覆盖的行写出
                # 当前 y 及之后的 patch 只影响 [y, ...) 行，所以 [flushed_to, y) 可以写出
                if y > flushed_to:
                    self._flush_rows(dst, score_buf, weight_buf, buf_y0,
                                     flushed_to, y)
                    flushed_to = y
                    # 裁剪缓冲区，丢弃已写出的行
                    keep_from = flushed_to - buf_y0
                    if keep_from > 0:
                        score_buf = score_buf[:, keep_from:, :].copy()
                        weight_buf = weight_buf[keep_from:, :].copy()
                        buf_y0 = flushed_to

                # 确保缓冲区能覆盖 [y, y + target_size)
                needed_end = min(y + self.target_size, h)
                buf_end = buf_y0 + score_buf.shape[1]
                if needed_end > buf_end:
                    extend = needed_end - buf_end
                    score_buf = np.concatenate(
                        [score_buf,
                         np.zeros((self.class_number, extend, w), dtype=np.float32)],
                        axis=1)
                    weight_buf = np.concatenate(
                        [weight_buf,
                         np.zeros((extend, w), dtype=np.float32)],
                        axis=0)

                # 处理当前 y 行的所有 x patch
                for x in xs:
                    block, rh, rw = self.read_block(ds, x, y)
                    scores = solver.predict_scores(
                        gdal_to_numpy(block), self.class_number)[:, :rh, :rw]
                    weight = full_weight[:rh, :rw]
                    local_y = y - buf_y0
                    score_buf[:, local_y:local_y + rh, x:x + rw] += \
                        scores * weight[None, :, :]
                    weight_buf[local_y:local_y + rh, x:x + rw] += weight
                    pbar.update(1)

        # 写出剩余行
        if flushed_to < h:
            self._flush_rows(dst, score_buf, weight_buf, buf_y0,
                             flushed_to, h)

        dst.FlushCache()
        dst = None
        print('分块预测耗时: %0.2f(min).' % ((time.time() - t0) / 60))

    @staticmethod
    def _flush_rows(dst, score_buf, weight_buf, buf_y0, y_start, y_end):
        """将 [y_start, y_end) 范围的行做 argmax 后写入 GDAL 文件。"""
        local_start = y_start - buf_y0
        local_end = y_end - buf_y0
        strip_score = score_buf[:, local_start:local_end, :]
        strip_weight = weight_buf[local_start:local_end, :]
        fused = strip_score / np.maximum(strip_weight, 1e-6)[None, :, :]
        result = np.argmax(fused, axis=0).astype(np.uint8)
        dst.GetRasterBand(1).WriteArray(result, 0, y_start)



def collect_images(folder, pattern):
    return [os.path.join(folder, n) for n in fnmatch.filter(os.listdir(folder), pattern)]


def build_output_name(img_path, solver, predictor):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    try:
        config_name = os.path.splitext(os.path.basename(solver.config_file))[0]
        model_name = os.path.splitext(os.path.basename(solver.checkpoint_file))[0]
        return (
            f'{stem}_result_'
            f'ver{solver.mmseg_version}_size{predictor.target_size}_'
            f'overlap{int(predictor.overlap_rate * 100)}_'
            f'weight{predictor.weight_mode}_config{config_name}_model{model_name}.tif'
        )
    except Exception:
        return f'{stem}_result.tif'


def parse_args():
    parser = argparse.ArgumentParser(description='统一的遥感影像滑窗预测脚本（支持 mmseg 0.x / 1.x）')
    parser.add_argument('--predictImgPath', type=str, required=True, help='待预测影像文件夹')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件夹')
    parser.add_argument('--config_file', type=str, required=True, help='模型配置文件')
    parser.add_argument('--model_file', type=str, required=True, help='模型权重文件')
    parser.add_argument('--mmseg_version', type=str, default='1x', choices=['0x', '1x'], help='mmseg 版本')
    parser.add_argument('--overlap_rate', type=float, default=0.2, help='重叠率')
    parser.add_argument('--target_size', type=int, default=512, help='滑窗尺寸')
    parser.add_argument('--numclass', type=int, default=2, help='类别数')
    parser.add_argument('--Img_type', type=str, default='*.tif', help='待预测影像类型')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--weight_mode', type=str, default='hann', choices=['hann', 'gaussian'], help='融合权重窗类型')
    return parser.parse_args()

# python tools/sliding_predict_mmseg.py --predictImgPath "G:\GoogelEarthImage\H48F017017" --output_path "G:\GoogelEarthImage\H48F017017\0405result_water" --config_file "G:\26_水体论文_新实验数据\模型\0331panooptinet_mit_b4_ndwi\segformer_mit-b4_greenland-0331.py" --model_file "G:\26_水体论文_新实验数据\模型\0331panooptinet_mit_b4_ndwi\iter_576000.pth" --mmseg_version "0x" --overlap_rate 0.2 --target_size 512 --numclass 2 --Img_type "*.tif" --device "cuda:0" --weight_mode "hann"

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    images = collect_images(args.predictImgPath, args.Img_type)
    if not images:
        print('listpic is none')
        sys.exit(1)
    print(images)
    solver = UnifiedMMSegSolver(
        config_file=args.config_file,
        model_file=args.model_file,
        device=args.device,
        mmseg_version=args.mmseg_version,
    )
    predictor = SlidingPredictor(
        target_size=args.target_size,
        overlap_rate=args.overlap_rate,
        class_number=args.numclass,
        weight_mode=args.weight_mode,
    )
    for img_path in images:
        ds = gdal.Open(img_path)
        if ds is None:
            print(f'failed to open img: {img_path}')
            sys.exit(1)
        out_name = build_output_name(img_path, solver, predictor)
        out_path = os.path.join(args.output_path, out_name)
        predictor.predict_large_image(ds, solver, out_path)
        print(f'预测完成: {out_path}')


if __name__ == '__main__':
    main()
