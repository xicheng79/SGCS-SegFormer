import os
import sys
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import cv2
import fnmatch
from mmseg.apis import init_segmentor, inference_segmentor

# 将GDAL图像数据转换为OpenCV格式
def GdalData2OpencvData(GdalImg_data):
    # 根据数据类型为OpenCV图像分配内存
    if 'int8' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint8)  # 为uint8类型数据分配空间
    elif 'int16' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint16)  # 为uint16类型数据分配空间
    else:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.float32)  # 为float32类型数据分配空间
    
    # 遍历每个波段，并将GDAL中的波段按正确的顺序存储到OpenCV图像中（GDAL是倒序存储的）
    for i in range(GdalImg_data.shape[0]):
        OpencvImg_data[:, :, i] = GdalImg_data[GdalImg_data.shape[0] - i - 1, :, :]
    
    return OpencvImg_data  # 返回转换后的OpenCV图像数据

# 定义Block类，用于处理每个图像块的索引、重叠区域等信息
# 这个流程的核心目的是将大尺寸的遥感影像分成多个小的图像块（block），并处理每个小块的预测结果，最后通过拼接这些小块恢复成完整的图像。分块预测常用于处理大尺寸图像，因为一次性处理整个大图像可能会导致内存不足或计算效率低。

# `Block` 类的作用
# `Block` 类用于存储每个小图像块的信息，包含其在大图像中的位置、重叠区域的情况以及相应的图像路径。每个小图像块代表图像的一部分，在图像分割过程中被单独预测，然后最终通过重叠区域拼接成完整的图像。
# `file`：图像块对应的文件路径。每个图像块都会被保存为一个独立的文件，后续需要根据文件路径加载图像进行预测。
# `idx_row` 和 `idx_col`：图像块在二维数组中的位置（行和列的索引）。这些索引帮助在大图像中定位该图像块的相对位置。
# `top_overlap` 和 `left_overlap`**：当前图像块与上面或左边的图像块的重叠区域的像素数。重叠区域用于保证拼接时的平滑过渡。比如，当前图像块的下半部分可能与上面的图像块有重叠部分，这样在拼接时可以避免边缘的拼接断裂。
# `top_overlap_pic` 和 `left_overlap_pic`：分别是与当前图像块有重叠的上方和左侧图像块的文件路径。用这些路径可以访问到上方和左侧图像块的数据，以便进行重叠区域的处理。
# `start_x`** 和 **`start_y`：图像块在原始大图像中的起始坐标（左上角的像素坐标）。这些信息在拼接图像时很重要，确保图像块在目标图像中的正确位置。

# 流程概述
# 大图像分割为小块：首先，将大尺寸的遥感图像根据给定的目标块大小（`target_size`）和重叠率（`overlap_rate`）分割为多个小块。每个小块可以单独进行处理，避免了直接处理整个大图像时可能遇到的内存问题。
# 预测每个图像块：对于每个图像块，调用预测模型（如SegFormer、Deeplab等）进行处理，得到该图像块的预测结果。每个图像块的预测结果会被保存为一个单独的文件。
# 处理重叠区域：在拼接图像时，重叠区域的像素值需要特别处理，通常是取重叠部分的平均值，避免拼接处出现明显的接缝。通过`top_overlap` 和 `left_overlap`可以知道每个图像块与其他图像块的重叠区域大小，从而正确地拼接。
# 拼接图像：所有小图像块预测完成后，通过处理每个图像块的重叠部分，将这些图像块按正确的顺序拼接成原始大图像。这一步就是“拼接”过程，通过逐个读取图像块并结合重叠部分进行拼接，最后得到完整的预测结果。
class Block:
    def __init__(self, file, idx_row, idx_col, top_overlap, top_overlap_pic, left_overlap, left_overlap_pic, start_x, start_y):
        self.file = file  # 图像块对应的文件路径
        self.idx_row = idx_row  # 图像块在二维数组中的行索引
        self.idx_col = idx_col  # 图像块在二维数组中的列索引
        self.top_overlap = top_overlap  # 与上一行图像块的重叠像素数
        self.top_overlap_pic = top_overlap_pic  # 与上一行有重叠的图像块的文件路径
        self.left_overlap = left_overlap  # 与左边图像块的重叠像素数
        self.left_overlap_pic = left_overlap_pic  # 与左边有重叠的图像块的文件路径
        self.start_x = start_x  # 图像块在整幅图像中的x起始像素位置
        self.start_y = start_y  # 图像块在整幅图像中的y起始像素位置

# 定义MMSegSolver类，用于加载和使用MMSeg模型进行图像分割预测
class MMSegSolver():
    def __init__(self, config_file, model_file):
        self.config_file = config_file  # 配置文件路径
        self.checkpoint_file = model_file  # 训练好的模型文件路径
        self.model = init_segmentor(self.config_file, self.checkpoint_file, device='cuda')  # 加载MMSeg模型，使用GPU进行推理
    
    # 预测概率图（根据配置文件的test_cfg.return_logits设置输出为logits或类别）
    def predict_x_probs(self, img):
        logits = inference_segmentor(self.model, img)[0]  # 获取模型的预测logits
        res = np.uint8(logits * 255)  # 将logits转换为0-255之间的概率图
        return res  # 返回预测的概率图

    # 预测类别图（通过概率图进行argmax操作获得类别）
    def predict_x(self, img):
        logits = inference_segmentor(self.model, img)[0]  # 获取模型的预测logits
        return logits  # 返回logits结果

# 定义Predict类，用于图像块的预测操作
class Predict():
    def __init__(self, target_size, overlap_rate, class_number):
        self.target_size = target_size  # 图像块的目标尺寸
        self.overlap_rate = overlap_rate  # 图像块的重叠率
        self.class_number = class_number  # 类别数
    
    # 预测单个图像块，并将结果保存为PNG图像
    def predict_block(self, gd_img_block, predict, save_path, idx_row, idx_col):
        img_block = gd_img_block.transpose(1, 2, 0)  # 将图像块从(c, h, w)格式转为(h, w, c)格式
        img_block = GdalData2OpencvData(gd_img_block)  # 转换为OpenCV格式
        
        predict_out = predict(img_block)  # 预测图像块
        
        predict_out = np.uint8(predict_out * 255)  # 将预测结果转换为0-255之间的灰度图
        
        # 构造保存文件路径
        save_file = os.path.join(save_path, str(idx_row) + '+' + str(idx_col) + '.png')
        cv2.imwrite(save_file, predict_out)  # 保存为PNG文件
        return save_file  # 返回保存的文件路径
    
    # 预测并将结果写入文件（测试用）
    def predict_block_test_write_tf(self, gd_img_block, predict, save_path, idx_row, idx_col):
        img_block = gd_img_block.transpose(1, 2, 0)  # 将图像块从(c, h, w)格式转为(h, w, c)格式
        save_file = os.path.join(save_path, str(idx_row) + '+' + str(idx_col) + '.png')
        cv2.imwrite(save_file, img_block)  # 将图像块保存为PNG文件
        return save_file  # 返回保存的文件路径

    # 将整幅图像按块进行预测并保存每个图像块的预测结果
    def predict_as_blocks(self, dataset, overlap_rate, predict, save_path):
        t0 = time.time()  # 记录开始时间
        img_x = dataset.RasterXSize  # 获取图像的x方向尺寸
        img_y = dataset.RasterYSize  # 获取图像的y方向尺寸
        
        target_size = self.target_size  # 图像块的目标尺寸
        space = target_size - int(target_size * overlap_rate)  # 计算块之间的间距
        x_num = math.ceil((img_x - target_size) / space) + 1  # x方向上块的数量
        y_num = math.ceil((img_y - target_size) / space) + 1  # y方向上块的数量
        print("x_num:", x_num)
        print("y_num:", y_num)
        
        dst_pngs = [[Block for i in range(x_num)] for j in range(y_num)]  # 创建二维数组存储图像块
        
        # 按行列顺序进行图像块预测
        overlap = int(target_size * overlap_rate)
        for j in tqdm(range(0, y_num - 1)):  # 遍历y方向上的图像块
            for i in range(0, x_num - 1):  # 遍历x方向上的图像块
                x_start = space * i  # 计算x方向的起始位置
                y_start = space * j  # 计算y方向的起始位置
                img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)  # 读取图像块数据
                pic = self.predict_block(gd_img_block=img_block, predict=predict, save_path=save_path, idx_row=j, idx_col=i)  # 进行预测并保存结果
                bk = Block(pic, j, i, overlap, "", overlap, "", x_start, y_start)  # 创建Block对象
                dst_pngs[j][i] = bk  # 存储预测结果
        
        print('分块预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))  # 输出预测耗时
        return dst_pngs  # 返回预测结果

        # 构建预测图像块的索引，处理重叠区域
    def build_pic_index(self, dst_pngs):
        for j in dst_pngs:  # 遍历每一行的图像块
            for png in j:  # 遍历每一列的图像块
                # 如果当前图像块是第一行，则没有上方重叠区域
                if png.idx_row == 0:
                    png.top_overlap = 0  # 设置上方重叠区域为0
                else:
                    # 如果不是第一行，则获取上方图像块的索引
                    top_y = png.idx_row - 1  # 上方图像块的行索引
                    top_x = png.idx_col  # 上方图像块的列索引
                    # 获取上方重叠区域的图像块文件路径
                    png.top_overlap_pic = dst_pngs[top_y][top_x].file  # 获取与当前图像块有重叠的上方图像块文件路径
                
                # 如果当前图像块是第一列，则没有左侧重叠区域
                if png.idx_col == 0:
                    png.left_overlap = 0  # 设置左侧重叠区域为0
                else:
                    # 如果不是第一列，则获取左侧图像块的索引
                    left_x = png.idx_col - 1  # 左侧图像块的列索引
                    left_y = png.idx_row  # 左侧图像块的行索引
                    # 获取左侧重叠区域的图像块文件路径
                    png.left_overlap_pic = dst_pngs[left_y][left_x].file  # 获取与当前图像块有重叠的左侧图像块文件路径


        # 拼接图像块为完整的图像，重叠区域各取一半进行拼接
    def stitch_by_blocks(self, dst_pngs, dst_ds):
        target_size = self.target_size  # 获取目标图像块的大小
        for j in tqdm(dst_pngs, desc='拼接分块png:', unit='batch'):  # 遍历每一行的图像块
            for png in j:  # 遍历每一列的图像块
                block = cv2.imread(png.file, cv2.IMREAD_GRAYSCALE)  # 读取图像块为灰度图

                # 处理上方重叠区域
                if png.top_overlap > 0:
                    top_png = png.top_overlap_pic  # 获取上方重叠区域对应的图像块文件路径
                    top_block = cv2.imread(top_png, cv2.IMREAD_GRAYSCALE)  # 读取上方重叠区域的图像块
                    overlap = png.top_overlap  # 获取上方重叠区域的像素数
                    # 从上方图像块提取重叠区域的一部分，取上方重叠部分的下半部分
                    half_overlap = top_block[target_size - overlap: target_size - int(overlap * 0.5), 0: target_size]
                    # 将当前图像块的上方重叠区域部分更新为半重叠部分
                    block[0:overlap - int(overlap * 0.5), 0: target_size] = half_overlap  # 拼接上方重叠区域
                
                # 处理左侧重叠区域
                if png.left_overlap > 0:
                    left_png = png.left_overlap_pic  # 获取左侧重叠区域对应的图像块文件路径
                    left_block = cv2.imread(left_png, cv2.IMREAD_GRAYSCALE)  # 读取左侧重叠区域的图像块
                    overlap = png.left_overlap  # 获取左侧重叠区域的像素数
                    # 从左侧图像块提取重叠区域的一部分，取左侧重叠部分的右半部分
                    half_overlap = left_block[0: target_size, target_size - overlap: target_size - int(overlap * 0.5)]
                    # 将当前图像块的左侧重叠区域部分更新为半重叠部分
                    block[0: target_size, 0: overlap - int(overlap * 0.5)] = half_overlap  # 拼接左侧重叠区域

                # 将处理过的图像块写入目标数据集
                dst_ds.GetRasterBand(1).WriteArray(block, png.start_x, png.start_y)  # 将拼接后的图像块写入目标图像的相应位置
                cv2.imwrite(png.file, block)  # 将图像块保存为文件，以便后续使用

        dst_ds.FlushCache()  # 将数据缓存写入磁盘

    # 主函数：处理所有影像的预测
    def main(self, allpath, outpath, solver, overlap_rate=0.5, target_size=512):  
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            ds = gdal.Open(one_path)  # 打开影像文件
            if ds == None:
                print("failed to open img")
                sys.exit(1)
            d, n = os.path.split(one_path)  # 获取文件夹和文件名
            save_pngs_path = os.path.join(outpath, n)  # 预测结果保存路径
            if not os.path.exists(save_pngs_path):
                os.makedirs(save_pngs_path)  # 创建保存目录
            dst_pngs = self.predict_as_blocks(dataset=ds, overlap_rate=overlap_rate, predict=lambda xx: solver.predict_x(xx), save_path=save_pngs_path)  # 执行分块预测并保存结果
            self.build_pic_index(dst_pngs)  # 构建图片索引
            
            # 新建输出tif文件
            projinfo = ds.GetProjection()  # 获取投影信息
            geotransform = ds.GetGeoTransform()  # 获取地理变换信息
            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 创建GTiff格式的驱动
            name = f"{n[:-4]}_result.tif"  # 输出文件名
            try:
                config_name = os.path.basename(solver.config_file).split('_')[0]  # 获取配置文件名称
                model_name = os.path.basename(solver.checkpoint_file).split('.')[0]  # 获取模型文件名称
                name = f"{n[:-4]}_result_size{self.target_size}_overlap{int(self.overlap_rate * 100)}_config{config_name}_model{model_name}.tif"  # 拼接文件名
            except:
                pass
            outtif = os.path.join(outpath, name)  # 输出文件路径
            dst_ds = driver.Create(outtif, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)  # 创建新的tif文件
            dst_ds.SetGeoTransform(geotransform)   # 设置地理变换
            dst_ds.SetProjection(projinfo)  # 设置投影信息
            self.stitch_by_blocks(dst_pngs, dst_ds)  # 执行图像拼接

#在终端传入待预测文件夹路径等，以及重叠率，每次处理的小块大小，样本数量和输出的影像类型
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='遥感影像分割预测')
    parser.add_argument('--predictImgPath', type=str, required=True, 
                       help='待预测影像的文件夹路径')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出的预测结果路径')
    parser.add_argument('--config_file', type=str, required=True,
                       help='模型配置文件路径')
    parser.add_argument('--model_file', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--overlap_rate', type=float, default=0.2,
                       help='重叠率，默认为0.2')
    parser.add_argument('--target_size', type=int, default=512,
                       help='目标尺寸，默认为512')
    parser.add_argument('--numclass', type=int, default=2,
                       help='样本类别数，默认为2')
    parser.add_argument('--Img_type', type=str, default='*.tif',
                       help='待预测影像的类型，默认为*.tif')
    
    args = parser.parse_args()

    # 载入模型，使用自己实现的MMSegSolver类
solver = MMSegSolver(config_file=args.config_file, model_file=args.model_file)
# 初始化MMSegSolver类的实例，传入配置文件和模型文件路径，加载训练好的模型

# 如果输出路径不存在，则创建该路径
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
# 检查输出文件夹是否存在，如果不存在则创建一个新文件夹来保存预测结果

# 获取待预测影像路径下符合条件的所有文件列表
listpic = fnmatch.filter(os.listdir(args.predictImgPath), args.Img_type)
# 使用fnmatch根据指定的影像类型（例如.tif）获取文件夹下所有符合条件的影像文件

# 将每个文件的路径添加到listpic中
for i in range(len(listpic)):
    listpic[i] = os.path.join(args.predictImgPath + '/' + listpic[i])
# 遍历所有影像文件，生成每个影像的完整路径（包括文件名）

# 如果没有找到待预测的影像文件，则退出程序
if not listpic:
    print('listpic is none')
    exit(1)
else:
    print(listpic)
# 如果listpic为空，说明没有符合条件的影像文件，输出提示信息并退出程序；否则输出文件列表

# 创建Predict类的实例，用于分块预测
predict_instantiation = Predict(target_size=args.target_size, 
                                class_number=args.numclass, 
                                overlap_rate=args.overlap_rate)
# 初始化Predict类，传入目标图像块的尺寸、类别数以及重叠率

# 调用Predict类的main方法进行批量预测，并将预测结果保存到输出路径
predict_instantiation.main(listpic, args.output_path, solver, 
                           overlap_rate=args.overlap_rate, 
                           target_size=args.target_size)
# 执行预测操作：传入待预测的影像文件路径列表、输出路径、模型实例、重叠率、目标图像块大小等参数，完成对所有影像的预测并保存结果


# 运行代码
# python tools/overlap_predict_mm0x.py --predictImgPath "D:\pending_predict" --output_path "D:\pending_predict\result" --config_file "D:\IGSW-Model-Greenland\work_dirs\segformer_mit-b3_512x512_160k_mygreenland.py" --model_file "D:\IGSW-Model-Greenland\work_dirs\segformer_mit-b3_512x512_160k_mygreenland\iter_160000.pth" --overlap_rate 0.2 --target_size 512 --numclass 2 --Img_type *.tif