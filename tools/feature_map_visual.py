# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import argparse
import os
import re
import numpy as np
import mmcv
import torch
import torch.nn as nn
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.models import build_segmentor

class FeatureRecorder:
    """特征记录器，用于捕获指定层的输出"""
    def __init__(self, layer_names):
        self.layer_names = layer_names
        self.features = []
        self.handles = []
        
    def __enter__(self):
        self.features = []
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 移除所有注册的钩子
        for handle in self.handles:
            handle.remove()
            
    def _hook_fn(self, module, input, output):
        """前向钩子函数"""
        try:
            self.features.append(output.detach().cpu())
        except Exception as e:
            print(f"捕获特征失败: {str(e)}")
            
    def register_hooks(self, model):
        """在指定层注册钩子"""
        for name, module in model.named_modules():
            if name in self.layer_names:
                print(f"注册钩子到层: {name}")
                handle = module.register_forward_hook(self._hook_fn)
                self.handles.append(handle)
                
def clean_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def visualize_heatmaps(image, features, layer_names, save_dir, alpha=0.5):
    """
    可视化特征热力图
    :param image: 原始图像 (BGR格式)
    :param features: 特征列表
    :param layer_names: 层名称列表
    :param save_dir: 保存目录
    :param alpha: 叠加透明度
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*30} 开始可视化 {'='*30}")
    print(f"将保存结果到: {os.path.abspath(save_dir)}")
    
    try:
        # 确保图像为BGR格式
        if image.shape[-1] == 3 and image.dtype == np.uint8:
            base_img = image.copy()
        else:
            raise ValueError(f"无效的图像格式: shape={image.shape}, dtype={image.dtype}")
            
        for idx, (feature, layer_name) in enumerate(zip(features, layer_names)):
            try:
                # 特征维度处理
                if feature.dim() == 4:  # [N,C,H,W]
                    feature = feature.squeeze(0)
                if feature.dim() != 3:
                    print(f"跳过非常规维度特征 {layer_name}: dim={feature.dim()}")
                    continue
                    
                # 转换为HWC格式
                feature = feature.permute(1, 2, 0).numpy()
                
                # 生成热力图
                heatmap = np.mean(feature, axis=-1)
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # 调整热力图尺寸
                if heatmap.shape[:2] != base_img.shape[:2]:
                    heatmap = cv2.resize(heatmap, (base_img.shape[1], base_img.shape[0]))
                    
                # 叠加图像
                blended = cv2.addWeighted(base_img, 1-alpha, heatmap, alpha, 0)
                
                # 保存结果
                safe_name = layer_name.replace('.', '_')
                save_path = os.path.join(save_dir, f"{idx+1:02d}_{safe_name}.png")
                cv2.imwrite(save_path, blended)
                print(f"已保存: {os.path.basename(save_path)}")
                
            except Exception as e:
                print(f"处理层 {layer_name} 失败: {str(e)}")
                
    except Exception as e:
        print(f"可视化过程中发生严重错误: {str(e)}")

# 删除原有的参数定义
# parser.add_argument('img', help='输入图像路径')
# parser.add_argument('config', help='模型配置文件路径')
# parser.add_argument('checkpoint', help='模型权重文件路径')

def main():
    # 硬编码路径参数
    img_path = r"E:\greenlanddata\predict\feature_map\img\512_512\shp1\1_cropped_0_0.png"
    config_path = r"E:\greenlanddata\pth\IGSW_SGSW\segformer_mit-b3_512x512_160k_greenland.py"
    checkpoint_path = r"E:\greenlanddata\pth\IGSW_SGSW\iter_160000.pth"
    output_root = r'E:\greenlanddata\predict\feature_map\result\IGSW'  # 新增硬编码输出目录
    opacity = 0.5  # 新增透明度参数

    # 初始化模型
    print("\n"+ "="*30 + " 初始化模型 " + "="*30)
    model = init_segmentor(config_path, checkpoint_path, device='cuda:0')
    
    # 获取原始图像
    image = mmcv.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    
    # 创建特征记录器
    layer_names = [
        'decode_head.convs.1.activate',
        'decode_head.convs.2.activate',
        'decode_head.convs.3.activate',
        'decode_head.convs_get_overlap.2.activate',
        'decode_head.convs_get_overlap.3.conv',
        'decode_head.convs_get_overlap.3.activate',
        'decode_head.fusion_conv_overlap.conv',
        'decode_head.fusion_conv_overlap.activate'
    ]
    
    with FeatureRecorder(layer_names) as recorder:
        recorder.register_hooks(model)
        
        # 执行推理（修复后的单一推理）
        print("\n"+ "="*30 + " 执行推理 " + "="*30)
        result = inference_segmentor(model, img_path)  # 使用img_path变量
        
        # 创建保存目录
        img_name = clean_filename(os.path.splitext(os.path.basename(img_path))[0])
        save_dir = os.path.join(output_root, img_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存预测结果
        print("\n"+ "="*30 + " 保存预测结果 " + "="*30)
        pred_path = os.path.join(save_dir, 'prediction.png')
        
        # 修改后：保持原始结果维度
        binary_result = (result[0] > 0).astype(np.uint8) * 255  # 二值化处理
        cv2.imwrite(pred_path, binary_result)  # 直接保存二值图像
        
        # 使用原始结果进行可视化
        model.show_result(
            img_path,
            result,  # 使用原始结果
            palette=[[0,0,0], [255,255,255]],  # 强制二值调色板
            out_file=pred_path,
            opacity=opacity)
            
        # 生成热力图
        print("\n"+ "="*30 + " 生成热力图 " + "="*30)
        visualize_heatmaps(
            image=image,
            features=recorder.features,
            layer_names=layer_names,
            save_dir=save_dir,
            alpha=opacity)
            
    print("\n"+ "="*30 + " 处理完成 " + "="*30)

if __name__ == '__main__':
    main()
# python "C:\Users\20546\Desktop\0828调试代码 - 副本\要迁移至mmseg1.x\py39mm1x\Lib\site-packages\mmseg\models\segmentors\feature_map_visual.py" "C:\Users\20546\Desktop\0828调试代码\要迁移至mmseg1.x\数据water_ndwi_3968_0907\JPEGImages\H48F017017_clip3\99_H48F017017_clip3_99_0_0.png" "C:\Users\20546\Desktop\segformer_mit-b0_8xb2-160k_ade20k-512x512.py" "C:\Users\20546\Desktop\0828调试代码\要迁移至mmseg1.x\workdir\iter_160000.pth"
#li
# python "E:\Code\IGSW-Model-Greenland\tools\feature_map_visual.py" "E:\greenlanddata\predict\feature_map\img\512_512\shp1\1_cropped_0_0.png" "E:\greenlanddata\pth\IGSW_SGSW\segformer_mit-b3_512x512_160k_greenland.py" "E:\greenlanddata\pth\IGSW_SGSW\iter_160000.pth"
