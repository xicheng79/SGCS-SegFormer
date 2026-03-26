import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from PIL import Image
import numpy as np

# 设置文件夹路径
shp_folder = r'E:\greenlanddata\predict\feature_map\shp'  # 多个shp文件夹路径
tif_file = r"E:\greenlanddata\predict\H48F017017\H48F017017.tif"  # tif文件路径
output_folder = r'E:\greenlanddata\predict\feature_map\img'  # 输出png文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历shp文件夹中的所有.shp文件
for shp_file in os.listdir(shp_folder):
    if shp_file.endswith('.shp'):
        # 获取shp文件的完整路径
        shp_path = os.path.join(shp_folder, shp_file)
        
        # 读取shp文件
        shapefile = gpd.read_file(shp_path)

        # 打开tif文件
        with rasterio.open(tif_file) as src:
            # 使用shp文件裁剪tif文件
            geoms = shapefile.geometry.values  # 获取shp的几何对象
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta

            # 更新metadata，以便将图像保存为PNG
            out_meta.update({"driver": "GTiff", "count": 3, "dtype": 'uint8'})  # 假设为RGB，3个波段

            # 去除冗余的维度并确保数据在0-255范围内
            out_image = np.moveaxis(out_image, 0, -1)  # 将波段轴移到最后，形成H x W x C的RGB图像
            out_image = np.clip(out_image, 0, 255).astype(np.uint8)  # 保证数据在0-255范围内，并转为uint8

            # 保存为PNG
            output_png_path = os.path.join(output_folder, f'{os.path.splitext(shp_file)[0]}.png')
            img = Image.fromarray(out_image)
            img.save(output_png_path)

        print(f'已保存 {output_png_path}')
