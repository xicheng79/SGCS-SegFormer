import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from PIL import Image
import numpy as np
from shapely.geometry import box

# 设置文件夹路径
shp_folder = r'E:\greenlanddata\predict\feature_map\shp'  # shp文件夹路径
tif_folder = r'E:\greenlanddata\predict\H48F017017'  # tif文件夹路径
output_base_folder = r'E:\greenlanddata\predict\feature_map\img'  # 输出png文件夹路径

# 遍历shp文件夹中的所有.shp文件
for idx, shp_file in enumerate(os.listdir(shp_folder)):
    if shp_file.endswith('.shp'):
        # 获取shp文件的完整路径
        shp_path = os.path.join(shp_folder, shp_file)
        
        # 读取shp文件
        shapefile = gpd.read_file(shp_path)
        
        # 获取shp文件的文件名，去除扩展名
        shp_name = os.path.splitext(shp_file)[0]
        
        # 为每个shp文件创建一个新的输出文件夹
        # 在创建文件夹后添加权限检查
        output_folder = os.path.join(output_base_folder, f"shp{idx + 1}")
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                print(f"已创建输出目录：{output_folder}")
            except PermissionError:
                print(f"权限不足无法创建目录：{output_folder}")
                continue
        
        # 遍历tif文件夹中的所有.tif文件
        for tif_file in os.listdir(tif_folder):
            # 修改条件检查，增加调试信息
            if tif_file.endswith('.tif') and shp_name in tif_file:
                print(f"正在处理匹配的TIF文件：{tif_file}")  # 添加调试输出
                # 获取tif文件的完整路径
                tif_path = os.path.join(tif_folder, tif_file)

                # 打开tif文件
                with rasterio.open(tif_path) as src:
                    # 确保shp文件的CRS与tif文件的CRS一致
                    # 修改坐标系转换部分
                    # 在坐标系转换后添加坐标验证
                    try:
                        shapefile = shapefile.to_crs(src.crs)
                        # 新增调试信息
                        print(f"转换后坐标系：{shapefile.crs}")  # 显示转换后的CRS
                        print(f"TIF文件坐标系：{src.crs}")  # 显示栅格文件的CRS
                        print(f"shp文件范围：{shapefile.total_bounds}")  # 显示转换后的shp范围
                        print(f"tif文件范围：{src.bounds}")  # 显示栅格文件范围
                    except Exception as e:
                        print(f"坐标系转换失败：{str(e)}")
                        continue

                    # 对应shp文件进行裁剪
                    geoms = shapefile.geometry.values  # 获取shp的几何对象
                    
                    # 检查是否有重叠
                    for geom in geoms:
                        # 将栅格的边界转换为 Polygon 对象
                        raster_bounds = box(*src.bounds)
                        
                        if geom.intersects(raster_bounds):  # 如果几何与栅格边界重叠
                            print(f"Shapefile ({shp_name}) 与 TIF 文件 ({tif_file}) 有重叠")
                        else:
                            print(f"Shapefile ({shp_name}) 与 TIF 文件 ({tif_file}) 没有重叠")

                    # 进行裁剪，添加all_touched=True参数来处理边界重叠问题
                    out_image, out_transform = mask(src, geoms, crop=True, all_touched=True)
                    out_meta = src.meta

                    # 更新metadata，以便将图像保存为PNG
                    out_meta.update({"driver": "GTiff", "count": 1, "dtype": 'uint8'})

                    # 去除冗余的维度并确保数据在0-255范围内
                    out_image = np.squeeze(out_image)  # 去除多余的维度
                    out_image = np.clip(out_image, 0, 255).astype(np.uint8)  # 保证数据在0-255范围内，并转为uint8

                    # 确保输出图像是二维的
                    if len(out_image.shape) == 3:  # 如果是3维图像（高、宽、波段数）
                        out_image = out_image[0]  # 选择第一个波段

                    # 保存为PNG到指定的输出文件夹
                    output_png_path = os.path.join(output_folder, f'{shp_name}_{os.path.splitext(tif_file)[0]}.png')
                    img = Image.fromarray(out_image)
                    img.save(output_png_path)

                print(f'已保存 {output_png_path}')

                # 在保存前添加数据验证
                if out_image.size == 0:
                    print(f"裁剪得到空图像，跳过保存：{tif_file}")
                    continue

                # 保存前检查数组维度
                print(f"输出图像形状：{out_image.shape}")  # 添加维度调试
