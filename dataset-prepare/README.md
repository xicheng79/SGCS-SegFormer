# dataset-prepare

## 文件说明

### GEE_RGB_12+1.js

该脚本用于处理 Sentinel-2 数据集，并生成每月和全年的 RGB 图像。主要步骤包括：

1. 加载自定义 Shapefile 作为边界。
2. 选择并筛选 Sentinel-2 数据集。
3. 定义 RGB 计算函数。
4. 计算每月平均 RGB。
5. 计算全年平均 RGB。
6. 可视化 RGB。
7. 导出 RGB 数据为 GeoTIFF 文件。

### GEE_NDVI_12+1.js

该脚本用于处理 Sentinel-2 数据集，并生成每月和全年的 NDVI 图像。主要步骤包括：

1. 加载自定义 Shapefile 作为边界。
2. 选择并筛选 Sentinel-2 数据集。
3. 定义 NDVI 计算函数。
4. 计算每月平均 NDVI。
5. 计算全年平均 NDVI。
6. 可视化 NDVI。
7. 导出 NDVI 数据为 GeoTIFF 文件。

## 使用方法

1. 确保您已在 Google Earth Engine 中上传了自定义 Shapefile，并替换脚本中的 `boundaryAsset` 路径。
2. 在 Google Earth Engine 代码编辑器中打开 `GEE_RGB_12+1.js` 或 `GEE_NDVI_12+1.js`。
3. 运行脚本以生成并可视化每月和全年的 RGB 或 NDVI 图像。
4. 脚本会自动将生成的图像导出到您的 Google Drive 中指定的文件夹。

## 注意事项

- 请确保您的 Google Drive 中已创建相应的文件夹（如 `GEE_RGB_Chengdu` 和 `GEE_NDVI_Chengdu`）。
- 根据需要调整脚本中的参数（如时间范围、云量阈值等）。