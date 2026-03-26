/*******************************
 * 1. 加载自定义Shapefile作为边界
 *******************************/

// 替换为您的Shapefile资产路径
var boundaryAsset = 'projects/ee-rayfineallday/assets/cdjcq';

// 加载Shapefile作为FeatureCollection
var boundary = ee.FeatureCollection(boundaryAsset);

// 打印边界信息以验证
print('加载的边界:', boundary);

// 在地图上显示边界
Map.centerObject(boundary, 10);
Map.addLayer(boundary, {color: 'red'}, '自定义边界');

/*******************************
 * 2. 选择并筛选 Sentinel-2 数据集
 *******************************/

// 加载 Sentinel-2 MSI 数据集
var sentinel2 = ee.ImageCollection("COPERNICUS/S2")
  .filterBounds(boundary)                             // 覆盖自定义边界
  .filterDate('2020-01-01', '2020-12-31')            // 时间范围：2020 年
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)); // 云量低于 20%

// 打印影像集信息到控制台
print('Sentinel-2 Image Collection大小:', sentinel2.size());
print('Sentinel-2 Image Collection:', sentinel2);

// 检查 Sentinel-2 影像集是否为空
sentinel2.size().evaluate(function(size) {
  if (size === 0) {
    throw '没有找到符合条件的 Sentinel-2 影像。请检查筛选条件，如日期范围、云量阈值等。';
  } else {
    print('找到符合条件的 Sentinel-2 影像数量:', size);
  }
});

/*******************************
 * 3. 定义 NDVI 计算函数
 *******************************/
function calculateNDVI(image) {
  // 使用近红外波段 (B8) 和红光波段 (B4) 计算 NDVI
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // 将NDVI波段添加到原始影像中
  return image.addBands(ndvi).copyProperties(image, ['system:time_start']);
}

/*******************************
 * 4. 计算每月平均 NDVI
 *******************************/

// 定义月份列表：1~12
var months = ee.List.sequence(1, 12);

// 按月计算平均 NDVI
var monthlyNDVI = months.map(function(m) {
  var start = ee.Date.fromYMD(2020, m, 1);   // 月初
  var end   = start.advance(1, 'month');     // 下个月初
  
  // 筛选当月影像并求平均值
  var monthlyImage = sentinel2
    .filterDate(start, end)
    .map(calculateNDVI)
    .select('NDVI')
    .mean()
    .set('month', m);
  
  return monthlyImage.clip(boundary);        // 裁剪到自定义边界区域
});

// 转换为 ImageCollection 便于进一步处理或查看
monthlyNDVI = ee.ImageCollection.fromImages(monthlyNDVI);
print('Monthly NDVI Collection:', monthlyNDVI);

/*******************************
 * 5. 计算全年平均 NDVI
 *******************************/

// 选择所有月份的NDVI波段并计算像素均值
var annualNDVI = monthlyNDVI.select('NDVI').mean().clip(boundary);
print('Annual NDVI:', annualNDVI);

// 检查全年平均NDVI是否包含波段
var bandNames = annualNDVI.bandNames();
print('全年平均NDVI波段名称:', bandNames);

bandNames.evaluate(function(names) {
  if (names.length === 0) {
    throw '全年平均NDVI影像没有任何波段。请检查NDVI计算步骤。';
  } else {
    print('全年平均NDVI影像包含波段数量:', names.length);
    print('波段名称:', names);
  }
});

/*******************************
 * 6. 可视化 NDVI
 *******************************/

// 定义NDVI可视化参数
var ndviVis = {
  min: 0,
  max: 1,
  palette: ['blue', 'white', 'green']
};

// 可视化每月平均NDVI（示例：1月和12月）
var januaryNDVI = monthlyNDVI.filter(ee.Filter.eq('month', 1)).first();
var decemberNDVI = monthlyNDVI.filter(ee.Filter.eq('month', 12)).first();

Map.addLayer(januaryNDVI, ndviVis, '2020年1月平均NDVI');
Map.addLayer(decemberNDVI, ndviVis, '2020年12月平均NDVI');

// 可视化全年平均NDVI
Map.addLayer(annualNDVI, ndviVis, '2020年全年平均NDVI');

/*******************************
 * 7. 导出 NDVI 数据为 GeoTIFF 文件
 *******************************/

// 定义导出函数
function exportNDVI(image, description) {
  Export.image.toDrive({
    image: image,
    description: description,
    folder: 'GEE_NDVI_Chengdu',              // 请提前在Google Drive中创建此文件夹
    fileNamePrefix: description,
    region: boundary.geometry(),
    scale: 10,                                // Sentinel-2的原始分辨率
    crs: 'EPSG:4326',                         // 坐标参考系统，可根据需要调整
    maxPixels: 1e13                           // 允许的最大像元数
  });
}

// 导出每月平均NDVI
months.evaluate(function(monthList) {
  monthList.forEach(function(m) {
    var monthImage = monthlyNDVI.filter(ee.Filter.eq('month', m)).first();
    var description = 'Monthly_NDVI_Chengdu_2020_Month_' + m;
    exportNDVI(monthImage, description);
  });
});

// 导出全年平均NDVI
exportNDVI(annualNDVI, 'Annual_NDVI_Chengdu_2020');
