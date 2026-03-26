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
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) // 云量低于 20%
  .select(['B2', 'B3', 'B4', 'B8']);                 // 选择蓝、绿、红、近红外波段

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
 * 3. 定义 RGB 计算函数
 *******************************/

/**
 * 生成RGB图像
 * @param {ee.Image} image - Sentinel-2影像
 * @return {ee.Image} - 添加RGB波段的影像
 */
function generateRGB(image) {
  // 选择红、绿、蓝波段（B4, B3, B2）
  var rgb = image.select(['B4', 'B3', 'B2']).rename(['Red', 'Green', 'Blue']);
  
  // 归一化波段值到0-1范围
  rgb = rgb.divide(10000);
  
  // 将RGB波段添加到影像中
  return image.addBands(rgb).copyProperties(image, ['system:time_start']);
}

/*******************************
 * 4. 计算每月平均 RGB
 *******************************/

// 定义月份列表：1~12
var months = ee.List.sequence(1, 12);

// 按月计算平均 RGB
var monthlyRGB = months.map(function(m) {
  var start = ee.Date.fromYMD(2020, m, 1);   // 月初
  var end   = start.advance(1, 'month');     // 下个月初
  
  // 筛选当月影像并生成RGB
  var monthlyImages = sentinel2
    .filterDate(start, end)
    .map(generateRGB)
    .select(['Red', 'Green', 'Blue']);
  
  // 计算当月平均 RGB
  var monthlyImage = monthlyImages.mean().set('month', m);
  
  // 返回裁剪后的影像
  return monthlyImage.clip(boundary);
});

// 转换为 ImageCollection 便于进一步处理或查看
monthlyRGB = ee.ImageCollection.fromImages(monthlyRGB);
print('Monthly RGB Collection:', monthlyRGB);

/*******************************
 * 5. 计算全年平均 RGB
 *******************************/

// 选择所有月份的RGB波段并计算像素均值
var annualRGB = monthlyRGB.select(['Red', 'Green', 'Blue']).mean().clip(boundary);

// 打印全年平均RGB影像信息到控制台
print('Annual RGB:', annualRGB);

// 检查全年平均RGB是否包含波段
var rgbBandNames = annualRGB.bandNames();
print('全年平均RGB波段名称:', rgbBandNames);

rgbBandNames.evaluate(function(names) {
  if (names.length === 0) {
    throw '全年平均RGB影像没有任何波段。请检查RGB计算步骤。';
  } else {
    print('全年平均RGB影像包含波段数量:', names.length);
    print('波段名称:', names);
  }
});

/*******************************
 * 6. 可视化 RGB
 *******************************/

// 定义RGB可视化参数
var rgbVis = {
  min: 0,
  max: 1,
  bands: ['Red', 'Green', 'Blue']
};

// 可视化每月平均RGB（示例：1月和12月）
var januaryRGB = monthlyRGB.filter(ee.Filter.eq('month', 1)).first();
var decemberRGB = monthlyRGB.filter(ee.Filter.eq('month', 12)).first();

Map.addLayer(januaryRGB, rgbVis, '2020年1月平均RGB');
Map.addLayer(decemberRGB, rgbVis, '2020年12月平均RGB');

// 可视化全年平均RGB
Map.addLayer(annualRGB, rgbVis, '2020年全年平均RGB');

/*******************************
 * 7. 导出 RGB 数据为 GeoTIFF 文件
 *******************************/

/**
 * 导出影像到Google Drive
 * @param {ee.Image} image - 要导出的影像
 * @param {string} description - 导出任务的描述
 */
function exportImage(image, description) {
  Export.image.toDrive({
    image: image,
    description: description,
    folder: 'GEE_RGB_Chengdu',              // 请提前在Google Drive中创建此文件夹
    fileNamePrefix: description,
    region: boundary.geometry(),
    scale: 10,                                // Sentinel-2的原始分辨率
    crs: 'EPSG:4326',                         // 坐标参考系统，可根据需要调整
    maxPixels: 1e13                           // 允许的最大像元数
  });
}

// 导出每月平均RGB
months.evaluate(function(monthList) {
  monthList.forEach(function(m) {
    var monthImage = monthlyRGB.filter(ee.Filter.eq('month', m)).first();
    if (monthImage) {
      var description = 'Monthly_RGB_Chengdu_2020_Month_' + m;
      exportImage(monthImage, description);
    }
  });
});

// 导出全年平均RGB
exportImage(annualRGB, 'Annual_RGB_Chengdu_2020');
