import torch
from osgeo import gdal
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# 用于进行图像分割, 这个类的输入是图像的路径，标注的路径，图像的信息列表，输出是图像的分割结果`,用NDVI作为指标，用最大和作为选择标准
class SpectralAwareSampling():
    def __init__(self, image_path, annotation_path, image_info_list):
        """
        Constructor method
        """
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.image = gdal.Open(image_path).ReadAsArray()
        self.height, self.width = self.image.shape[-2:]
        self.num_bands = self.image.shape[0]
        self.shift_size = 128 # 这个是按经验定的，可能需要调整
        self.region_size = 512 # 这个是按经验定的，可能需要调整
        self.image_dir = r'mydata\greenland_ndvi_3968x3968_test\JPEGImages' # 这个路径下的图片都是3968x3968的PNG图片，第四个波段下是NDVI波段。
        self.annotation_dir = r'mydata\greenland_ndvi_3968x3968_test\SegmentationClass' # 这个路径下的图片都是3968x3968的PNG图片，是标注的图片。
        # 新增：初始化示意图存储目录（在image_dir下创建schematic子文件夹）
        self.schematic_dir = os.path.join(self.image_dir, 'schematic')
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.image_list = []
        self.image_info_list = image_info_list
        self.max_iterations = 100
        self.ndvi_band = self.image[-1,:,:]  # 提取NDVI波段
        self.integral_image = self.calculate_integral_image(self.ndvi_band)
        self.path_coords = []  # 记录路径中的区域左上角坐标(i, j)
        # 初始化示意图（使用前3波段作为基础图，转换为HWC格式的BGR图像）
        self.schematic_img = self.image[:3, :, :].transpose(1, 2, 0).copy()
        if self.schematic_img.dtype != np.uint8:  # 确保是uint8格式
            self.schematic_img = (self.schematic_img * 255).astype(np.uint8)
        
        # 添加一个检查函数检查image_dir和annotation_dir在本机上是否存在的函数，不存在这报为正确配置presegmentation.py下的image_dir和annotation_dir未配置正确的错误
        self.check_directories()
        
    def check_directories(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.annotation_dir):
            raise RuntimeError("请检查sgsw.py 下的 image_dir 和 annotation_dir 配置是否正确，如果路径正确仍然有错请检查是否正确挂载代码到环境。")
    

    def find_first_region(self, image_array):     
        """
        通过NDVI波段找到NDVI值最大的位置作为初始点。
        如果多个最大值相同，则随机选择一个最大值作为起点。
        返回：
        max_i_first: 最大NDVI值的位置的行索引
        max_j_first: 最大NDVI值的位置的列索引
        visited_regions: 用于记录已经访问过的区域的矩阵
        """
        print("[阶段1] 开始寻找初始区域...")
        ndwi_band = image_array[-1,:,:]
        visited_regions = np.zeros_like(ndwi_band)
        max_sum_first = -19e9
        max_positions = []  # 用来存储最大NDVI值位置

        for i in range(0, self.height - self.region_size + 1, self.region_size - self.shift_size):
            for j in range(0, self.width - self.region_size + 1, self.region_size - self.shift_size):
                region_ndwi = ndwi_band[i:i+self.region_size, j:j+self.region_size]
                region_sum = self.get_region_sum(i, j)  # 替换原sum计算

                if region_sum > max_sum_first:
                    max_sum_first = region_sum
                    max_positions = [(i, j)]  # 重新开始存储位置
                elif region_sum == max_sum_first:
                    max_positions.append((i, j))  # 添加相同NDVI值的位置

        # 随机选择一个位置作为起点
        max_i_first, max_j_first = max_positions[np.random.choice(len(max_positions))]
        # 记录初始区域坐标并绘制绿色虚线边框
        self.path_coords.append((max_i_first, max_j_first))
        x, y = max_j_first, max_i_first  # 列j对应x，行i对应y
        w, h = self.region_size, self.region_size
        cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 255, 0), 2, lineType=cv2.LINE_4)  # 绿色虚线
        # 添加序号（第一个区域序号为1）
        cv2.putText(self.schematic_img, "1", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 白色字体，大小1，厚度2
        
        # 标记已访问的区域
        visited_regions[max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size] = 1
        print(f"[阶段1完成] 找到初始区域: ({max_i_first}, {max_j_first})")
        
        return max_i_first, max_j_first, visited_regions
        
    def find_second_region(self, image_array, max_i_first, max_j_first, visited_regions):
        """
        在第一个区域的周围找到第二个最大NDVI区域
        返回：
            max_i_second: 第二个最大NDVI值的位置的行索引
            max_j_second: 第二个最大NDVI值的位置的列索引
            visited_regions: 用于记录已经访问过的区域的矩阵
        """
        print(f"[阶段2] 开始寻找相邻区域，基于初始区域: ({max_i_first}, {max_j_first})")
        ndwi_band = image_array[-1,:,:]
        height, width = image_array.shape[-2:]
        max_sum_second = -19e9
        max_i_second = max_j_second = -9
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        best_dx = best_dy = 0
        for direction in directions:
            dx, dy = direction
            i = max_i_first + dx * (self.region_size - self.shift_size)
            j = max_j_first + dy * (self.region_size - self.shift_size)
            if i < 0 or i + self.region_size > height or j < 0 or j + self.region_size > width:
                continue
            if visited_regions[i:i+self.region_size, j:j+self.region_size].sum() == self.region_size * self.region_size:
                continue
            region_ndwi = ndwi_band[i:i+self.region_size, j:j+self.region_size]
            region_sum = self.get_region_sum(i, j)  # 替换原sum计算

            if region_sum > max_sum_second:
                max_sum_second = region_sum
                best_dx = dx
                best_dy = dy
                max_i_second = i
                max_j_second = j
        
        if max_i_second != -9:  # 找到有效区域时
            # 记录当前区域坐标并绘制绿色虚线边框
            self.path_coords.append((max_i_second, max_j_second))
            x, y = max_j_second, max_i_second
            w, h = self.region_size, self.region_size
            cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 255, 0), 2, lineType=cv2.LINE_4)  # 绿色虚线
            # 添加序号（当前路径长度即为序号）
            current_index = len(self.path_coords)
            cv2.putText(self.schematic_img, str(current_index), (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 白色字体
            # 绘制红色实线连接前一个区域
            if len(self.path_coords) >= 2:
                prev_i, prev_j = self.path_coords[-2]
                prev_center = (prev_j + w//2, prev_i + h//2)  # 前一个区域中心(x,y)
                current_center = (x + w//2, y + h//2)  # 当前区域中心
                cv2.line(self.schematic_img, prev_center, current_center, (0, 0, 255), 2)  # 红色实线
        
        visited_regions[max_i_second:max_i_second+self.region_size, max_j_second:max_j_second+self.region_size] = 1
        if max_i_second != -9:
            print(f"[阶段2完成] 找到相邻区域: ({max_i_second}, {max_j_second})")
        else:
            print("[阶段2完成] 未找到有效相邻区域")

        return max_i_first, max_j_first, max_i_second, max_j_second, visited_regions, best_dx, best_dy
    
    def save_gdal_image(self, image_array, index):
        driver = gdal.GetDriverByName('PNG')
        driver.CreateCopy(self.image_dir + self.image_name + f"_{index}.png", image_array)
    
    def pre_seg(self, visited_regions, image, annotation, index):
        """
        主要用于在递归分割结束后，对未被访问的区域进行分割和保存。
        """
        num_rows = (self.height - self.region_size) // (self.region_size - self.shift_size) + 1
        num_cols = (self.width - self.region_size) // (self.region_size - self.shift_size) + 1
        
        # 计算剩余区域总数并初始化进度条
        total_regions = num_rows * num_cols
        processed_regions = 0
        pbar = tqdm(total=total_regions, desc='剩余区域处理进度', unit='区域')
        
        for row in range(num_rows):
            for col in range(num_cols):
                start_x = col * (self.region_size - self.shift_size)
                start_y = row * (self.region_size - self.shift_size)
                if visited_regions[start_y:start_y+self.region_size, start_x:start_x+self.region_size].sum() != self.region_size * self.region_size:
                    image_patch = image[:, start_y:start_y+self.region_size, start_x:start_x+self.region_size]
                    annotation_patch = annotation[:, start_y:start_y+self.region_size, start_x:start_x+self.region_size]
                    index += 1
                    best_dx = best_dy = 0
                    self.save_patch(image_patch, self.image_dir, index, best_dx, best_dy)
                    self.save_patch(annotation_patch, self.annotation_dir, index, best_dx, best_dy)
                
                # 更新进度条
                processed_regions += 1
                pbar.update(1)
                pbar.set_postfix({'当前区域': f'({row},{col})', '总进度': f'{processed_regions}/{total_regions}'})
                
        # 关闭进度条
        pbar.close()

    def save_patch(self, patch, path, index, best_dx=0, best_dy=0):
        """
        将图像块保存到指定目录
        """
        filename = self.image_name + "/" + self.image_name + f"_{index}" + f"_{best_dx}_{best_dy}.png"
        save_path = os.path.join(path, filename)
        if path == self.image_dir:
            self.image_list = dict(filename=filename)
            self.image_list['annotation'] = dict(seg_map=filename)
            self.image_info_list.append(self.image_list)
        if os.path.exists(save_path):
            return
        patch = patch.transpose(1, 2, 0)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, patch)
       
    def recursive_segmentation(self, max_i_first, max_j_first, visited_regions, image, annotation, index, prev_max_i_second=None, prev_max_j_second=None):
        """
        递归的调用find_second_region函数，
        通过find_second_region函数找到下一个最大值的坐标，
        并保存图像块。
        """
        if index >= self.max_iterations:
            print("[递归阶段] 达到最大迭代次数，开始处理剩余区域")
            self.pre_seg(visited_regions, image, annotation, index)
            return
        
        # 初始化进度条
        if index == 1:
            self.pbar = tqdm(total=self.max_iterations, desc='分割进度', unit='iter')
        
        # 更新进度条
        self.pbar.update(1)
        self.pbar.set_postfix({'当前迭代': index})

        max_i_first, max_j_first, max_i_second, max_j_second, visited_regions, best_dx, best_dy = self.find_second_region(self.image, max_i_first, max_j_first, visited_regions)

        if max_i_second == -9:
            second_region_rgb = image[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            second_region_annotation = annotation[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            self.save_patch(second_region_rgb, self.image_dir, index, best_dx, best_dy)
            self.save_patch(second_region_annotation, self.annotation_dir, index, best_dx, best_dy)
            print("[递归阶段] 未找到有效相邻区域，开始处理剩余区域")
            self.pre_seg(visited_regions, image, annotation, index)
            return

        if prev_max_i_second is not None:
            second_region_rgb = image[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            second_region_annotation = annotation[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            self.save_patch(second_region_rgb, self.image_dir, index, best_dx, best_dy)
            self.save_patch(second_region_annotation, self.annotation_dir, index, best_dx, best_dy)

        if index == 1:
            first_region_rgb = image[:, max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size]
            first_region_annotation = annotation[:, max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size]
            self.save_patch(first_region_rgb, self.image_dir, index, best_dx, best_dy)
            self.save_patch(first_region_annotation, self.annotation_dir, index, best_dx, best_dy)

        self.recursive_segmentation(max_i_second, max_j_second, visited_regions, image, annotation, index+1, max_i_second, max_j_second)

    def main(self):
        image_rgb = self.image[:3,:,:] 
        annotation = gdal.Open(self.annotation_path)
        annotation = annotation.ReadAsArray()
        annotation = np.expand_dims(annotation, axis=0)
        max_i_first, max_j_first, visited_regions = self.find_first_region(self.image)   
        self.recursive_segmentation(max_i_first, max_j_first, visited_regions, image_rgb, annotation, 1)
        
        # 关闭进度条
        if hasattr(self, 'pbar'):
            self.pbar.close()
        
        # 新增：创建示意图目录（若不存在）
        os.makedirs(self.schematic_dir, exist_ok=True)
        # 新增：生成带有序号的文件名（根据当前目录已有文件数量递增）
        existing_files = [f for f in os.listdir(self.schematic_dir) if f.startswith(f"{self.image_name}_schematic_")]
        schematic_num = len(existing_files) + 1
        schematic_path = os.path.join(self.schematic_dir, f"{self.image_name}_schematic_{schematic_num}.png")
        cv2.imwrite(schematic_path, self.schematic_img)  # 保存示意图到新路径
        
        return self.image_info_list
    
    def calculate_integral_image(self, ndvi):
        """计算积分图像"""
        h, w = ndvi.shape
        integral = np.zeros((h+1, w+1), dtype=np.float32)
        integral[1:, 1:] = ndvi.cumsum(0).cumsum(1)
        return integral

    def get_region_sum(self, i, j):
        """通过积分图像计算区域和"""
        i_end = i + self.region_size
        j_end = j + self.region_size
        return (self.integral_image[i_end, j_end] 
                - self.integral_image[i, j_end] 
                - self.integral_image[i_end, j] 
                + self.integral_image[i, j])
# if __name__ == "__main__":
#     image_path = r"C:\Users\Hi\Downloads\H48F015017_clip1.png"
#     annotation_path = r"C:\Users\Hi\Downloads\H48F015017_clip1_.png"
#     image_info_list = []
    
#     pre_segmentation = SGSW(image_path, annotation_path, image_info_list)
#     result = pre_segmentation.main()
#     print(result)