import torch
from osgeo import gdal
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# 用于进行图像分割, 这个类的输入是图像的路径，标注的路径，图像的信息列表，输出是图像的分割结果`,用NDWI作为指标，用最大和作为选择标准
class SpectralAwareSampling():
    def __init__(self, image_path, annotation_path, image_info_list,
                 ndwi_threshold_k=0.0):
        """
        Constructor method

        参数：
            image_path (str): 输入影像路径（多波段，最后一波段为NDWI）。
            annotation_path (str): 标注掩码路径。
            image_info_list (list): 用于收集输出的图像信息列表。
            ndwi_threshold_k (float): 相对阈值系数，默认 0.0。
                主搜索终止阈值 = mean(所有候选窗口NDWI和) + k * std(...)。
                k=0.0：均值截断，仅让高于平均NDWI水平的窗口进入主搜索段（推荐起点）；
                k>0.0：更严格，仅保留高于均值+k倍标准差的高NDWI区域进入主搜索段；
                k<0.0：更宽松，允许低于均值的窗口也进入主搜索段。
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
        self.ndwi_band = self.image[-1,:,:]  # 提取NDWI波段（最后一波段）
        self.integral_image = self.calculate_integral_image(self.ndwi_band)
        self.path_coords = []  # 记录路径中的区域左上角坐标(i, j)
        # 相对阈值系数（由外部传入）
        self.ndwi_threshold_k = ndwi_threshold_k
        # 主搜索终止阈值：由 find_first_region 遍历所有候选窗口后统计计算，
        # 公式为 mean + k * std（基于全图所有候选窗口的NDWI区域和）。
        # 初始化为 None；find_second_region 在此值为 None 时会跳过阈值判断。
        self.ndwi_threshold = None
        # 量纲检测：自动识别 NDWI 波段的数值范围，计算物理水体分界零点
        # 对应的窗口区域和，用于增强输出诊断。
        self.ndwi_scale, self.ndwi_zero_sum = self._detect_ndwi_scale()
        # 初始化示意图（使用前3波段作为基础图，转换为HWC格式的BGR图像）
        self.schematic_img = self.image[:3, :, :].transpose(1, 2, 0).copy()
        if self.schematic_img.dtype != np.uint8:  # 确保是uint8格式
            self.schematic_img = (self.schematic_img * 255).astype(np.uint8)
        
        # 添加一个检查函数检查image_dir和annotation_dir在本机上是否存在的函数，不存在这报为正确配置presegmentation.py下的image_dir和annotation_dir未配置正确的错误
        self.check_directories()
        
    def check_directories(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.annotation_dir):
            raise RuntimeError("请检查sgsw.py 下的 image_dir 和 annotation_dir 配置是否正确，如果路径正确仍然有错请检查是否正确挂载代码到环境。")

    def _detect_ndwi_scale(self):
        """
        自动检测 NDWI 波段的数值量纲，计算物理水体分界零点对应的窗口区域和。

        NDWI 在不同处理流程下可能以三种量纲存储：
          - [-1, 1]  浮点数：原始物理值，水体分界 NDWI=0，区域和零点 = 0
          - [0, 1]   归一化浮点：线性映射，NDWI=0 对应像素值 0.5，
                     区域和零点 = 0.5 × region_size²
          - [0, 255] 拉伸整数：常见遥感产品存储格式，NDWI=0 对应像素值 127.5，
                     区域和零点 = 127.5 × region_size²

        检测方法：直接读取 NDWI 波段的最小值和最大值，根据数值范围判断量纲。
        这是最直接可靠的方法，不依赖任何分布假设。

        返回：
            scale (str): 量纲标识，'float_signed' / 'float_normalized' / 'uint8'
            zero_sum (float): 物理水体分界零点（NDWI=0）对应的窗口区域和
        """
        ndwi_min = float(self.ndwi_band.min())
        ndwi_max = float(self.ndwi_band.max())
        window_pixels = self.region_size * self.region_size

        if ndwi_min < -0.01:
            # 存在负值 → [-1, 1] 原始浮点量纲
            scale = 'float_signed'
            # NDWI=0 时窗口内所有像素均为 0，区域和为 0
            zero_pixel = 0.0
        elif ndwi_max <= 1.01:
            # 范围在 [0, 1] 内 → 归一化浮点
            # 原始 NDWI=0 经 (x+1)/2 映射后为 0.5
            scale = 'float_normalized'
            zero_pixel = 0.5
        else:
            # 最大值超过 1 → [0, 255] 拉伸整数
            # 原始 NDWI=0 经 (x+1)/2×255 映射后为 127.5
            scale = 'uint8'
            zero_pixel = 127.5

        zero_sum = zero_pixel * window_pixels
        return scale, zero_sum

    def _print_threshold_diagnosis(self, mean_sum, std_sum, all_region_sums):
        """
        打印增强的阈值诊断信息。

        输出内容：
          1. 量纲识别结果（数值范围及判断依据）
          2. 全图候选窗口 NDWI 区域和的统计信息
          3. 当前阈值与物理水体分界零点的相对位置
          4. 超过/低于阈值的窗口数量及占比（即实际进入主搜索段的比例）
          5. 基于上述信息的 k 值调参建议

        参数：
            mean_sum (float): 所有候选窗口 NDWI 区域和的均值
            std_sum  (float): 所有候选窗口 NDWI 区域和的标准差
            all_region_sums (np.ndarray): 所有候选窗口的 NDWI 区域和列表
        """
        threshold = self.ndwi_threshold
        window_pixels = self.region_size * self.region_size
        total_windows = len(all_region_sums)

        # ── 1. 量纲信息 ──────────────────────────────────────────────────────
        scale_desc = {
            'float_signed':    '[-1, 1] 原始浮点（物理NDWI值）',
            'float_normalized': '[0, 1]  归一化浮点（NDWI=0 对应像素值 0.5）',
            'uint8':           '[0, 255] 拉伸整数（NDWI=0 对应像素值 127.5）',
        }[self.ndwi_scale]

        # ── 2. 像素均值（将区域和还原为单像素均值，更直观）────────────────────
        mean_pixel = mean_sum / window_pixels
        threshold_pixel = threshold / window_pixels
        zero_pixel = self.ndwi_zero_sum / window_pixels

        # ── 3. 超过阈值的窗口数量（即将进入主搜索段的窗口数）─────────────────
        above_threshold = int(np.sum(all_region_sums >= threshold))
        above_ratio = above_threshold / total_windows * 100

        # ── 4. threshold 与物理零点的偏差程度（以 std 为单位）────────────────
        if std_sum > 0:
            deviation_sigmas = (threshold - self.ndwi_zero_sum) / std_sum
        else:
            deviation_sigmas = float('inf')

        # ── 5. 调参建议逻辑 ──────────────────────────────────────────────────
        # 核心判断：threshold 相对于物理零点的位置
        #   threshold >> zero_sum → 阈值过高，主搜索段覆盖窗口过少，
        #                           可能漏掉部分水体区域 → 建议减小 k
        #   threshold ≈ zero_sum  → 阈值合理，主搜索段与水体区域基本对齐
        #   threshold << zero_sum → 阈值过低，主搜索段纳入了大量非水体窗口 → 建议增大 k
        if deviation_sigmas > 1.0:
            diagnosis = '⚠ 阈值偏高'
            advice = (f'当前阈值高于物理水体零点 {deviation_sigmas:.1f}σ，'
                      f'主搜索段可能过于保守（仅覆盖 {above_ratio:.1f}% 的窗口），'
                      f'部分水体区域可能被划入补充扫描段，减少 TBTI 训练机会。\n'
                      f'    建议：适当减小 ndwi_threshold_k'
                      f'（如 k={self.ndwi_threshold_k - 0.5:.1f} 或更小）。')
        elif deviation_sigmas < -1.0:
            diagnosis = '⚠ 阈值偏低'
            advice = (f'当前阈值低于物理水体零点 {abs(deviation_sigmas):.1f}σ，'
                      f'主搜索段覆盖了 {above_ratio:.1f}% 的窗口，其中可能包含'
                      f'大量非水体区域，导致 TBTI 在语义断裂处被激活。\n'
                      f'    建议：适当增大 ndwi_threshold_k'
                      f'（如 k={self.ndwi_threshold_k + 0.5:.1f} 或更大）。')
        else:
            diagnosis = '✓ 阈值合理'
            advice = (f'当前阈值与物理水体零点偏差在 1σ 以内（{deviation_sigmas:+.2f}σ），'
                      f'主搜索段覆盖 {above_ratio:.1f}% 的窗口，与水体区域基本对齐，'
                      f'k={self.ndwi_threshold_k} 当前设置合理。')

        # ── 打印 ──────────────────────────────────────────────────────────────
        sep = '─' * 60
        print(f'\n{sep}')
        print(f'[SGSW 阈值诊断报告]')
        print(f'{sep}')
        print(f'  【量纲识别】')
        print(f'    NDWI 波段数值范围: [{self.ndwi_band.min():.3f}, {self.ndwi_band.max():.3f}]')
        print(f'    判断量纲: {scale_desc}')
        print(f'    物理水体分界（NDWI=0）对应像素值: {zero_pixel:.2f}')
        print(f'    物理水体分界对应窗口区域和: {self.ndwi_zero_sum:.2f}')
        print(f'{sep}')
        print(f'  【全图候选窗口 NDWI 统计】')
        print(f'    候选窗口总数: {total_windows}')
        print(f'    窗口区域和  均值 (mean): {mean_sum:.2f}  '
              f'→ 单像素均值: {mean_pixel:.4f}')
        print(f'    窗口区域和  标准差 (std): {std_sum:.2f}')
        print(f'{sep}')
        print(f'  【当前阈值信息】')
        print(f'    k = {self.ndwi_threshold_k}')
        print(f'    threshold = mean + k×std = {threshold:.2f}  '
              f'→ 对应单像素均值: {threshold_pixel:.4f}')
        print(f'    threshold 与物理零点偏差: {deviation_sigmas:+.2f}σ')
        print(f'    高于阈值的窗口数: {above_threshold} / {total_windows} '
              f'({above_ratio:.1f}%)  ← 此比例的窗口将进入主搜索段')
        print(f'{sep}')
        print(f'  【诊断结论】{diagnosis}')
        print(f'    {advice}')
        print(f'{sep}\n')

    def find_first_region(self, image_array):
        """
        通过NDWI波段找到NDWI值最大的位置作为初始点。
        如果多个最大值相同，则随机选择一个最大值作为起点。

        【新增】同步计算全局相对阈值 self.ndwi_threshold：
            在遍历所有候选窗口时顺带收集每个窗口的NDWI区域和，
            全部遍历后计算 mean + k * std，赋值给 self.ndwi_threshold，
            供 find_second_region 作为主搜索终止判断的依据。
            此步骤复用了积分图遍历，不引入额外的时间复杂度。

        返回：
            max_i_first: 最大NDWI值区域的行索引
            max_j_first: 最大NDWI值区域的列索引
            visited_regions: 用于记录已经访问过的区域的矩阵
        """
        print("[阶段1] 开始寻找初始区域并计算全局NDWI阈值...")
        ndwi_band = image_array[-1,:,:]
        visited_regions = np.zeros_like(ndwi_band)
        max_sum_first = -19e9
        max_positions = []  # 用来存储最大NDWI值位置

        # 收集所有候选窗口的NDWI区域和，用于后续统计阈值
        all_region_sums = []

        for i in range(0, self.height - self.region_size + 1, self.region_size - self.shift_size):
            for j in range(0, self.width - self.region_size + 1, self.region_size - self.shift_size):
                region_sum = self.get_region_sum(i, j)
                all_region_sums.append(region_sum)

                if region_sum > max_sum_first:
                    max_sum_first = region_sum
                    max_positions = [(i, j)]  # 重新开始存储位置
                elif region_sum == max_sum_first:
                    max_positions.append((i, j))  # 添加相同NDWI值的位置

        # 计算全局相对阈值：mean + k * std（基于所有候选窗口的NDWI区域和）
        # 该阈值与窗口面积无关，直接对区域和进行统计，便于与 get_region_sum 的返回值直接比较。
        all_region_sums = np.array(all_region_sums, dtype=np.float64)
        mean_sum = float(np.mean(all_region_sums))
        std_sum  = float(np.std(all_region_sums))
        self.ndwi_threshold = mean_sum + self.ndwi_threshold_k * std_sum

        # 打印增强诊断报告（含量纲识别、阈值分析、调参建议）
        self._print_threshold_diagnosis(mean_sum, std_sum, all_region_sums)

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
        在第一个区域的周围找到第二个最大NDWI区域。

        【新增】相对阈值终止判断：
            贪心搜索选出的最优候选窗口（8邻域中NDWI最高者）若其NDWI区域和
            低于 self.ndwi_threshold，则视为"已无高NDWI邻域可扩展"，
            返回 max_i_second=-9（与全部邻域不可达时行为一致），
            触发主搜索提前终止，将剩余区域交由补充扫描段处理。

        返回：
            max_i_second: 第二个最大NDWI值的位置的行索引
            max_j_second: 第二个最大NDWI值的位置的列索引
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
            region_sum = self.get_region_sum(i, j)

            if region_sum > max_sum_second:
                max_sum_second = region_sum
                best_dx = dx
                best_dy = dy
                max_i_second = i
                max_j_second = j

        # ── 相对阈值终止判断（方案二核心逻辑） ────────────────────────────────
        # 仅当阈值已被计算（find_first_region 已执行）且最优候选窗口低于阈值时触发。
        # 此时即使物理邻域尚未全部访问，也认为已无值得继续扩展的高NDWI区域，
        # 主搜索应当终止，后续区域交由补充扫描段（pre_seg）以 D=(0,0) 处理。
        if (max_i_second != -9
                and self.ndwi_threshold is not None
                and max_sum_second < self.ndwi_threshold):
            print(f"[阶段2] 最优候选窗口 NDWI 和 ({max_sum_second:.2f}) "
                  f"低于阈值 ({self.ndwi_threshold:.2f})，主搜索提前终止。")
            max_i_second = max_j_second = -9
            best_dx = best_dy = 0
        # ────────────────────────────────────────────────────────────────────────

        # 在信息最充分时（SAS 阶段），将方向转换为几何边缘标识符，
        # 避免下游模块在方向改变时用 last_dx/last_dy 间接推算而出错。
        ftc_side, recv_side = self._direction_to_sides(best_dx, best_dy)

        if max_i_second != -9:  # 找到有效且高于阈值的区域时
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
            print(f"[阶段2完成] 找到相邻区域: ({max_i_second}, {max_j_second})，"
                  f"ftc_side={ftc_side}，recv_side={recv_side}")
        else:
            print("[阶段2完成] 未找到有效相邻区域")

        return max_i_first, max_j_first, max_i_second, max_j_second, visited_regions, ftc_side, recv_side
    
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
                    # 补充扫描段：无移动关系，ftc_side=-1, recv_side=-1
                    self.save_patch(image_patch, self.image_dir, index, ftc_side=-1, recv_side=-1)
                    self.save_patch(annotation_patch, self.annotation_dir, index, ftc_side=-1, recv_side=-1)
                
                # 更新进度条
                processed_regions += 1
                pbar.update(1)
                pbar.set_postfix({'当前区域': f'({row},{col})', '总进度': f'{processed_regions}/{total_regions}'})
                
        # 关闭进度条
        pbar.close()

    @staticmethod
    def _direction_to_sides(dx, dy):
        """
        将移动方向 (dx, dy) 转换为 (ftc_side, recv_side)。

        ftc_side：当前 patch 应在 FTC 阶段裁剪的边（前沿，即朝向下一个 patch 的那条边）
        recv_side：下一个 patch 应在 FTP 阶段接收注入的边（后沿，即与当前 patch 接触的那条边）
        两者互为对边，由 SAS 在信息最充分时直接计算，避免下游用 dx/dy 间接推算。

        编码：0=底 1=顶 2=右 3=左 4=右下角 5=左下角 6=右上角 7=左上角 -1=无移动
        对边映射：{0↔1, 2↔3, 4↔7, 5↔6}
        """
        _opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, -1: -1}
        _dir_to_ftc = {
            (1,  0): 0,   # 向下前进 → 前沿在底部
            (-1, 0): 1,   # 向上前进 → 前沿在顶部
            (0,  1): 2,   # 向右前进 → 前沿在右侧
            (0, -1): 3,   # 向左前进 → 前沿在左侧
            (1,  1): 4,   # 向右下   → 前沿在右下角
            (1, -1): 5,   # 向左下   → 前沿在左下角
            (-1, 1): 6,   # 向右上   → 前沿在右上角
            (-1,-1): 7,   # 向左上   → 前沿在左上角
            (0,  0): -1,  # 无移动
        }
        ftc_side = _dir_to_ftc.get((dx, dy), -1)
        recv_side = _opposite[ftc_side]
        return ftc_side, recv_side

    def save_patch(self, patch, path, index, ftc_side=-1, recv_side=-1):
        """
        将图像块保存到指定目录。

        文件名格式：<name>/<name>_<id>_<ftc_side>_<recv_side>.png
          ftc_side：本 patch 的 FTC 裁剪边（-1 表示无移动）
          recv_side：本 patch 的 FTP 接收边（-1 表示无移动）
        """
        filename = self.image_name + "/" + self.image_name + f"_{index}" + f"_{ftc_side}_{recv_side}.png"
        save_path = os.path.join(path, filename)
        if path == self.image_dir:
            self.image_list = dict(filename=filename)
            self.image_list['annotation'] = dict(seg_map=filename)
            self.image_info_list.append(self.image_list)
        if os.path.exists(save_path):
            return
        patch = patch.transpose(1, 2, 0)  # (C,H,W) → (H,W,C)，此时通道顺序仍为 GDAL 原始顺序（RGB）
        patch = patch[:, :, ::-1].copy()   # RGB → BGR，与 cv2.imwrite 的写入约定对齐
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, patch)
       
    def recursive_segmentation(self, max_i_first, max_j_first, visited_regions, image, annotation, index, prev_max_i_second=None, prev_max_j_second=None, prev_recv_side=-1):
        """
        递归的调用find_second_region函数，
        通过find_second_region函数找到下一个最大值的坐标，
        并保存图像块。

        prev_recv_side：上一次 find_second_region 返回的 recv_side，
                        即本次要保存的 patch 的 FTP 接收边编码。
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

        max_i_first, max_j_first, max_i_second, max_j_second, visited_regions, ftc_side, recv_side = self.find_second_region(self.image, max_i_first, max_j_first, visited_regions)

        if max_i_second == -9:
            # 无有效邻域：将上一帧位置以 ftc_side=-1/recv_side=prev_recv_side 保存后终止
            second_region_rgb = image[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            second_region_annotation = annotation[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            self.save_patch(second_region_rgb, self.image_dir, index, ftc_side=-1, recv_side=prev_recv_side)
            self.save_patch(second_region_annotation, self.annotation_dir, index, ftc_side=-1, recv_side=prev_recv_side)
            print("[递归阶段] 未找到有效相邻区域，开始处理剩余区域")
            self.pre_seg(visited_regions, image, annotation, index)
            return

        if prev_max_i_second is not None:
            # 保存上一帧 patch：
            #   ftc_side = 本次搜索得到的 ftc_side（上一帧应裁剪的前沿边）
            #   recv_side = 上一次传入的 prev_recv_side（上一帧应接收注入的后沿边）
            second_region_rgb = image[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            second_region_annotation = annotation[:, prev_max_i_second:prev_max_i_second+self.region_size, prev_max_j_second:prev_max_j_second+self.region_size]
            self.save_patch(second_region_rgb, self.image_dir, index, ftc_side=ftc_side, recv_side=prev_recv_side)
            self.save_patch(second_region_annotation, self.annotation_dir, index, ftc_side=ftc_side, recv_side=prev_recv_side)

        if index == 1:
            # 第一个 patch：无前驱，recv_side=-1；ftc_side 由本次搜索决定
            first_region_rgb = image[:, max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size]
            first_region_annotation = annotation[:, max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size]
            self.save_patch(first_region_rgb, self.image_dir, index, ftc_side=ftc_side, recv_side=-1)
            self.save_patch(first_region_annotation, self.annotation_dir, index, ftc_side=ftc_side, recv_side=-1)

        self.recursive_segmentation(max_i_second, max_j_second, visited_regions, image, annotation, index+1, max_i_second, max_j_second, prev_recv_side=recv_side)

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