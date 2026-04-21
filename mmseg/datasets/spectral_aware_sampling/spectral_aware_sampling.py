import torch
from osgeo import gdal
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


class WindowNode:
    """窗口节点，用于DFS遍历"""
    def __init__(self, i, j, ndwi_sum, direction=None):
        self.i = i
        self.j = j
        self.ndwi_sum = ndwi_sum
        self.direction = direction


class ForkPoint:
    """分叉点，用于回溯"""
    def __init__(self, i, j, unexplored_neighbors):
        self.i = i
        self.j = j
        self.unexplored = unexplored_neighbors


class SpectralAwareSampling:
    def __init__(self, image_path, annotation_path, image_info_list,
                 ndwi_threshold_k=0.0,
                 label_validation_enabled=True,
                 seed_water_ratio_threshold=0.3,
                 new_isp_water_ratio_threshold=0.2,
                 path_water_ratio_threshold=0.2,
                 top_k_candidates=100,
                 max_tolerance_steps=2,
                 tolerance_ratio=0.75):
        """
        基于深度优先搜索的光谱感知采样

        参数：
            image_path (str): 输入影像路径（多波段，最后一波段为NDWI）
            annotation_path (str): 标注掩码路径
            image_info_list (list): 用于收集输出的图像信息列表
            ndwi_threshold_k (float): 相对阈值系数，默认 0.0
            label_validation_enabled (bool): 是否启用标注验证，默认 True
            seed_water_ratio_threshold (float): 种子点水体占比阈值，默认 0.3
            new_isp_water_ratio_threshold (float): 新段ISP水体占比阈值，默认 0.2
            path_water_ratio_threshold (float): 路径水体占比阈值，默认 0.2
            top_k_candidates (int): Top-K 候选数量
            max_tolerance_steps (int): 最大容忍窗口数，默认 2
            tolerance_ratio (float): 容忍阈值系数（相对于ndwi_threshold），默认 0.75
        """
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.image = gdal.Open(image_path).ReadAsArray()
        self.height, self.width = self.image.shape[-2:]
        self.num_bands = self.image.shape[0]
        self.shift_size = 128
        self.region_size = 512
        self.image_dir = r'D:\dataset\water_ndwi_3968_0907\JPEGImages'
        self.annotation_dir = r'D:\dataset\water_ndwi_3968_0907\SegmentationClass'
        self.schematic_dir = os.path.join(self.image_dir, 'schematic_dfs4')
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.image_list = []
        self.image_info_list = image_info_list
        self.max_iterations = 100
        self.ndwi_band = self.image[-1, :, :]
        self.integral_image = self.calculate_integral_image(self.ndwi_band)
        self.path_coords = []

        self.ndwi_threshold_k = ndwi_threshold_k
        self.label_validation_enabled = label_validation_enabled
        self.seed_water_ratio_threshold = seed_water_ratio_threshold
        self.new_isp_water_ratio_threshold = new_isp_water_ratio_threshold
        self.path_water_ratio_threshold = path_water_ratio_threshold
        self.top_k_candidates = top_k_candidates

        self.max_tolerance_steps = max_tolerance_steps
        self.tolerance_ratio = tolerance_ratio

        self.ndwi_threshold = None
        self.tolerance_threshold = None
        self.ndwi_scale, self.ndwi_zero_sum = self._detect_ndwi_scale()

        self.schematic_img = self.image[:3, :, :].transpose(1, 2, 0).copy()
        if self.schematic_img.dtype != np.uint8:
            self.schematic_img = (self.schematic_img * 255).astype(np.uint8)

        self.saved_coords = set()

        self.check_directories()

    def check_directories(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.annotation_dir):
            raise RuntimeError("请检查dfs_sgsw.py下的image_dir和annotation_dir配置是否正确")

    def _detect_ndwi_scale(self):
        ndwi_min = float(self.ndwi_band.min())
        ndwi_max = float(self.ndwi_band.max())
        window_pixels = self.region_size * self.region_size

        if ndwi_min < -0.01:
            scale = 'float_signed'
            zero_pixel = 0.0
        elif ndwi_max <= 1.01:
            scale = 'float_normalized'
            zero_pixel = 0.5
        else:
            scale = 'uint8'
            zero_pixel = 127.5

        zero_sum = zero_pixel * window_pixels
        return scale, zero_sum

    def _validate_seed_with_label(self, i, j, water_ratio_threshold=None):
        if not self.label_validation_enabled:
            return True, 1.0

        if water_ratio_threshold is None:
            water_ratio_threshold = self.seed_water_ratio_threshold

        if not hasattr(self, '_annotation_array'):
            try:
                annotation_ds = gdal.Open(self.annotation_path)
                if annotation_ds is None:
                    print(f"[警告] 无法打开标注文件: {self.annotation_path}, 禁用标注验证")
                    self.label_validation_enabled = False
                    return True, 1.0

                self._annotation_array = annotation_ds.ReadAsArray()
                if len(self._annotation_array.shape) == 3:
                    self._annotation_array = self._annotation_array[0]

                if self._annotation_array.max() > 1:
                    self._annotation_array = (self._annotation_array > 0).astype(np.float32)

            except Exception as e:
                print(f"[警告] 标注数据加载失败: {e}, 禁用标注验证")
                self.label_validation_enabled = False
                return True, 1.0

        window_label = self._annotation_array[
            i:i+self.region_size,
            j:j+self.region_size
        ]

        water_ratio = float(window_label.mean())
        is_valid = water_ratio >= water_ratio_threshold

        return is_valid, water_ratio

    def calculate_integral_image(self, ndvi):
        h, w = ndvi.shape
        integral = np.zeros((h+1, w+1), dtype=np.float32)
        integral[1:, 1:] = ndvi.cumsum(0).cumsum(1)
        return integral

    def get_region_sum(self, i, j):
        i_end = i + self.region_size
        j_end = j + self.region_size
        return (self.integral_image[i_end, j_end]
                - self.integral_image[i, j_end]
                - self.integral_image[i_end, j]
                + self.integral_image[i, j])

    def find_first_region(self, image_array):
        print("[DFS阶段1] 开始寻找初始区域并计算全局NDWI阈值...")
        ndwi_band = image_array[-1, :, :]
        visited_regions = np.zeros_like(ndwi_band)

        all_region_sums = []
        candidates = []

        for i in range(0, self.height - self.region_size + 1, self.region_size - self.shift_size):
            for j in range(0, self.width - self.region_size + 1, self.region_size - self.shift_size):
                region_sum = self.get_region_sum(i, j)
                all_region_sums.append(region_sum)
                candidates.append((i, j, region_sum))

        all_region_sums = np.array(all_region_sums, dtype=np.float64)
        mean_sum = float(np.mean(all_region_sums))
        std_sum = float(np.std(all_region_sums))
        self.ndwi_threshold = mean_sum + self.ndwi_threshold_k * std_sum
        self.tolerance_threshold = self.ndwi_threshold * self.tolerance_ratio

        candidates.sort(key=lambda x: x[2], reverse=True)
        self.max_ndwi_sum = candidates[0][2] if candidates else 0

        print(f"[DFS阶段1] NDWI阈值: {self.ndwi_threshold:.2f}, 容忍阈值: {self.tolerance_threshold:.2f}")

        validated_isp = None
        for candidate_i, candidate_j, candidate_sum in candidates[:self.top_k_candidates]:
            is_valid, water_ratio = self._validate_seed_with_label(
                candidate_i, candidate_j,
                water_ratio_threshold=self.seed_water_ratio_threshold
            )
            if is_valid:
                validated_isp = (candidate_i, candidate_j)
                print(f"[DFS阶段1] 标注验证通过: ({candidate_i}, {candidate_j}), 水体占比={water_ratio:.2%}")
                break

        if validated_isp is None:
            print(f"[DFS阶段1] ⚠ 警告:所有Top-K候选均未通过标注验证,放弃主搜索,全部交给补充扫描")
            return None, None, visited_regions

        max_i_first, max_j_first = validated_isp
        self.path_coords.append((max_i_first, max_j_first))
        x, y = max_j_first, max_i_first
        w, h = self.region_size, self.region_size
        cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(self.schematic_img, "1", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        visited_regions[max_i_first:max_i_first+self.region_size, max_j_first:max_j_first+self.region_size] = 1
        print(f"[DFS阶段1完成] 找到初始区域: ({max_i_first}, {max_j_first})")

        return max_i_first, max_j_first, visited_regions

    def _get_valid_neighbors(self, current_i, current_j, visited_regions):
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        neighbors = []

        for dx, dy in directions:
            i = current_i + dx * (self.region_size - self.shift_size)
            j = current_j + dy * (self.region_size - self.shift_size)

            if i < 0 or i + self.region_size > self.height or j < 0 or j + self.region_size > self.width:
                continue
            if visited_regions[i:i+self.region_size, j:j+self.region_size].sum() == self.region_size * self.region_size:
                continue

            region_sum = self.get_region_sum(i, j)
            neighbors.append(WindowNode(i, j, region_sum, (dx, dy)))

        neighbors.sort(key=lambda x: x.ndwi_sum, reverse=True)
        return neighbors

    def _validate_with_tolerance(self, neighbor, tolerance_counter):
        ndwi_sum = neighbor.ndwi_sum

        if ndwi_sum >= self.ndwi_threshold:
            is_valid, water_ratio = self._validate_seed_with_label(
                neighbor.i, neighbor.j,
                water_ratio_threshold=self.path_water_ratio_threshold
            )
            if is_valid:
                return True, 0, 'strong'

        if tolerance_counter < self.max_tolerance_steps:
            if ndwi_sum >= self.tolerance_threshold:
                return True, tolerance_counter + 1, 'tolerance'

        return False, tolerance_counter, 'rejected'

    @staticmethod
    def _direction_to_sides(dx, dy):
        _opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, -1: -1}
        _dir_to_ftc = {
            (1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3,
            (1, 1): 4, (1, -1): 5, (-1, 1): 6, (-1, -1): 7,
            (0, 0): -1,
        }
        ftc_side = _dir_to_ftc.get((dx, dy), -1)
        recv_side = _opposite[ftc_side]
        return ftc_side, recv_side

    def _dfs_explore(self, current_i, current_j, visited_regions, path_stack,
                     image, annotation, index, prev_recv_side=-1, is_segment_start=False):
        tolerance_counter = 0
        tolerance_path = []

        if not hasattr(self, '_failed_tolerance_paths'):
            self._failed_tolerance_paths = set()

        while index <= self.max_iterations:
            neighbors = self._get_valid_neighbors(current_i, current_j, visited_regions)

            next_window = None
            unexplored_high_quality = []
            validation_type = None

            for neighbor in neighbors:
                is_valid, new_tolerance_counter, v_type = self._validate_with_tolerance(
                    neighbor, tolerance_counter
                )

                if is_valid:
                    if next_window is None:
                        next_window = neighbor
                        tolerance_counter = new_tolerance_counter
                        validation_type = v_type
                    elif v_type == 'strong':
                        unexplored_high_quality.append(neighbor)

            if next_window is None:
                if len(tolerance_path) > 0:
                    print(f"[DFS容忍] 容忍路径失败（末尾），将{len(tolerance_path)+1}个窗口保存为补充段")
                    for item in tolerance_path:
                        patch_rgb = image[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                        if (item['i'], item['j']) not in self.saved_coords:
                            patch_rgb = image[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                            patch_ann = annotation[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                            self.save_patch(patch_rgb, self.image_dir, index, ftc_side=-1, recv_side=-1)
                            self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=-1, recv_side=-1)
                            self.saved_coords.add((item['i'], item['j']))
                            index += 1
                        self._failed_tolerance_paths.add((item['next_i'], item['next_j']))
                    
                    if (current_i, current_j) not in self.saved_coords:
                        patch_rgb = image[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                        patch_ann = annotation[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                        self.save_patch(patch_rgb, self.image_dir, index, ftc_side=-1, recv_side=-1)
                        self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=-1, recv_side=-1)
                        self.saved_coords.add((current_i, current_j))
                        index += 1
                    tolerance_path = []
                else:
                    if (current_i, current_j) not in self.saved_coords:
                        patch_rgb = image[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                        patch_ann = annotation[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                        r_side = -1 if is_segment_start else prev_recv_side
                        self.save_patch(patch_rgb, self.image_dir, index, ftc_side=-1, recv_side=r_side)
                        self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=-1, recv_side=r_side)
                        self.saved_coords.add((current_i, current_j))
                        index += 1
                    if is_segment_start:
                        print(f"[DFS] 孤立水坑: ({current_i}, {current_j}) 已保存")

                return index, visited_regions

            if len(unexplored_high_quality) > 0:
                path_stack.append(ForkPoint(current_i, current_j, unexplored_high_quality))

            dx, dy = next_window.direction
            ftc_side, recv_side = self._direction_to_sides(dx, dy)

            if validation_type == 'tolerance':
                tolerance_path.append({
                    'i': current_i,
                    'j': current_j,
                    'ftc_side': ftc_side,
                    'recv_side': -1 if is_segment_start else prev_recv_side,
                    'next_i': next_window.i,
                    'next_j': next_window.j
                })
            elif validation_type == 'strong':
                if len(tolerance_path) > 0:
                    print(f"[DFS容忍] 容忍路径成功，保存{len(tolerance_path)}个容忍窗口")
                    for item in tolerance_path:
                        if (item['i'], item['j']) not in self.saved_coords:
                            patch_rgb = image[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                            patch_ann = annotation[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                            self.save_patch(patch_rgb, self.image_dir, index, ftc_side=item['ftc_side'], recv_side=item['recv_side'])
                            self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=item['ftc_side'], recv_side=item['recv_side'])
                            self.saved_coords.add((item['i'], item['j']))
                            index += 1

                        # 延迟绘制验证成功的容忍窗口
                        tol_i, tol_j = item['next_i'], item['next_j']
                        self.path_coords.append((tol_i, tol_j))
                        x, y = tol_j, tol_i
                        w, h = self.region_size, self.region_size
                        cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                        cv2.putText(self.schematic_img, str(len(self.path_coords)),
                                   (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        prev_center = (item['j'] + w//2, item['i'] + h//2)
                        current_center = (x + w//2, y + h//2)
                        cv2.line(self.schematic_img, prev_center, current_center, (0, 0, 255), 2)

                    tolerance_path = []
                    tolerance_counter = 0

                if (current_i, current_j) not in self.saved_coords:
                    patch_rgb = image[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                    patch_ann = annotation[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
                    r_side = -1 if is_segment_start else prev_recv_side
                    self.save_patch(patch_rgb, self.image_dir, index, ftc_side=ftc_side, recv_side=r_side)
                    self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=ftc_side, recv_side=r_side)
                    self.saved_coords.add((current_i, current_j))
                    index += 1

            visited_regions[next_window.i:next_window.i+self.region_size, next_window.j:next_window.j+self.region_size] = 1

            if validation_type == 'strong':
                self.path_coords.append((next_window.i, next_window.j))
                x, y = next_window.j, next_window.i
                w, h = self.region_size, self.region_size
                cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(self.schematic_img, str(len(self.path_coords)),
                           (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 使用物理父子节点进行连线
                prev_center = (current_j + w//2, current_i + h//2)
                current_center = (x + w//2, y + h//2)
                cv2.line(self.schematic_img, prev_center, current_center, (0, 0, 255), 2)

            current_i, current_j = next_window.i, next_window.j
            prev_recv_side = recv_side
            is_segment_start = False

        if len(tolerance_path) > 0:
            for item in tolerance_path:
                if (item['i'], item['j']) not in self.saved_coords:
                    patch_rgb = image[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                    patch_ann = annotation[:, item['i']:item['i']+self.region_size, item['j']:item['j']+self.region_size]
                    self.save_patch(patch_rgb, self.image_dir, index, ftc_side=-1, recv_side=-1)
                    self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=-1, recv_side=-1)
                    self.saved_coords.add((item['i'], item['j']))
                    index += 1
                self._failed_tolerance_paths.add((item['next_i'], item['next_j']))
            tolerance_path = []

        if (current_i, current_j) not in self.saved_coords:
            patch_rgb = image[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
            patch_ann = annotation[:, current_i:current_i+self.region_size, current_j:current_j+self.region_size]
            r_side = -1 if is_segment_start else prev_recv_side
            self.save_patch(patch_rgb, self.image_dir, index, ftc_side=-1, recv_side=r_side)
            self.save_patch(patch_ann, self.annotation_dir, index, ftc_side=-1, recv_side=r_side)
            self.saved_coords.add((current_i, current_j))
            index += 1

        return index, visited_regions

    def _backtrack(self, path_stack, visited_regions, image, annotation, index):
        while len(path_stack) > 0:
            fork_point = path_stack.pop()

            # 过滤掉已访问或在失败黑名单中的窗口
            still_valid = [
                n for n in fork_point.unexplored
                if (visited_regions[n.i:n.i+self.region_size, n.j:n.j+self.region_size].sum()
                    < self.region_size * self.region_size
                    and (n.i, n.j) not in getattr(self, '_failed_tolerance_paths', set()))
            ]

            if len(still_valid) > 0:
                print(f"[DFS回溯] 从分叉点({fork_point.i}, {fork_point.j})继续探索")

                # 不重新添加到path_coords，因为分叉点已经在第一次探索时添加过了
                # 只在示意图上标记分叉点
                x, y = fork_point.j, fork_point.i
                w, h = self.region_size, self.region_size
                cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 165, 255), 3)

                return fork_point.i, fork_point.j, still_valid, index

        return None, None, [], index

    def dfs_single_segment(self):
        image_rgb = self.image[:3, :, :]
        annotation = gdal.Open(self.annotation_path)
        annotation = annotation.ReadAsArray()
        annotation = np.expand_dims(annotation, axis=0)

        max_i, max_j, visited_regions = self.find_first_region(self.image)

        # 如果没有找到有效起点，直接进入补充扫描
        if max_i is None:
            print("[DFS] 未找到有效起点，跳过主搜索，直接进入补充扫描")
            self.pre_seg(visited_regions, image_rgb, annotation, 1)
            return

        path_stack = []
        index = 1  # 从1开始，匹配greenland.py的file_idx逻辑

        print("[DFS] 开始深度优先搜索...")
        index, visited_regions = self._dfs_explore(
            max_i, max_j, visited_regions, path_stack,
            image_rgb, annotation, index, prev_recv_side=-1, is_segment_start=True
        )

        while True:
            fork_i, fork_j, unexplored, index = self._backtrack(
                path_stack, visited_regions, image_rgb, annotation, index
            )

            if fork_i is None:
                print("[DFS] 所有分支探索完成")
                break

            index, visited_regions = self._dfs_explore(
                fork_i, fork_j, visited_regions, path_stack,
                image_rgb, annotation, index, prev_recv_side=-1
            )

        self.pre_seg(visited_regions, image_rgb, annotation, index)

    def _find_new_isp(self, visited_regions, t_water):
        candidates = []
        for i in range(0, self.height - self.region_size + 1, self.region_size - self.shift_size):
            for j in range(0, self.width - self.region_size + 1, self.region_size - self.shift_size):
                if visited_regions[i:i+self.region_size, j:j+self.region_size].sum() == self.region_size * self.region_size:
                    continue
                region_sum = self.get_region_sum(i, j)
                if region_sum > t_water:
                    candidates.append((i, j, region_sum))

        if not candidates:
            return -9, -9

        candidates.sort(key=lambda x: x[2], reverse=True)

        validated_isp = None
        for candidate_i, candidate_j, candidate_sum in candidates[:self.top_k_candidates]:
            is_valid, water_ratio = self._validate_seed_with_label(
                candidate_i, candidate_j,
                water_ratio_threshold=self.new_isp_water_ratio_threshold
            )
            if is_valid:
                validated_isp = (candidate_i, candidate_j)
                print(f"[DFS多段] 新ISP验证通过: ({candidate_i}, {candidate_j}), 水体占比={water_ratio:.2%}")
                break

        if validated_isp is None:
            print(f"[DFS多段] ⚠ 所有候选均未通过验证，终止多段搜索")
            return -9, -9

        return validated_isp

    def dfs_multi_segment(self, max_segments=None, alpha=0):
        image_rgb = self.image[:3, :, :]
        annotation = gdal.Open(self.annotation_path)
        annotation = annotation.ReadAsArray()
        annotation = np.expand_dims(annotation, axis=0)

        max_i, max_j, visited_regions = self.find_first_region(self.image)

        # 如果没有找到有效起点，直接进入补充扫描
        if max_i is None:
            print("[DFS多段] 未找到有效起点，跳过主搜索，直接进入补充扫描")
            self.pre_seg(visited_regions, image_rgb, annotation, 1)
            return

        t_water = self.ndwi_zero_sum + alpha * (self.max_ndwi_sum - self.ndwi_zero_sum)
        window_pixels = self.region_size * self.region_size
        t_water_pixel = t_water / window_pixels
        print(f"[DFS多段] T_expand={self.ndwi_threshold:.2f}, T_water={t_water:.2f}")

        segment_count = 0
        index = 1  # 从1开始，匹配greenland.py的file_idx逻辑

        while True:
            segment_count += 1
            print(f"\n[DFS多段] ═══ 第{segment_count}段 ═══ ISP=({max_i}, {max_j})")

            path_stack = []
            is_first_segment = (segment_count == 1)

            index, visited_regions = self._dfs_explore(
                max_i, max_j, visited_regions, path_stack,
                image_rgb, annotation, index, prev_recv_side=-1,
                is_segment_start=(not is_first_segment)
            )

            while True:
                fork_i, fork_j, unexplored, index = self._backtrack(
                    path_stack, visited_regions, image_rgb, annotation, index
                )

                if fork_i is None:
                    break

                index, visited_regions = self._dfs_explore(
                    fork_i, fork_j, visited_regions, path_stack,
                    image_rgb, annotation, index, prev_recv_side=-1
                )

            if max_segments is not None and segment_count >= max_segments:
                print(f"[DFS多段] 达到最大段数{max_segments}")
                break

            new_i, new_j = self._find_new_isp(visited_regions, t_water)
            if new_i == -9:
                print("[DFS多段] 全图覆盖完成")
                break

            visited_regions[new_i:new_i+self.region_size, new_j:new_j+self.region_size] = 1
            self.path_coords.append((new_i, new_j))
            x, y = new_j, new_i
            w, h = self.region_size, self.region_size
            cv2.rectangle(self.schematic_img, (x, y), (x+w, y+h), (0, 165, 255), 3)
            cv2.putText(self.schematic_img, f'S{segment_count+1}', (x + 10, y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)

            max_i, max_j = new_i, new_j

        self.pre_seg(visited_regions, image_rgb, annotation, index)

    def pre_seg(self, visited_regions, image, annotation, index):
        num_rows = (self.height - self.region_size) // (self.region_size - self.shift_size) + 1
        num_cols = (self.width - self.region_size) // (self.region_size - self.shift_size) + 1

        total_regions = num_rows * num_cols
        pbar = tqdm(total=total_regions, desc='补充扫描', unit='区域')

        for row in range(num_rows):
            for col in range(num_cols):
                start_x = col * (self.region_size - self.shift_size)
                start_y = row * (self.region_size - self.shift_size)
                if (start_y, start_x) not in self.saved_coords:
                    image_patch = image[:, start_y:start_y+self.region_size, start_x:start_x+self.region_size]
                    annotation_patch = annotation[:, start_y:start_y+self.region_size, start_x:start_x+self.region_size]
                    self.save_patch(image_patch, self.image_dir, index, ftc_side=-1, recv_side=-1)
                    self.save_patch(annotation_patch, self.annotation_dir, index, ftc_side=-1, recv_side=-1)
                    self.saved_coords.add((start_y, start_x))
                    index += 1

                pbar.update(1)

        pbar.close()

    def save_patch(self, patch, path, index, ftc_side=-1, recv_side=-1):
        filename = self.image_name + "/" + self.image_name + f"_{index}" + f"_{ftc_side}_{recv_side}.png"
        save_path = os.path.join(path, filename)
        # 规范化路径，确保使用正确的分隔符
        save_path = os.path.normpath(save_path)

        if path == self.image_dir:
            self.image_list = dict(filename=filename)
            self.image_list['annotation'] = dict(seg_map=filename)
            self.image_info_list.append(self.image_list)
        if os.path.exists(save_path):
            return

        patch = patch.transpose(1, 2, 0)
        patch = patch[:, :, ::-1].copy()
        # 确保数据类型为uint8
        if patch.dtype != np.uint8:
            if patch.max() <= 1.0:
                patch = (patch * 255).astype(np.uint8)
            else:
                patch = patch.astype(np.uint8)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, patch)

    def main(self, method='multi_segment', max_segments=2, alpha=0):
        """
        DFS-SGSW 主入口

        参数：
            method (str): 'single_segment' 或 'multi_segment'
            max_segments (int or None): 最大段数
            alpha (float): T_water系数
        """
        if method == 'single_segment':
            self.dfs_single_segment()
        elif method == 'multi_segment':
            self.dfs_multi_segment(max_segments=max_segments, alpha=alpha)
        else:
            raise ValueError(f"未知方法: {method}")

        os.makedirs(self.schematic_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(self.schematic_dir)
                         if f.startswith(f"{self.image_name}_dfs_schematic_")]
        schematic_num = len(existing_files) + 1
        schematic_path = os.path.join(self.schematic_dir,
                                     f"{self.image_name}_dfs_schematic_{schematic_num}.png")
        print(f"[DFS] 路径保存: {schematic_path}")
        cv2.imwrite(schematic_path, self.schematic_img)
        print(f"[DFS] 示意图已保存: {schematic_path}")

        return self.image_info_list
