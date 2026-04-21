# 基于深度优先搜索的光谱感知采样方法设计

## 1. 方法概述

本方法借鉴图的深度优先遍历（DFS）原理，设计一种适合水体窗口遍历的采样策略。核心思想是：从高NDWI种子点出发，沿着水体区域"深入"搜索，直到遇到边界或非水体区域，然后回溯到分叉点继续探索其他方向。

### 1.1 与传统DFS的对应关系

| 图遍历概念 | 本项目对应 |
|-----------|-----------|
| 图的节点 | 512×512的滑动窗口 |
| 节点的邻居 | 窗口的8邻域方向 |
| 访问标记 | visited_regions矩阵 |
| 遍历终止条件 | NDWI阈值 + 标签验证 |
| 回溯栈 | 显式维护的分叉点栈 |

## 2. 算法核心机制

### 2.1 深度优先探索

**原理**：从当前窗口的8邻域中选择NDWI最高且通过验证的邻域，持续"深入"，直到无法继续前进。

**实现**：
```python
def dfs_explore(current_i, current_j, visited_regions, path_stack):
    while True:
        # 收集8邻域候选
        neighbors = get_valid_neighbors(current_i, current_j, visited_regions)
        
        # 按NDWI降序排列
        neighbors.sort(key=lambda x: x.ndwi_sum, reverse=True)
        
        # 选择第一个通过验证的邻域
        next_window = None
        for neighbor in neighbors:
            if validate_window(neighbor):  # NDWI阈值 + 标签验证
                next_window = neighbor
                break
        
        if next_window is None:
            # 无法继续深入，触发回溯
            return
        
        # 保存当前窗口（如果有多个未探索的邻域，压入栈作为分叉点）
        unexplored_neighbors = [n for n in neighbors if n != next_window and validate_window(n)]
        if len(unexplored_neighbors) > 0:
            path_stack.append((current_i, current_j, unexplored_neighbors))
        
        # 移动到下一个窗口
        mark_visited(next_window)
        save_patch(next_window)
        current_i, current_j = next_window.i, next_window.j
```

### 2.2 回溯机制

**原理**：当前路径无法继续时，从栈中弹出最近的分叉点，探索其未访问的分支。

**实现**：
```python
def backtrack(path_stack, visited_regions):
    while len(path_stack) > 0:
        fork_i, fork_j, unexplored = path_stack.pop()
        
        # 过滤掉已被其他路径访问的邻域
        still_valid = [n for n in unexplored if not is_visited(n, visited_regions)]
        
        if len(still_valid) > 0:
            # 找到可探索的分支，从这里重新开始DFS
            return fork_i, fork_j, still_valid
    
    # 栈为空，当前段搜索完成
    return None, None, []
```

### 2.3 容忍机制（跨越非水体区域）

**目的**：连接被道路、裸地等短距离分隔的多块水体区域。

**设计**：
- **容忍窗口计数器**：允许连续通过N个"低置信度"窗口（NDWI略低于阈值但未完全失效）
- **容忍阈值**：`tolerance_threshold`（如0.8 × ndwi_threshold）
- **最大容忍数**：`max_tolerance_steps`（如2-3个窗口）

**实现**：
```python
def validate_window_with_tolerance(neighbor, tolerance_counter, max_tolerance_steps):
    ndwi_sum = neighbor.ndwi_sum
    
    # 强验证：高于主阈值且标签验证通过
    if ndwi_sum >= self.ndwi_threshold:
        if label_validation_pass(neighbor):
            return True, 0  # 重置容忍计数器
    
    # 弱验证：在容忍范围内
    if tolerance_counter < max_tolerance_steps:
        if ndwi_sum >= self.tolerance_threshold:
            # 允许通过，但消耗一次容忍机会
            return True, tolerance_counter + 1
    
    # 验证失败
    return False, tolerance_counter
```

**容忍机制的触发条件**：
1. 当前窗口NDWI在 `[tolerance_threshold, ndwi_threshold)` 区间
2. 容忍计数器未超过 `max_tolerance_steps`
3. 容忍窗口后必须在N步内重新遇到高NDWI水体，否则回溯

## 3. 完整算法流程

### 3.1 单段DFS搜索

```
输入：image, annotation, ndwi_threshold, max_tolerance_steps
输出：visited_regions, saved_patches

1. 初始化：
   - 找到全图NDWI最高且通过标签验证的窗口作为种子点
   - 初始化空栈 path_stack = []
   - 初始化 tolerance_counter = 0

2. DFS主循环：
   current = seed_point
   while True:
       2.1 获取8邻域候选，按NDWI降序排列
       2.2 遍历候选：
           - 尝试强验证（NDWI >= threshold + 标签验证）
           - 若失败，尝试弱验证（容忍机制）
           - 选择第一个通过验证的邻域
       
       2.3 如果找到有效邻域：
           - 检查是否有其他未探索的高质量邻域
           - 若有，将当前位置和未探索邻域压入 path_stack
           - 标记新邻域为已访问
           - 保存patch（带方向标签）
           - 移动到新邻域，继续循环
       
       2.4 如果无有效邻域（触发回溯）：
           - 从 path_stack 弹出分叉点
           - 若栈非空，从分叉点继续DFS
           - 若栈为空，当前段搜索结束

3. 返回 visited_regions
```

### 3.2 多段DFS搜索

```
输入：同上 + max_segments, alpha
输出：visited_regions, saved_patches

1. 计算双阈值：
   - T_expand = ndwi_threshold（段内扩展）
   - T_water = ndwi_zero_sum + alpha × (max_ndwi_sum - ndwi_zero_sum)（新段启动）

2. 多段循环：
   for segment_id in range(max_segments):
       2.1 找到新段ISP（NDWI > T_water 且未访问）
       2.2 若无ISP，退出循环
       2.3 执行单段DFS搜索（使用T_expand作为段内阈值）
       2.4 更新 visited_regions

3. 补充扫描：
   - 遍历所有未访问窗口，保存为补充段（ftc_side=-1, recv_side=-1）
```

## 4. 关键参数说明

| 参数名 | 含义 | 推荐值 | 调参建议 |
|-------|------|--------|---------|
| `ndwi_threshold_k` | 主搜索阈值系数 | 0.0 | k=0为均值截断；k>0更严格；k<0更宽松 |
| `max_tolerance_steps` | 最大容忍窗口数 | 2-3 | 根据典型道路宽度调整（道路宽度/窗口步长） |
| `tolerance_threshold` | 容忍阈值系数 | 0.7-0.8 | 相对于ndwi_threshold的比例 |
| `seed_water_ratio_threshold` | 种子点水体占比 | 0.3-0.5 | 过高可能找不到种子；过低易被道路误导 |
| `path_water_ratio_threshold` | 路径验证水体占比 | 0.2-0.3 | 低于种子阈值，允许边缘区域通过 |
| `alpha` | 新段启动系数 | 0.3-0.5 | 控制多段搜索的激进程度 |

## 5. 与现有实现的对比

### 5.1 现有方法（贪心最优邻域选择）

**特点**：
- 每步选择NDWI最高的邻域
- 无回溯机制
- 容易陷入局部最优（如沿着狭长水体走到尽头后无法返回探索分支）

**示意图**：
```
起点 → A → B → C（死路）
       ↓
       D（未探索的大片水体）
```

### 5.2 DFS方法（本设计）

**特点**：
- 深度优先探索，遇到死路自动回溯
- 显式维护分叉点栈
- 能够系统性地遍历所有连通的水体区域

**示意图**：
```
起点 → A → B → C（死路，回溯到A）
       ↓
       D → E → F（继续探索）
```

### 5.3 优势总结

1. **完整性**：DFS保证遍历所有可达的高NDWI窗口，不会遗漏分支
2. **可解释性**：算法逻辑清晰，对应经典图遍历算法，易于在论文中描述
3. **灵活性**：容忍机制可调，适应不同场景的连通性需求
4. **可扩展性**：栈结构天然支持多段搜索（每段独立DFS）

## 6. 实现建议

### 6.1 数据结构

```python
class WindowNode:
    def __init__(self, i, j, ndwi_sum):
        self.i = i
        self.j = j
        self.ndwi_sum = ndwi_sum
        self.direction = None  # 相对于父节点的方向

class ForkPoint:
    def __init__(self, i, j, unexplored_neighbors):
        self.i = i
        self.j = j
        self.unexplored = unexplored_neighbors  # List[WindowNode]
```

### 6.2 核心函数签名

```python
def dfs_single_segment(self, seed_i, seed_j, visited_regions, 
                       max_tolerance_steps=2, tolerance_ratio=0.8):
    """单段DFS搜索"""
    pass

def dfs_multi_segment(self, max_segments=None, alpha=0.4, 
                      max_tolerance_steps=2, tolerance_ratio=0.8):
    """多段DFS搜索"""
    pass

def _dfs_explore(self, current_i, current_j, visited_regions, 
                 path_stack, tolerance_counter, max_tolerance_steps):
    """DFS探索核心逻辑（递归或迭代实现）"""
    pass

def _backtrack(self, path_stack, visited_regions):
    """回溯到最近的分叉点"""
    pass

def _validate_with_tolerance(self, neighbor, tolerance_counter, max_tolerance_steps):
    """带容忍机制的窗口验证"""
    pass
```

### 6.3 可视化增强

在示意图中用不同颜色标记：
- **绿色实线**：DFS主路径
- **黄色虚线**：回溯路径
- **橙色边框**：分叉点
- **蓝色边框**：容忍窗口（跨越非水体）

## 7. 论文描述建议

### 7.1 方法命名

**DFS-SGSW**（Depth-First Search based Spectral-Guided Sliding Window）

### 7.2 核心贡献点

1. **图遍历视角**：将窗口采样问题形式化为图遍历，引入DFS保证完整性
2. **容忍机制**：设计自适应容忍策略，连接被短距离分隔的水体区域
3. **双阈值策略**：T_expand（段内）+ T_water（段间），平衡覆盖率与精度

### 7.3 算法复杂度分析

- **时间复杂度**：O(N)，其中N为候选窗口总数（每个窗口最多访问一次）
- **空间复杂度**：O(D)，其中D为最大搜索深度（栈空间）
- **相比贪心方法**：时间复杂度相同，但空间增加O(D)用于回溯栈

## 8. 后续优化方向

1. **优先级队列优化**：在回溯时优先选择NDWI更高的分叉点
2. **自适应容忍**：根据局部NDWI方差动态调整容忍阈值
3. **并行DFS**：多段搜索时并行启动多个DFS线程
4. **增量更新**：支持在线添加新影像时增量更新visited_regions

---

**设计完成时间**：2026-04-19  
**版本**：v1.0  
**状态**：待实现
