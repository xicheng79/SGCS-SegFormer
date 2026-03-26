import os
import sys
import math
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import time
import cv2
import fnmatch
import threading
import torch
from mmseg.apis import init_segmentor, inference_segmentor
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

# ----------------- 核心预测功能 -----------------
def GdalData2OpencvData(GdalImg_data):
    if 'int8' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint8)
    elif 'int16' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint16)
    else:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.float32)
    for i in range(GdalImg_data.shape[0]):
        OpencvImg_data[:, :, i] = GdalImg_data[GdalImg_data.shape[0] - i - 1, :, :]
    return OpencvImg_data

class Block:
    def __init__(self, file, idx_row, idx_col, top_overlap, top_overlap_pic, left_overlap, left_overlap_pic, start_x, start_y):
        self.file = file
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.top_overlap = top_overlap
        self.top_overlap_pic = top_overlap_pic
        self.left_overlap = left_overlap
        self.left_overlap_pic = left_overlap_pic
        self.start_x = start_x
        self.start_y = start_y

class MMSegSolver:
    def __init__(self, config_file, model_file):
        self.config_file = config_file
        self.checkpoint_file = model_file
        self.model = None
        self._init_model()
    
    def _init_model(self):
        self.model = init_segmentor(self.config_file, self.checkpoint_file, device='cuda')
    
    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def predict_x(self, img):
        if self.model is None:
            self._init_model()
        return inference_segmentor(self.model, img)[0]

class Predict:
    def __init__(self, target_size, overlap_rate, class_number):
        self.target_size = target_size
        self.overlap_rate = overlap_rate
        self.class_number = class_number
    
    def predict_block(self, gd_img_block, predict, save_path, idx_row, idx_col):
        img_block = GdalData2OpencvData(gd_img_block)
        predict_out = predict(img_block)
        predict_out = np.uint8(predict_out * 255)
        save_file = os.path.join(save_path, f"{idx_row}+{idx_col}.png")
        cv2.imwrite(save_file, predict_out)
        return save_file

    def predict_as_blocks(self, dataset, overlap_rate, predict, save_path, stop_event):
        t0 = time.time()
        img_x = dataset.RasterXSize
        img_y = dataset.RasterYSize
        target_size = self.target_size
        space = target_size - int(target_size*overlap_rate)
        x_num = math.ceil((img_x-target_size)/space)+1
        y_num = math.ceil((img_y-target_size)/space)+1

        dst_pngs = [[None for _ in range(x_num)] for _ in range(y_num)]
        overlap = int(target_size*overlap_rate)

        # 处理主区块
        for j in tqdm(range(y_num-1), desc="分块处理"):
            if stop_event.is_set():
                return None
            for i in range(x_num-1):
                x_start = space*i
                y_start = space*j
                img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
                pic = self.predict_block(img_block, predict, save_path, j, i)
                dst_pngs[j][i] = Block(pic, j, i, overlap, "", overlap, "", x_start, y_start)

        # 处理边缘区块
        if not stop_event.is_set():
            # 下边缘
            cur_y = y_num-1
            y_start = img_y - target_size
            overlap_y = space*(cur_y-1) + target_size - y_start
            for i in tqdm(range(x_num-1), desc="下边缘处理"):
                if stop_event.is_set():
                    break
                x_start = space*i
                img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
                pic = self.predict_block(img_block, predict, save_path, cur_y, i)
                dst_pngs[cur_y][i] = Block(pic, cur_y, i, overlap_y, "", overlap, "", x_start, y_start)

            # 右边缘
            cur_x = x_num-1
            x_start = img_x - target_size
            overlap_x = space*(cur_x-1) + target_size - x_start
            for j in tqdm(range(y_num-1), desc="右边缘处理"):
                if stop_event.is_set():
                    break
                y_start = space*j
                img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
                pic = self.predict_block(img_block, predict, save_path, j, cur_x)
                dst_pngs[j][cur_x] = Block(pic, j, cur_x, overlap, "", overlap_x, "", x_start, y_start)

            # 右下角
            if not stop_event.is_set():
                img_block = dataset.ReadAsArray(img_x-target_size, img_y-target_size, target_size, target_size)
                pic = self.predict_block(img_block, predict, save_path, cur_y, cur_x)
                dst_pngs[cur_y][cur_x] = Block(pic, cur_y, cur_x, overlap_y, "", overlap_x, "", img_x-target_size, img_y-target_size)

        print(f'分块预测耗时: {(time.time() - t0)/60:.2f}分钟')
        return dst_pngs if not stop_event.is_set() else None

    def stitch_by_blocks(self, dst_pngs, dst_ds, stop_event):
        target_size = self.target_size
        for j in tqdm(dst_pngs, desc='拼接分块'):
            if stop_event.is_set():
                break
            for png in j:
                if png is None:
                    continue
                block = cv2.imread(png.file, cv2.IMREAD_GRAYSCALE)
                
                # 处理顶部重叠
                if png.top_overlap > 0 and png.top_overlap_pic:
                    top_block = cv2.imread(png.top_overlap_pic, cv2.IMREAD_GRAYSCALE)
                    overlap = png.top_overlap
                    half_overlap = top_block[target_size-overlap:target_size-int(overlap*0.5), :]
                    block[:overlap-int(overlap*0.5), :] = half_overlap
                
                # 处理左侧重叠
                if png.left_overlap > 0 and png.left_overlap_pic:
                    left_block = cv2.imread(png.left_overlap_pic, cv2.IMREAD_GRAYSCALE)
                    overlap = png.left_overlap
                    half_overlap = left_block[:, target_size-overlap:target_size-int(overlap*0.5)]
                    block[:, :overlap-int(overlap*0.5)] = half_overlap
                
                # 写入结果
                dst_ds.GetRasterBand(1).WriteArray(block, png.start_x, png.start_y)
                cv2.imwrite(png.file, block)
        dst_ds.FlushCache()

# ----------------- GUI界面 -----------------
class PredictGUI:
    def __init__(self, master):
        self.master = master
        master.title("遥感影像预测")
        master.geometry("720x550")
        
        # 初始化状态变量
        self.stop_event = threading.Event()
        self.predict_thread = None
        self.solver = None
        self.current_ds = None
        
        # 初始化UI
        self.create_widgets()
        
        # 绑定窗口关闭事件
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # 输入框架
        input_frame = LabelFrame(self.master, text="数据输入", padx=10, pady=10)
        input_frame.pack(fill="x", padx=15, pady=5)

        # 输入目录
        Label(input_frame, text="影像目录:").grid(row=0, column=0, sticky=W, pady=3)
        self.input_entry = Entry(input_frame, width=45)
        self.input_entry.grid(row=0, column=1, padx=5)
        Button(input_frame, text="浏览", command=self.select_input).grid(row=0, column=2)

        # 输出目录
        Label(input_frame, text="输出目录:").grid(row=1, column=0, sticky=W, pady=3)
        self.output_entry = Entry(input_frame, width=45)
        self.output_entry.grid(row=1, column=1, padx=5)
        Button(input_frame, text="浏览", command=self.select_output).grid(row=1, column=2)

        # 模型配置框架
        model_frame = LabelFrame(self.master, text="模型配置", padx=10, pady=10)
        model_frame.pack(fill="x", padx=15, pady=5)

        Label(model_frame, text="配置文件:").grid(row=0, column=0, sticky=W)
        self.config_entry = Entry(model_frame, width=45)
        self.config_entry.grid(row=0, column=1, padx=5)
        Button(model_frame, text="浏览", command=lambda: self.select_file(self.config_entry)).grid(row=0, column=2)

        Label(model_frame, text="模型权重:").grid(row=1, column=0, sticky=W)
        self.model_entry = Entry(model_frame, width=45)
        self.model_entry.grid(row=1, column=1, padx=5)
        Button(model_frame, text="浏览", command=lambda: self.select_file(self.model_entry)).grid(row=1, column=2)

        # 参数设置框架
        param_frame = LabelFrame(self.master, text="处理参数", padx=10, pady=10)
        param_frame.pack(fill="x", padx=15, pady=5)

        params = [
            ("分块尺寸:", "target_size", "512"),
            ("重叠比例:", "overlap_rate", "0.2"),
            ("类别数量:", "class_num", "2")
        ]

        for idx, (text, name, default) in enumerate(params):
            Label(param_frame, text=text).grid(row=idx, column=0, sticky=W, padx=5, pady=2)
            entry = Entry(param_frame, width=15)
            entry.insert(0, default)
            setattr(self, name, entry)
            entry.grid(row=idx, column=1, sticky=W, padx=5)

        # 控制按钮
        control_frame = Frame(self.master)
        control_frame.pack(pady=10)
        
        self.start_btn = Button(control_frame, text="开始处理", command=self.start_predict, 
                              width=12, bg="#4CAF50", fg="white")
        self.start_btn.pack(side=LEFT, padx=8)
        
        self.stop_btn = Button(control_frame, text="停止处理", command=self.stop_predict,
                             width=12, bg="#f44336", fg="white", state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=8)

        # 进度指示
        self.progress = Progressbar(self.master, orient=HORIZONTAL, length=680, mode='indeterminate')
        self.progress.pack(pady=12)

        # 状态栏
        self.status_bar = Label(self.master, text="就绪", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

    def select_input(self):
        path = filedialog.askdirectory(title="选择输入影像目录")
        self.input_entry.delete(0, END)
        self.input_entry.insert(0, path)

    def select_output(self):
        path = filedialog.askdirectory(title="选择结果输出目录")
        self.output_entry.delete(0, END)
        self.output_entry.insert(0, path)

    def select_file(self, entry):
        path = filedialog.askopenfilename(title="选择文件")
        entry.delete(0, END)
        entry.insert(0, path)

    def validate_inputs(self):
        required = [
            (self.input_entry.get(), "请输入影像输入目录"),
            (self.output_entry.get(), "请选择结果输出目录"),
            (self.config_entry.get(), "请选择模型配置文件"),
            (self.model_entry.get(), "请选择模型权重文件")
        ]

        try:
            params = {
                "target_size": int(self.target_size.get()),
                "overlap_rate": float(self.overlap_rate.get()),
                "class_num": int(self.class_num.get())
            }
            if not (0 < params["overlap_rate"] < 1):
                raise ValueError("重叠比例必须在0到1之间")
        except ValueError as e:
            messagebox.showerror("参数错误", f"无效的参数输入: {str(e)}")
            return False

        for value, msg in required:
            if not value.strip():
                messagebox.showerror("输入错误", msg)
                return False
            if any(part.strip() == '' for part in os.path.split(value)):
                messagebox.showerror("路径错误", "检测到无效的空路径组件")
                return False

        return True

    def start_predict(self):
        if not self.validate_inputs():
            return

        self.stop_event.clear()
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.progress.start()
        self.status_bar.config(text="处理进行中...", fg="green")

        params = {
            "input_dir": self.input_entry.get(),
            "output_dir": self.output_entry.get(),
            "config_file": self.config_entry.get(),
            "model_file": self.model_entry.get(),
            "target_size": int(self.target_size.get()),
            "overlap_rate": float(self.overlap_rate.get()),
            "class_num": int(self.class_num.get())
        }

        self.predict_thread = threading.Thread(target=self.run_prediction, kwargs=params)
        self.predict_thread.start()
        self.check_thread()

    def check_thread(self):
        if self.predict_thread.is_alive():
            self.master.after(100, self.check_thread)
        else:
            self.progress.stop()
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)
            status_text = "处理已取消" if self.stop_event.is_set() else "处理完成"
            self.status_bar.config(text=status_text, fg="red" if self.stop_event.is_set() else "green")

    def stop_predict(self):
        if messagebox.askyesno("确认", "确定要停止当前处理吗？"):
            self.stop_event.set()
            self.cleanup_resources()
            self.status_bar.config(text="正在停止...", fg="orange")

    def on_close(self):
        self.stop_predict()
        if self.predict_thread and self.predict_thread.is_alive():
            self.predict_thread.join(timeout=2)
        self.cleanup_resources()
        self.master.destroy()

    def cleanup_resources(self):
        if self.solver:
            self.solver.release()
            self.solver = None
        if self.current_ds:
            del self.current_ds
            self.current_ds = None

    def run_prediction(self, **kwargs):
        try:
            self.solver = MMSegSolver(kwargs["config_file"], kwargs["model_file"])
            predictor = Predict(
                target_size=kwargs["target_size"],
                overlap_rate=kwargs["overlap_rate"],
                class_number=kwargs["class_num"]
            )

            img_list = fnmatch.filter(os.listdir(kwargs["input_dir"]), "*.tif")
            if not img_list:
                messagebox.showerror("错误", "未找到任何TIFF影像文件")
                return

            total = len(img_list)
            for idx, img_name in enumerate(img_list, 1):
                if self.stop_event.is_set():
                    break

                img_path = os.path.join(kwargs["input_dir"], img_name)
                try:
                    self.current_ds = gdal.Open(img_path)
                    if self.current_ds is None:
                        continue

                    # 更新状态
                    self.status_bar.config(text=f"正在处理 {img_name} ({idx}/{total})...")
                    
                    # 创建输出目录
                    output_subdir = os.path.join(kwargs["output_dir"], os.path.splitext(img_name)[0])
                    os.makedirs(output_subdir, exist_ok=True)

                    # 执行分块预测
                    dst_pngs = predictor.predict_as_blocks(
                        self.current_ds,
                        predictor.overlap_rate,
                        lambda x: self.safe_predict(x),
                        output_subdir,
                        self.stop_event
                    )

                    if dst_pngs is None or self.stop_event.is_set():
                        continue

                    # 构建索引
                    predictor.build_pic_index(dst_pngs)

                    # 创建输出文件
                    driver = gdal.GetDriverByName("GTiff")
                    out_path = os.path.join(kwargs["output_dir"], f"{os.path.splitext(img_name)[0]}_result.tif")
                    dst_ds = driver.Create(out_path, self.current_ds.RasterXSize, 
                                         self.current_ds.RasterYSize, 1, gdal.GDT_Byte)
                    try:
                        dst_ds.SetGeoTransform(self.current_ds.GetGeoTransform())
                        dst_ds.SetProjection(self.current_ds.GetProjection())
                        predictor.stitch_by_blocks(dst_pngs, dst_ds, self.stop_event)
                    finally:
                        dst_ds.FlushCache()
                        del dst_ds

                finally:
                    if self.current_ds:
                        del self.current_ds
                        self.current_ds = None

        except Exception as e:
            messagebox.showerror("错误", f"处理过程中发生异常: {str(e)}")
        finally:
            self.cleanup_resources()
            self.stop_event.set()

    def safe_predict(self, img):
        if self.stop_event.is_set():
            raise KeyboardInterrupt("用户终止操作")
        return self.solver.predict_x(img)

if __name__ == "__main__":
    root = Tk()
    app = PredictGUI(root)
    root.mainloop()
