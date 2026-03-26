# -------------------------
# tkinter GUI 封装
# -------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

class SegmentationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("遥感影像分割预测")
        self.geometry("600x400")
        self.create_widgets()

    def create_widgets(self):
        padx = 5
        pady = 5

        tk.Label(self, text="待预测影像文件夹:").grid(row=0, column=0, sticky="e", padx=padx, pady=pady)
        self.predictImgPath_entry = tk.Entry(self, width=50)
        self.predictImgPath_entry.grid(row=0, column=1, padx=padx, pady=pady)
        tk.Button(self, text="选择文件夹", command=self.choose_predict_folder).grid(row=0, column=2, padx=padx, pady=pady)

        tk.Label(self, text="输出结果路径:").grid(row=1, column=0, sticky="e", padx=padx, pady=pady)
        self.output_path_entry = tk.Entry(self, width=50)
        self.output_path_entry.grid(row=1, column=1, padx=padx, pady=pady)
        tk.Button(self, text="选择文件夹", command=self.choose_output_folder).grid(row=1, column=2, padx=padx, pady=pady)

        tk.Label(self, text="模型配置文件:").grid(row=2, column=0, sticky="e", padx=padx, pady=pady)
        self.config_file_entry = tk.Entry(self, width=50)
        self.config_file_entry.grid(row=2, column=1, padx=padx, pady=pady)
        tk.Button(self, text="选择文件", command=self.choose_config_file).grid(row=2, column=2, padx=padx, pady=pady)

        tk.Label(self, text="模型文件:").grid(row=3, column=0, sticky="e", padx=padx, pady=pady)
        self.model_file_entry = tk.Entry(self, width=50)
        self.model_file_entry.grid(row=3, column=1, padx=padx, pady=pady)
        tk.Button(self, text="选择文件", command=self.choose_model_file).grid(row=3, column=2, padx=padx, pady=pady)

        tk.Label(self, text="重叠率 (默认0.2):").grid(row=4, column=0, sticky="e", padx=padx, pady=pady)
        self.overlap_rate_entry = tk.Entry(self, width=10)
        self.overlap_rate_entry.insert(0, "0.2")
        self.overlap_rate_entry.grid(row=4, column=1, sticky="w", padx=padx, pady=pady)

        tk.Label(self, text="目标尺寸 (默认512):").grid(row=5, column=0, sticky="e", padx=padx, pady=pady)
        self.target_size_entry = tk.Entry(self, width=10)
        self.target_size_entry.insert(0, "512")
        self.target_size_entry.grid(row=5, column=1, sticky="w", padx=padx, pady=pady)

        self.status_label = tk.Label(self, text="状态：就绪", fg="green")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)

        self.start_button = tk.Button(self, text="开始预测", command=self.start_prediction_thread)
        self.start_button.grid(row=7, column=1, pady=10)

    def choose_predict_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.predictImgPath_entry.delete(0, tk.END)
            self.predictImgPath_entry.insert(0, path)

    def choose_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, path)

    def choose_config_file(self):
        path = filedialog.askopenfilename(filetypes=[("Config files", "*.py *.json *.yaml *.yml")])
        if path:
            self.config_file_entry.delete(0, tk.END)
            self.config_file_entry.insert(0, path)

    def choose_model_file(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth *.pt")])
        if path:
            self.model_file_entry.delete(0, tk.END)
            self.model_file_entry.insert(0, path)

    def start_prediction_thread(self):
        self.status_label.config(text="状态：正在预测...", fg="orange")
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        try:
            from overlap_predict_mm0x import MMSegSolver, Predict  # 根据你预测代码文件的名字修改

            predict_folder = self.predictImgPath_entry.get()
            output_path = self.output_path_entry.get()
            config_file = self.config_file_entry.get()
            model_file = self.model_file_entry.get()
            overlap_rate = float(self.overlap_rate_entry.get())
            target_size = int(self.target_size_entry.get())

            all_files = [os.path.join(predict_folder, f) for f in os.listdir(predict_folder) if f.endswith(".tif")]
            solver = MMSegSolver(config_file=config_file, model_file=model_file)
            predictor = Predict(target_size=target_size, overlap_rate=overlap_rate, class_number=1)

            predictor.main(all_files, output_path, solver, overlap_rate, target_size)
            
            self.status_label.config(text="状态：预测完成", fg="green")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.status_label.config(text="状态：发生错误", fg="red")
        finally:
            self.start_button.config(state=tk.NORMAL)

# 启动 GUI
if __name__ == "__main__":
    app = SegmentationGUI()
    app.mainloop()