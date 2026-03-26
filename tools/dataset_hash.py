# 保证数据集一致性的代码
# 文件作用：对指定文件夹内所有文件计算一致性的MD5散列值，
#           先分别计算每个文件的MD5值，然后拼接所有文件的散列值再计算整体MD5值

import os
import hashlib
from tqdm import tqdm

def md5_hash_file(file_path, block_size=65536):
    """
    计算单个文件的MD5散列值
    参数：
      file_path：文件的路径
      block_size：读取文件时的块大小（默认为65536字节）
    返回：
      文件的MD5散列值（字符串）
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def hash_folder(folder_path):
    """
    遍历目标文件夹及子文件夹，收集所有文件路径并计算各文件MD5散列，
    结合文件的相对路径参与计算，确保即使文件名发生变化也能检测到，
    最后将各文件散列值拼接后计算出文件夹的整体MD5散列值
    参数：
      folder_path：目标文件夹的路径
    返回：
      整个文件夹的MD5散列值（字符串）
    """
    file_paths = []
    # 收集所有文件的路径，保证排序以确保结果一致
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()   # 对子目录进行排序
        files.sort()  # 对文件名进行排序
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_hashes = []
    # 逐个文件计算散列，并显示进度条
    for file_path in tqdm(file_paths, desc="处理文件", unit="个文件"):
        relative_path = os.path.relpath(file_path, folder_path)
        # 统一路径分隔符，避免不同操作系统差异
        relative_path = relative_path.replace(os.sep, "/")
        content_hash = md5_hash_file(file_path)
        # 将相对路径和内容哈希合并
        combined = relative_path + content_hash
        file_specific_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()
        file_hashes.append(file_specific_hash)

    # 拼接所有文件的MD5值，并计算整体的MD5散列值
    combined_hash = hashlib.md5("".join(file_hashes).encode('utf-8')).hexdigest()
    return combined_hash

if __name__ == "__main__":
    folder = r"C:\Users\17905\Desktop\mydata"  # 目标文件夹路径
    print("文件夹 {} 的哈希值: {}".format(folder, hash_folder(folder)))