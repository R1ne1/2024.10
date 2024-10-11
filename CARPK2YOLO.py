import os
import numpy as np
from tqdm import tqdm

# 输入标签文件夹
label_folder = r"F:\目标样本\object_detection\CARPK\datasets\PUCPR+_devkit\data\Annotations"
label_list = os.listdir(label_folder)

# 输出标签文件夹
label_output = r"F:\目标样本\object_detection\CARPK\datasets\PUCPR+_devkit\data\labels"

# 图像宽高
img_width = 1280
img_height = 720

if __name__ == '__main__':
    for label_file in tqdm([f for f in os.listdir(label_folder) if f.endswith('.txt')]):
        # 输出文件路径
        output_file_path = os.path.join(label_output, label_file)

        # 读取标签
        with open(os.path.join(label_folder, label_file), 'r') as f:
            label_content = f.read().strip()

        # 若输入文件为空，则直接创建对应的空输出文件
        if not label_content:
            with open(output_file_path, 'w') as out_file:
                pass  # 创建一个空文件
            continue

        # 如果文件不为空，则进行标签转换
        lb = np.array([x.split() for x in label_content.splitlines()], dtype=int)
        for obj in lb:
            class_index = obj[4]
            xmin, ymin, xmax, ymax = obj[0], obj[1], obj[2], obj[3]

            # 将box信息转换到 YOLO 格式
            xcenter = xmin + (xmax - xmin) / 2
            ycenter = ymin + (ymax - ymin) / 2
            w = xmax - xmin
            h = ymax - ymin

            # 绝对坐标转相对坐标，保留6位小数
            xcenter = round(xcenter / img_width, 6)
            ycenter = round(ycenter / img_height, 6)
            w = round(w / img_width, 6)
            h = round(h / img_height, 6)

            info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

            # 写入标签（追加写入）
            with open(output_file_path, 'a') as out_file:
                out_file.write(" ".join(info) + "\n")
