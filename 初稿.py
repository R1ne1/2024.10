import os
import pandas as pd
from PIL import Image

# 假设每个数据集存放在不同的文件夹中
datasets = {
    # "dataset1": "E:/数据集/COWC/train_data/Selwyn_LINZ",
    "COWC": "E:/数据集/MAR20/",
    # 你可以继续添加更多的数据集路径
}

# 初始化一个空列表，用于存储结果
data = []

# 遍历每个数据集
for dataset_name, dataset_path in datasets.items():
    image_folder = os.path.join(dataset_path, "image")
    label_folder = os.path.join(dataset_path, "label")  # 假设标注文件在 labels 文件夹中，YOLO 格式

    # 遍历每个图片文件
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # 检查图片格式
            image_path = os.path.join(image_folder, image_file)

            # 获取图片的尺寸
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # 获取对应的标注文件
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_folder, label_file)

            # 检查标注文件是否存在
            if not os.path.exists(label_path):
                print(f"标注文件 {label_path} 不存在，跳过该图片。")
                continue

            # 读取标注文件 (YOLO格式: 类别 x_center y_center width height)
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    category = parts[0]
                    bbox_width = float(parts[3]) * img_width
                    bbox_height = float(parts[4]) * img_height

                    # 将信息添加到列表中
                    data.append({
                        "Dataset": dataset_name,
                        "Category": category,
                        "Image_Name": image_file,
                        "Object": "Object",  # 可以自定义具体对象信息
                        "BBox_Width": bbox_width,
                        "BBox_Height": bbox_height,
                        "Image_Width": img_width,
                        "Image_Height": img_height
                    })

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 写入 Excel 文件
output_file = "dataset_statistics.xlsx"
df.to_excel(output_file, index=False)

print(f"数据已成功写入 {output_file}")
