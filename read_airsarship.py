import os
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
import re

# 假设每个数据集存放在不同的文件夹中
datasets = {
    "AIR-SARShip-2.0": r"F:\目标样本\object_detection\AIR-SARship\AIR-SARShip-2.0",
    # "FAIR1M": r"F:\目标样本\object_detection\FAIR1M\train\part1",
}

# 初始化一个空列表，用于存储结果
data = []

# 遍历每个数据集
for dataset_name, dataset_path in datasets.items():
    image_folder = os.path.join(dataset_path, "images")
    label_folder = os.path.join(dataset_path, "Annotations")  # 假设标注文件在 labels 文件夹中，XML格式

    # 遍历每个图片文件
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".tif") or image_file.endswith(".tiff"):
            image_path = os.path.join(image_folder, image_file)

            # 获取图片的尺寸
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # 获取对应的标注文件
            label_file = os.path.splitext(image_file)[0] + ".xml"
            label_path = os.path.join(label_folder, label_file)

            # 检查标注文件是否存在
            if not os.path.exists(label_path):
                print(f"标注文件 {label_path} 不存在，跳过该图片。")
                continue

            # 解析 XML 文件
            tree = ET.parse(label_path)
            root = tree.getroot()

            # 创建一个字典来跟踪每个类别的计数
            category_count = {}

            # 遍历每个 objects 节点
            for objects in root.findall('objects'):
                for member in objects.findall('object'):
                    # 解析每个 object
                    category_node = member.find('possibleresult/name')
                    if category_node is None:
                        print("类别信息缺失，跳过此对象。")
                        continue
                    category = category_node.text  # 类别名称

                    # 提取所有点
                    points = member.find('points')
                    coordinates = []
                    for point in points.findall('point'):
                        xy = re.findall(r'\d+', point.text)  # 提取数字
                        x, y = int(xy[0]), int(xy[1])
                        coordinates.append((x, y))

                    # 获取 xmin, ymin, xmax, ymax
                    x_values = [coord[0] for coord in coordinates]
                    y_values = [coord[1] for coord in coordinates]
                    xmin, xmax = min(x_values), max(x_values)
                    ymin, ymax = min(y_values), max(y_values)

                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    print(f'Category: {category}, BBox: ({xmin}, {ymin}, {xmax}, {ymax}), Width: {bbox_width}, Height: {bbox_height}')

                    # 更新类别计数
                    if category not in category_count:
                        category_count[category] = 0
                    category_count[category] += 1

                    # 创建对象命名格式
                    object_name = f"{os.path.splitext(image_file)[0]}_{category}_{category_count[category]}"

                    # 将信息添加到列表中
                    data.append({
                        "Dataset": dataset_name,
                        "Category": category,
                        "Image_Name": image_file,
                        "Object": object_name,
                        "BBox_Width": bbox_width,
                        "BBox_Height": bbox_height,
                        "Image_Width": img_width,
                        "Image_Height": img_height
                    })

# 打印解析到的对象数量
print(f"总共解析到 {len(data)} 个对象。")

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 写入 CSV 文件
output_file = "AIR-SARShip-2.0.csv"
df.to_csv(output_file, index=False)

print(f"数据已成功写入 {output_file}")
