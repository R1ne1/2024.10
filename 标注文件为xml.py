import os
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET

# 假设每个数据集存放在不同的文件夹中
datasets = {
    "MAR20": "E:/数据集/MAR20/",
}

# 初始化一个空列表，用于存储结果
data = []

# 遍历每个数据集
for dataset_name, dataset_path in datasets.items():
    image_folder = os.path.join(dataset_path, "image")
    label_folder = os.path.join(dataset_path, "Annotations/Horizontal Bounding Boxes")  # 假设标注文件在 labels 文件夹中，XML格式

    # 遍历每个图片文件
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # 检查图片格式
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

            # 遍历每个对象
            for idx, member in enumerate(root.findall('object')):
                category = member.find('name').text
                bbox = member.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                # 创建对象命名格式
                object_name = f"{os.path.splitext(image_file) [0]}_{category}_{idx + 1}"

                # 将信息添加到列表中
                data.append({
                    "Dataset": dataset_name,
                    "Category": category,
                    "Image_Name": image_file,
                    "Object": object_name,  # 使用自定义对象命名格式
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
