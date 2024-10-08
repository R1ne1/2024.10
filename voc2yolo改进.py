import os
import xml.etree.ElementTree as ET
##处理Mar20数据集
##此代码中，直接将voc文件中的object name作为yolo标签的第一个字段未给予对应数字id

def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图像的宽度和高度
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_name

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算YOLO格式的中心点坐标和宽高，并归一化
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    return yolo_labels


def batch_convert_xml_to_yolo(xml_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            yolo_labels = convert_xml_to_yolo(xml_path)

            # 获取文件名，不带扩展名
            file_name = os.path.splitext(xml_file) [0]
            yolo_file = os.path.join(output_dir, file_name + '.txt')

            with open(yolo_file, 'w') as f:
                for label in yolo_labels:
                    f.write(label + '\n')


# 示例XML文件目录路径
xml_dir = 'E:/数据集/MAR20/Annotations/Horizontal Bounding Boxes'
# 保存YOLO格式标签文件的目录路径
output_dir = 'E:/数据集/MAR20/Annotations/yolo_Horizontal Bounding Boxes'

# 批量转换
batch_convert_xml_to_yolo(xml_dir, output_dir)
