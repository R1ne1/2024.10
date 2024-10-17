import xml.etree.ElementTree as ET
import os
import math

#旋转框
# 定义类别映射字典 (XML 中 Class_ID -> YOLO 类别 ID)
class_mapping = {
    100000001: 1, 100000002: 2, 100000003: 3, 100000004: 4, 100000005: 5,
    100000006: 6, 100000007: 7, 100000008: 8, 100000009: 9, 100000010: 10,
    100000011: 11, 100000012: 12, 100000013: 13, 100000014: 14, 100000015: 15,
    100000016: 16, 100000017: 17, 100000018: 18, 100000019: 19, 100000020: 20,
    100000021: 21, 100000022: 22, 100000023: 23, 100000024: 24, 100000025: 25,
    100000026: 26, 100000027: 27, 100000028: 28, 100000029: 29, 100000030: 30,
    100000031: 31, 100000032: 32, 100000033: 33
}

def hrsc_to_yolo_rotated(xml_path, label_path, class_mapping):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"解析 {xml_path} 时出错：{e}")
        return
    except FileNotFoundError:
        print(f"未找到文件：{xml_path}")
        return

    with open(label_path, 'w') as f:
        for obj in root.iter('HRSC_Object'):
            difficult = obj.find('difficult').text
            original_class_id = int(obj.find('Class_ID').text)

            # 跳过无法映射的类别并记录文件路径
            if original_class_id not in class_mapping:
                print(f"跳过未映射的类别 ID：{original_class_id}（文件：{xml_path}）")
                continue

            # 根据映射表获取 YOLO 类别 ID
            class_id = class_mapping[original_class_id]

            if int(difficult) == 1:
                continue

            # 提取边界框参数
            mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang = (
                float(obj.find('mbox_cx').text),
                float(obj.find('mbox_cy').text),
                float(obj.find('mbox_w').text),
                float(obj.find('mbox_h').text),
                float(obj.find('mbox_ang').text)
            )
            image_width = int(root.find('Img_SizeWidth').text)
            image_height = int(root.find('Img_SizeHeight').text)

            # 归一化坐标
            norm_cx = mbox_cx / image_width
            norm_cy = mbox_cy / image_height
            norm_w = mbox_w / image_width
            norm_h = mbox_h / image_height

            # 将角度转换为度数
            rotation_angle_degrees = mbox_ang * (180.0 / math.pi)

            # 以 YOLO 格式写入文件
            yolo_rotated_bbox = [class_id, norm_cx, norm_cy, norm_w, norm_h, rotation_angle_degrees]
            f.write(" ".join([str(a) for a in yolo_rotated_bbox]) + '\n')

if __name__ == "__main__":
    xml_root = r"F:\目标样本\object_detection\HRSC2016_dataset\HRSC2016\FullDataSet\Annotations"
    txt_root = r"F:\目标样本\object_detection\HRSC2016_dataset\HRSC2016\FullDataSet\YOLO_labels"

    # 创建 YOLO_labels 目录（如果不存在）
    os.makedirs(txt_root, exist_ok=True)

    xml_names = os.listdir(xml_root)
    total_files = len(xml_names)

    for index, xml_name in enumerate(xml_names, start=1):
        xml_path = os.path.join(xml_root, xml_name)
        txt_path = os.path.join(txt_root, xml_name.split('.')[0] + '.txt')
        hrsc_to_yolo_rotated(xml_path, txt_path, class_mapping)
        print(f"[{index}/{total_files}] 已处理文件：{xml_name}")

    print("处理完成！")
