import os
from PIL import Image

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    center_x = (x1 + x2) / 2.0 * dw
    center_y = (y1 + y2) / 2.0 * dh
    width = (x2 - x1) * dw
    height = (y2 - y1) * dh
    return center_x, center_y, width, height

# 目录路径
annotations_dir = r'F:\目标样本\object_detection\NWPU_VHR-10\汇总\annotations'
images_dir = r'F:\目标样本\object_detection\NWPU_VHR-10\汇总\images'
output_dir = r'F:\目标样本\object_detection\NWPU_VHR-10\汇总\labels'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 类别映射
class_map = {
    '1': '0', '2': '1', '3': '2', '4': '3', '5': '4',
    '6': '5', '7': '6', '8': '7', '9': '8', '10': '9'
}

def get_image_size(image_path):
    """获取图片尺寸"""
    with Image.open(image_path) as img:
        return img.size

def parse_bbox(line):
    """解析标注行中的边界框和类别信息"""
    try:
        line = line.strip().replace('(', '').replace(')', '')
        parts = line.split(',')
        if len(parts) != 5:
            raise ValueError(f"格式无效: {line}")

        x1, y1, x2, y2 = map(int, map(float, parts[:4]))  # 支持浮点数转换
        class_id = parts[4].strip()
        return x1, y1, x2, y2, class_id
    except ValueError as e:
        print(f"解析行出错: {line}。错误: {e}")
        return None, None, None, None, None

# 遍历标注文件
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith('.txt'):
        image_file = os.path.splitext(annotation_file)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_file)

        print(f"处理图片: {image_path}")

        if not os.path.exists(image_path):
            print(f"图片文件 {image_path} 不存在，跳过。")
            continue

        img_width, img_height = get_image_size(image_path)

        annotation_path = os.path.join(annotations_dir, annotation_file)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        output_path = os.path.join(output_dir, annotation_file)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for line in lines:
                x1, y1, x2, y2, class_id = parse_bbox(line)
                if x1 is None:
                    continue

                mapped_class_id = class_map.get(class_id)
                if mapped_class_id is None:
                    print(f"未知类别ID: {class_id}，跳过。")
                    continue

                center_x, center_y, width, height = convert_to_yolo_format(
                    x1, y1, x2, y2, img_width, img_height
                )

                out_f.write(f"{mapped_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

print("转换完成。")
