import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir):
    def convert_box(size, box):
        # Convert VisDrone box to YOLO CxCywh box, 坐标进行了归一化
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    # 创建 YOLO 标签文件夹
    (dir / 'Annotations_YOLO').mkdir(parents=True, exist_ok=True)
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')

    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []

        with open(f, 'r') as file:  # 读取 annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        # 仅在处理完所有行后写入文件
        output_file = dir / 'Annotations_YOLO' / f.name
        with open(output_file, 'w') as fl:
            fl.writelines(lines)  # 写入 label.txt


# 设置汇总后的文件夹路径
dir = Path(r'F:/目标样本/object_detection/VisDrone/VisDrone2019_all')  # 汇总后文件夹路径

# 进行转换
visdrone2yolo(dir)  # 转换 VisDrone 标注为 YOLO 标签
