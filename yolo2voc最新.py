"""
YOLO标注格式（txt文件）如下
 0 0.521000 0.235075 0.362000 0.450249
 0 0.213000 0.645522 0.418000 0.519900
 0 0.794000 0.665423 0.376000 0.470149

txt中信息说明：
每一行代表标注的一个目标
第一个数代表标注目标的标签
后面的四个数代表标注框的中心坐标和标注框的相对宽和高（进行归一化）
五个数据从左到右以此为：class_index,x_center,y_center,W,h.(后四个数据都是归一化的）。
"""

import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2

'''
import xml
xml.dom.minidom.Document().writexml()
def writexml(self,
             writer: Any,
             indent: str = "",
             addindent: str = "",
             newl: str = "",
             encoding: Any = None) -> None
'''


class YOLO2VOCConvert:
    def __init__(self, txts_path, xmls_path, imgs_path):
        self.txts_path = txts_path  # 标注的yolo格式标签文件路径
        self.xmls_path = xmls_path  # 转化为voc格式标签之后保存路径
        self.imgs_path = imgs_path  # 读取图片的路径和图片名字，存储到xml标签文件中
        self.classes = ["1","2","3","5","6"]  # 目标的类别

    # 从所有的txt文件中提取出所有的类别， yolo格式的标签格式类别为数字 0,1,...
    # writer为True时，把提取的类别保存到'./Annotations/classes.txt'文件中
    def search_all_classes(self, writer=False):
        # 读取每一个txt标签文件，取出每个目标的标注信息
        all_names = set()
        txts = os.listdir(self.txts_path)
        # 使用列表生成式过滤出只有后缀名为txt的标签文件
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        print(len(txts), txts)
        # 11 ['0002030.txt', '0002031.txt', ... '0002039.txt', '0002040.txt']
        for txt in txts:
            txt_file = os.path.join(self.txts_path, txt)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']
                    all_names.add(int(object[0]))
            # print(objects)  # ['2 0.506667 0.553333 0.490667 0.658667\n', '0 0.496000 0.285333 0.133333 0.096000\n', '8 0.501333 0.412000 0.074667 0.237333\n']

        print("所有的类别标签：", all_names, "共标注数据集：%d张" % len(txts))

        # 把从xmls标签文件中提取的类别写入到'./Annotations/classes.txt'文件中
        # if writer:
        #     with open('./Annotations/classes.txt', 'w') as f:
        #         for label in all_names:
        #             f.write(label + '\n')

        return list(all_names)

    def yolo2voc(self):
        # 创建一个保存xml标签文件的文件夹
        if not os.path.exists(self.xmls_path):
            os.mkdir(self.xmls_path)

        # 获取图片尺寸信息（shape）     读取每个txt标签文件并取出每个目标的标注信息
        imgs = os.listdir(self.imgs_path)
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]  # 过滤掉classes.txt文件
        print(txts)
        # 注意，这里保持图片的数量和标签txt文件数量相等，且要保证名字是一一对应的   (后面改进，通过判断txt文件名是否在imgs中即可)
        if len(imgs) == len(txts):  # 注意：./Annotation_txt 不要把classes.txt文件放进去
            map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
            txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
            print(len(txts), txts)
            for img_name, txt_name in map_imgs_txts:
                # 读取图片的尺度信息
                print("读取图片：", img_name)
                img_path = os.path.join(self.imgs_path, img_name)
                print("从以下路径读取图像:", img_path)
                img = cv2.imread(img_path)
                height_img, width_img, depth_img = img.shape
                print(height_img, width_img, depth_img)  # h 就是多少行（对应图片的高度）， w就是多少列（对应图片的宽度）

                # 获取标注文件txt中的标注信息
                all_objects = []
                txt_file = os.path.join(self.txts_path, txt_name)
                with open(txt_file, 'r') as f:
                    objects = f.readlines()
                    for object in objects:
                        object = object.strip().split(' ')
                        all_objects.append(object)
                        print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']

                # 创建xml标签文件中的标签
                xmlBuilder = Document()
                # 创建annotation标签，也是根标签
                annotation = xmlBuilder.createElement("annotation")

                # 给标签annotation添加一个子标签
                xmlBuilder.appendChild(annotation)

                # 创建子标签folder
                folder = xmlBuilder.createElement("folder")
                # 给子标签folder中存入内容，folder标签中的内容是存放图片的文件夹，例如：JPEGImages
                folderContent = xmlBuilder.createTextNode(self.imgs_path.split('/')[-1])  # 标签内存
                folder.appendChild(folderContent)  # 把内容存入标签
                annotation.appendChild(folder)  # 把存好内容的folder标签放到 annotation根标签下

                # 创建子标签filename
                filename = xmlBuilder.createElement("filename")
                # 给子标签filename中存入内容，filename标签中的内容是图片的名字，例如：000250.jpg
                filenameContent = xmlBuilder.createTextNode(txt_name.split('.')[0] + '.jpg')  # 标签内容
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                # 把图片的shape存入xml标签中
                size = xmlBuilder.createElement("size")
                # 给size标签创建子标签width
                width = xmlBuilder.createElement("width")  # size子标签width
                widthContent = xmlBuilder.createTextNode(str(width_img))
                width.appendChild(widthContent)
                size.appendChild(width)  # 把width添加为size的子标签
                # 给size标签创建子标签height
                height = xmlBuilder.createElement("height")  # size子标签height
                heightContent = xmlBuilder.createTextNode(str(height_img))  # xml标签中存入的内容都是字符串
                height.appendChild(heightContent)
                size.appendChild(height)  # 把width添加为size的子标签
                # 给size标签创建子标签depth
                depth = xmlBuilder.createElement("depth")  # size子标签width
                depthContent = xmlBuilder.createTextNode(str(depth_img))
                depth.appendChild(depthContent)
                size.appendChild(depth)  # 把width添加为size的子标签
                annotation.appendChild(size)  # 把size添加为annotation的子标签

                # 每一个object中存储的都是['2', '0.506667', '0.553333', '0.490667', '0.658667']一个标注目标
                for object_info in all_objects:
                    # 开始创建标注目标的label信息的标签
                    object = xmlBuilder.createElement("object")  # 创建object标签
                    # 创建label类别标签
                    # 创建name标签
                    imgName = xmlBuilder.createElement("name")  # 创建name标签
                    imgNameContent = xmlBuilder.createTextNode(self.classes[int(object_info[0])])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)  # 把name添加为object的子标签

                    # 创建pose标签
                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose)  # 把pose添加为object的标签

                    # 创建truncated标签
                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    # 创建difficult标签
                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    # 先转换坐标：(objx_center, objy_center, obj_width, obj_height)->(xmin，ymin, xmax,ymax)
                    x_center = float(object_info[1]) * width_img + 1
                    y_center = float(object_info[2]) * height_img + 1
                    xminVal = int(x_center - 0.5 * float(object_info[3]) * width_img)  # object_info列表中的元素都是字符串类型
                    yminVal = int(y_center - 0.5 * float(object_info[4]) * height_img)
                    xmaxVal = int(x_center + 0.5 * float(object_info[3]) * width_img)
                    ymaxVal = int(y_center + 0.5 * float(object_info[4]) * height_img)

                    # 创建bndbox标签(三级标签)
                    bndbox = xmlBuilder.createElement("bndbox")
                    # 在bndbox标签下再创建四个子标签(xmin，ymin, xmax,ymax) 即标注物体的坐标和宽高信息
                    # 在voc格式中，标注信息：左上角坐标（xmin, ymin） （xmax, ymax）右下角坐标
                    # 1、创建xmin标签
                    xmin = xmlBuilder.createElement("xmin")  # 创建xmin标签（四级标签）
                    xminContent = xmlBuilder.createTextNode(str(xminVal))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    # 2、创建ymin标签
                    ymin = xmlBuilder.createElement("ymin")  # 创建ymin标签（四级标签）
                    yminContent = xmlBuilder.createTextNode(str(yminVal))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)
                    # 3、创建xmax标签
                    xmax = xmlBuilder.createElement("xmax")  # 创建xmax标签（四级标签）
                    xmaxContent = xmlBuilder.createTextNode(str(xmaxVal))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)
                    # 4、创建ymax标签
                    ymax = xmlBuilder.createElement("ymax")  # 创建ymax标签（四级标签）
                    ymaxContent = xmlBuilder.createTextNode(str(ymaxVal))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)  # 把object添加为annotation的子标签
                f = open(os.path.join(self.xmls_path, os.path.splitext(txt_name)[0] + '.xml'), 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()


if __name__ == '__main__':
    txts_path1 = 'E:/datasets/COWC/train_data/Potsdam_ISPRS/label/'  # 标注的yolo格式标签文件路径
    xmls_path1 = 'E:/datasets/COWC/train_data/Potsdam_ISPRS/Annotations/'  # 转化为voc格式标签之后保存路径
    imgs_path1 = 'E:/datasets/COWC/train_data/Potsdam_ISPRS/image/'  # 图片路径

    yolo2voc_obj1 = YOLO2VOCConvert(txts_path1, xmls_path1, imgs_path1)
    labels = yolo2voc_obj1.search_all_classes()
    print('labels: ', labels)
    yolo2voc_obj1.yolo2voc()