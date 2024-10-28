import re
import xml.etree.ElementTree as ET
import os
import json

voc_clses = ['ship']

categories = []
for iind, cat in enumerate(voc_clses):
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind
    categories.append(cate)


def getimages(xmlname, id):
    sig_xml_box = []
    tree = ET.parse(xmlname)
    root = tree.getroot()
    images = {}

    for i in root:  # 遍历一级节点
        chd = list(root)  # 用 list(root) 来替代 getchildren()
        for z in chd:  # 遍历二级节点
            chdd = list(z)  # 用 list() 来替代 getchildren()
            for y in chdd:  # 遍历三级节点
                if y.tag == 'filename':
                    file_name = y.text[:-5] + '.tiff'  # 0001.jpg
                    images['file_name'] = file_name

                if y.tag == 'object':
                    chddd = list(y)  # 用 list() 来替代 getchildren()
                    for j in chddd:  # 遍历四级节点
                        if j.tag == 'possibleresult':
                            chdddd = list(j)  # 用 list() 来替代 getchildren()
                            for x in chdddd:  # 遍历五级节点
                                if x.tag == 'name':
                                    cls_name = x.text
                            cat_id = voc_clses.index(cls_name) + 1
                        if j.tag == 'points':
                            bbox = []
                            xmin = 0
                            ymin = 0
                            xmax = 0
                            ymax = 0
                            for ind, r in enumerate(j):
                                if r.tag == 'point' and ind == 0:
                                    tmp = eval(r.text)
                                    tmp = ",".join("%s" % i for i in tmp)
                                    tmp = re.findall(r'\d+', tmp)
                                    xmin = int(tmp[0])
                                    ymin = int(tmp[1])

                                if r.tag == 'point' and ind == 2:
                                    tmp = eval(r.text)
                                    tmp = ",".join("%s" % i for i in tmp)
                                    tmp = re.findall(r'\d+', tmp)
                                    xmax = int(tmp[0])
                                    ymax = int(tmp[1])

                            bbox.append(xmin)
                            bbox.append(ymin)
                            bbox.append(xmax - xmin)
                            bbox.append(ymax - ymin)
                            bbox.append(id)  # 保存当前box对应的image_id
                            bbox.append(cat_id)
                            bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)  # bbox的面积
                            sig_xml_box.append(bbox)
    images['width'] = "1000"
    images['height'] = "1000"
    images['id'] = id
    return images, sig_xml_box


def get_xml_files(directory):
    # 获取指定目录下的所有 .xml 文件
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xml')]


# 使用原始字符串（r'...'）避免路径中的转义字符问题
voc2007xmls = r'F:\目标样本\object_detection\AIR-SARship\AIR-SARShip-2.0-xml'

# 获取所有 XML 文件
xmls = get_xml_files(voc2007xmls)

bboxes = []
ann_js = {}

# 保存路径也要使用原始字符串
json_name = r'F:\目标样本\object_detection\AIR-SARship\instances_train2017.json'
images = []

for i_index, xml_file in enumerate(xmls):
    image, sig_xml_bbox = getimages(xml_file, i_index)
    images.append(image)
    bboxes.extend(sig_xml_bbox)

ann_js['images'] = images
ann_js['categories'] = categories
annotations = []
for box_ind, box in enumerate(bboxes):
    anno = {}
    anno['image_id'] = box[-3]
    anno['category_id'] = box[-2]
    anno['bbox'] = box[:-3]
    anno['id'] = box_ind
    anno['area'] = box[-1]
    anno['iscrowd'] = 0
    annotations.append(anno)

ann_js['annotations'] = annotations
json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4 美化输出 JSON
