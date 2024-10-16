import pandas as pd

# 读取标签文件和映射表文件
data = pd.read_csv('dataset_labels.csv')  # YOLO格式标签数据
mapping = pd.read_csv('class_mapping.csv')  # 数据集映射表


# 定义一个函数，按数据集名称和类别ID进行映射
def map_class_name(row):
    dataset_name = row ['Dataset']
    class_id = row ['Category']

    # 根据当前行的数据集和ID查找映射名称
    name = mapping.loc [
        (mapping ['dataset'] == dataset_name) & (mapping ['id'] == class_id), 'name'
    ]

    # 返回映射的类别名称，如果没找到则返回原ID
    return name.values [0] if not name.empty else class_id


# 替换class列的数字为对应的类别名称
data ['Category'] = data.apply(map_class_name, axis=1)

# 保存替换后的结果
data.to_csv('updated_dataset_labels.csv', index=False)

print("替换完成，文件已保存为 'updated_dataset_labels.csv'")
