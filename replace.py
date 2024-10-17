import pandas as pd

# 读取标签文件和映射表文件，指定编码
data = pd.read_csv(r'C:\Users\Administrator\Desktop\2024.10-main\YOLO_HRSC2016.csv', encoding='utf-8')
mapping = pd.read_csv(r'C:\Users\Administrator\Desktop\2024.10-main\yolo映射表.csv', encoding='utf-8')

# 定义映射函数
def map_class_name(row):
    dataset_name = row['Dataset']
    category_id = row['Category']

    # 查找映射表中的类别名称
    name = mapping.loc[
        (mapping['Dataset'] == dataset_name) & (mapping['id'] == category_id), 'name'
    ]

    # 返回映射的类别名称，如果没找到则保留原ID
    return name.values[0] if not name.empty else category_id

# 替换类别ID为对应的类别名称
data['Category'] = data.apply(map_class_name, axis=1)

# 保存结果时指定编码，避免中文乱码
data.to_csv(r'C:\Users\Administrator\Desktop\2024.10-main\updated_dataset_labels.csv', encoding='utf-8-sig', index=False)

print("替换完成，文件已保存为 'updated_dataset_labels.csv'")
