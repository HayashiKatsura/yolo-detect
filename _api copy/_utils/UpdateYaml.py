import yaml

def update_yaml_config(yaml_file, new_values):
    """
    读取并更新YAML配置文件中的字段
    
    Args:
        yaml_file (str): YAML文件的路径
        new_values (dict): 要更新的字段和对应的新值
                          例如: {"path": "/new/path", "train": "/new/train", "val": "/new/val"}
    """
    # 读取现有YAML文件
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        # 如果文件不存在，创建一个空配置
        config = {}
    
    # 确保config是一个字典
    if config is None:
        config = {}
    
    # 更新或添加字段
    for field, value in new_values.items():
        config[field] = value
    
    # 将更新后的配置写回文件
    with open(yaml_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print(f"YAML配置文件 '{yaml_file}' 已更新")

# 使用示例
if __name__ == "__main__":
    yaml_file = "mydata.yaml"  # 替换为实际的文件路径
    new_values = {
        "path": "/new/path/to/dataset",
        "train": "/new/path/to/train/data",
        "val": "/new/path/to/validation/data"
    }
    
    update_yaml_config(yaml_file, new_values)