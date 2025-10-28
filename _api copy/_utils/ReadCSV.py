import pandas as pd

def csv_key_to_value(source,target):
    """
    根据source key-value获取target value
    Args:
        csv (_type_): _description_
        source (_type_): _description_
        target (_type_): _description_
    Returns:
        _type_: _description_
    """
    try:
        df = pd.read_csv(source['csv'])
        result = df[df[source['key']] == source['value']][target]
        if not result.empty:
            return result.iloc[0]
        return None
    except:
        return None
    
def get_line_data(csv_file, line_num):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 获取指定行的数据（可以修改为需要的行数）
    row_data = df.iloc[line_num]
    # 将行数据转化为键值对形式（字典形式）
    row_dict = row_data.to_dict()
    
    
    # 将所有行转化为字典列表
    data_dict = df.to_dict(orient='records')
    
    return row_dict, data_dict

    
    

if __name__ == '__main__':
    # # 使用示例
    # source = {
    #     'key':'file_id', 
    #     'value': '2223f7d1-ec42-45db-b123-db1e2d1308ea',
    #     'csv': '/Users/katsura/Documents/code/ultralytics/_api/data/data_record.csv'
    #     }
    # target = 'file_path'
    # name = csv_key_to_value(source,target)
    # print(f"ID {target} 对应的名称是: {name}")
    
    get_line_data('/Users/katsura/Documents/code/ultralytics/_api/data/results/Train-202504141209-c18-d2bb6362f654/train_202504141209/results.csv',-1)