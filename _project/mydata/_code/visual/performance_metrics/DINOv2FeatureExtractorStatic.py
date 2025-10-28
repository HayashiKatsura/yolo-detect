import os
import csv

def export_file_info_to_csv(input_dir, output_csv):
    """
    从指定文件夹中读取所有文件名（不含扩展名），并记录所属文件夹名，输出到CSV。

    参数:
        input_dir (str): 要读取的文件夹路径。
        output_csv (str): 输出的CSV文件路径。
    """
    # 提取当前文件夹名（最后一级）
    folder_name = os.path.basename(os.path.normpath(input_dir))
    rows = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path):
            file_name_no_ext = os.path.splitext(filename)[0]
            rows.append([file_name_no_ext, folder_name])

    # 写入CSV文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'folder_name'])
        writer.writerows(rows)

    print(f"已成功将数据写入CSV：{output_csv}")


if __name__ == '__main__':
    input_dir_list = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_0",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_1",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_2",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_3",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_4",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_5"
    ]
    output_csv_list = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_0.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_1.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_2.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_3.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_4.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/聚类分析/cluster_5.csv"
    ]
    for input_dir, output_csv in zip(input_dir_list, output_csv_list):
        export_file_info_to_csv(input_dir, output_csv)
    
