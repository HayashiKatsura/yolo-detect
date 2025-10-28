import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List

def plot_csv_comparison(csv_files: List[str], target_column: str, legend_labels: List[str], output_path: str):
    """
    绘制多个CSV文件中指定列的对比曲线图
    
    参数:
    csv_files (List[str]): CSV文件路径列表
    target_column (str): 目标表头字段名
    legend_labels (List[str]): 图例标签列表
    output_path (str): 输出图像路径
    """
    
    # 验证参数
    if len(csv_files) != len(legend_labels):
        raise ValueError("CSV文件数量与图例标签数量不匹配")
    
    # 设置中文字体支持（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形和轴
    plt.figure(figsize=(12, 8))
    
    # 定义颜色列表，确保每条线颜色不同
    colors = plt.cm.tab10(range(len(csv_files)))  # 使用tab10颜色映射
    
    # 遍历每个CSV文件
    for i, (csv_file, label) in enumerate(zip(csv_files, legend_labels)):
        try:
            # 检查文件是否存在
            if not os.path.exists(csv_file):
                print(f"警告: 文件 {csv_file} 不存在，跳过...")
                continue
            
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查目标列是否存在
            if target_column not in df.columns:
                print(f"警告: 文件 {csv_file} 中不存在列 '{target_column}'，跳过...")
                continue
            
            # 提取目标列数据
            y_data = df[target_column].values
            x_data = range(len(y_data))  # 使用索引作为x轴
            
            # 绘制曲线
            plt.plot(x_data, y_data, 
                    color=colors[i], 
                    linewidth=2, 
                    label=label,
                    marker='o',
                    markersize=3,
                    alpha=0.8)
            
            print(f"成功绘制: {csv_file} ({len(y_data)} 个数据点)")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
            continue
    
    # 设置图形属性
    plt.xlabel('数据点索引 (Epoch/Step)', fontsize=12)
    plt.ylabel(f'{target_column}', fontsize=12)
    plt.title(f'{target_column} 对比曲线图', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图像已保存到: {output_path}")
    
    # 显示图像（可选）
    # plt.show()
    
    # 清理
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 示例用法
    csv_files = [
        "model1_results.csv",
        "model2_results.csv", 
        "model3_results.csv"
    ]
    
    target_column = "train/loss"  # 例如YOLO训练结果中的训练损失
    
    legend_labels = [
        "YOLOv8n",
        "YOLOv8s", 
        "YOLOv8m"
    ]
    
    output_path = "comparison_plot.png"
    
    # 调用函数
    plot_csv_comparison(csv_files, target_column, legend_labels, output_path)


# 扩展版本：支持多个指标对比
def plot_multiple_metrics_comparison(csv_files: List[str], target_columns: List[str], 
                                   legend_labels: List[str], output_path: str):
    """
    绘制多个CSV文件中多个指标的对比图（子图形式）
    
    参数:
    csv_files (List[str]): CSV文件路径列表
    target_columns (List[str]): 目标列名列表
    legend_labels (List[str]): 图例标签列表
    output_path (str): 输出图像路径
    """
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算子图布局
    n_metrics = len(target_columns)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(range(len(csv_files)))
    
    # 为每个指标创建子图
    for metric_idx, target_column in enumerate(target_columns):
        ax = axes[metric_idx]
        
        # 为每个CSV文件绘制当前指标
        for file_idx, (csv_file, label) in enumerate(zip(csv_files, legend_labels)):
            try:
                if not os.path.exists(csv_file):
                    continue
                
                df = pd.read_csv(csv_file)
                if target_column not in df.columns:
                    continue
                
                y_data = df[target_column].values
                x_data = range(len(y_data))
                
                ax.plot(x_data, y_data, 
                       color=colors[file_idx], 
                       linewidth=2, 
                       label=label,
                       marker='o',
                       markersize=2,
                       alpha=0.8)
                
            except Exception as e:
                print(f"处理文件 {csv_file} 的列 {target_column} 时出错: {str(e)}")
                continue
        
        # 设置子图属性
        ax.set_xlabel('Epoch/Step', fontsize=10)
        ax.set_ylabel(target_column, fontsize=10)
        ax.set_title(f'{target_column} 对比', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"多指标对比图已保存到: {output_path}")
    plt.close()


# 多指标使用示例
if __name__ == "__main__":
    # 多指标对比示例
    csv_files = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202506242003_yolo12_Train2419_Val444/results.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507011747_yolo12-convnextv2-dyt_Train2419_Val444/results.csv", 
    ]
    
    target_columns = ["metrics/mAP50(B)","metrics/recall(B)"]
    
    legend_labels = [
        "v12",
        "convnextv2-dyt", 
    ]
    
    output_path = "/home/panxiang/coding/kweilx/ultralytics/_visual/multi_metrics_comparison.png"
    
    # 调用多指标对比函数
    plot_multiple_metrics_comparison(csv_files, target_columns, legend_labels, output_path)