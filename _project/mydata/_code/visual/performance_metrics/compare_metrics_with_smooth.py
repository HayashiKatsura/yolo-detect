import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional

def smooth_data(data: np.ndarray, window_size: int = 15) -> np.ndarray:
    """简单移动平均平滑"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return np.array(smoothed)


def plot_csv_comparison(csv_files: List[str], target_column, legend_labels: List[str], 
                       output_path: str, smooth: bool = False, window_size: int = 15):
    """绘制多个CSV文件对比曲线图，支持多列子图"""
    
    # 处理单列或多列
    if isinstance(target_column, str):
        target_columns = [target_column]
    else:
        target_columns = target_column
    
    if len(csv_files) != len(legend_labels):
        raise ValueError("CSV文件数量与图例标签数量不匹配")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 单列：一个图，多列：子图
    if len(target_columns) == 1:
        plt.figure(figsize=(12, 8))
        axes = [plt.gca()]
    else:
        # 多列：创建子图
        n_cols = min(2, len(target_columns))
        n_rows = (len(target_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        if len(target_columns) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
    
    colors = plt.cm.tab10(range(len(csv_files)))
    
    # 为每个指标创建图
    for col_idx, col in enumerate(target_columns):
        ax = axes[col_idx] if len(target_columns) > 1 else axes[0]
        
        for i, (csv_file, label) in enumerate(zip(csv_files, legend_labels)):
            try:
                if not os.path.exists(csv_file):
                    continue
                
                df = pd.read_csv(csv_file)
                if col not in df.columns:
                    continue
                
                y_data = df[col].values
                x_data = range(len(y_data))
                
                # 数据平滑
                if smooth:
                    y_data = smooth_data(y_data, window_size)
                
                ax.plot(x_data, y_data, 
                       color=colors[i], 
                       linewidth=2, 
                       label=label,
                       marker='o' if not smooth else None,
                       markersize=3 if not smooth else 0,
                       alpha=0.8)
                
                print(f"成功绘制: {csv_file} - {col} ({len(y_data)} 点)")
                
            except Exception as e:
                continue
        
        # 设置每个子图的属性
        ax.set_xlabel('Epoch/Step', fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(f'{col} Comparison', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 隐藏多余的子图
    if len(target_columns) > 1:
        for i in range(len(target_columns), len(axes)):
            axes[i].set_visible(False)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    import os
    
    # # 单列
    # target_column = "train/loss"
    # plot_csv_comparison(csv_files, target_column, legend_labels, "single_column.png")
    csv_files = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060012_yolo12_Train2419_Val479/results.csv", 
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507062048_yolo11_Train2419_Val479/results.csv",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507070208_yolov8_Train2419_Val479/results.csv",
        '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060825_yolo12-convnextv2-MambaOutx4_Train2419_Val479/results.csv'
    ]
    
    target_columns = ["metrics/mAP50(B)","metrics/mAP50-95(B)"]
    
    legend_labels = [
        "Yolov8",
        "Yolov11",
        "Yolov12",
        "ChipsYolo"
    ]
    save_path = '/home/panxiang/coding/kweilx/ultralytics/_visual/metrics'
    
    # 多列 - 你需要的功能
    plot_csv_comparison(csv_files, target_columns, legend_labels, os.path.join(save_path, "multi_columns_ap50.png"))
    
    # 多列 + 平滑
    plot_csv_comparison(csv_files, target_columns, legend_labels, os.path.join(save_path, "multi_columns_smooth_ap50.png"), 
                       smooth=True, window_size=20)