import matplotlib.pyplot as plt

def plot_histogram(labels, values, title="", save_path=None):
    """
    绘制柱状图，每个柱子顶部显示数值，底部显示名称。
    
    参数：
        labels (list): 名称列表
        values (list): 数值列表（百分比）
        title (str): 图表标题
        save_path (str): 保存路径，如 'chart.png'，默认为 None 表示只显示不保存
    """
    plt.figure(figsize=(9, 5))
    colors = plt.cm.tab20(range(len(labels)))  # 不同颜色

    # 绘制柱状图
    bars = plt.bar(labels, values, color=colors, edgecolor="white")

    # 在每个柱子顶部标注数值
    for bar, val in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y, f"{val:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.ylim(0, max(values) * 1.25)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
        
if __name__ == '__main__':
    labels = ["PI", "Residue", "Black", "Pollution", "Circuit_damage", "Circle"]
    values = [17.72, 2.04, 56.01, 9.98, 7.13, 7.13]

    # 显示图像
    # plot_histogram(labels, values)

    # 保存图像到本地
    plot_histogram(labels, values, save_path="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/histogram.png")

