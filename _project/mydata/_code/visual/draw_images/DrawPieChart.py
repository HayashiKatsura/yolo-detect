import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_pie_chart(values, labels, title="饼图", figsize=(10, 8), 
                   colors=None, autopct='%1.1f%%', startangle=90, 
                   save_path=None, dpi=300, bbox_inches='tight'):
    """
    绘制饼图函数
    
    参数:
    values: 数值列表
    labels: 标签名称列表
    title: 图表标题
    figsize: 图形大小 (宽, 高)
    colors: 颜色列表（可选，如果不提供会自动生成）
    autopct: 百分比显示格式
    startangle: 起始角度
    save_path: 保存路径（可选，如果提供则保存图片到指定路径）
    dpi: 保存图片的分辨率（默认300）
    bbox_inches: 保存时的边界框设置（默认'tight'确保完整保存）
    
    返回: 无
    """
    
    # 检查输入数据
    if len(values) != len(labels):
        raise ValueError("数值列表和标签列表长度必须相同")
    
    if any(v < 0 for v in values):
        raise ValueError("数值不能为负数")
    
    # 转换为numpy数组便于计算
    values = np.array(values)
    
    # 如果没有提供颜色，自动生成不同的颜色
    if colors is None:
        # 使用matplotlib的颜色映射生成不同颜色
        cmap = plt.cm.Set3  # 可以改为其他颜色映射如：Set1, Set2, Pastel1, Pastel2等
        colors = [cmap(i) for i in np.linspace(0, 1, len(values))]
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(values, 
                                    #   labels=labels,
                                    labels = None,
                                      autopct=autopct,
                                      startangle=startangle,
                                      colors=colors,
                                      explode=None,  # 可以设置为突出显示某些扇形
                                      shadow=True,   # 添加阴影效果
                                      textprops={'fontsize': 12})
    
    # 设置百分比文字的样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 确保饼图是圆形
    ax.axis('equal')
    
    # 添加图例 - 放在右下角并缩小字体，避免遮挡
    legend = ax.legend(wedges, labels, title="anomalies", 
                      loc="lower right", 
                      bbox_to_anchor=(1.0, 0.0),
                      fontsize=10,
                      title_fontsize=11,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
    
    # 调整布局以确保图例不被截断
    plt.tight_layout()
    
    # 保存图片（如果提供了保存路径）
    if save_path:
        try:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                       facecolor='white', edgecolor='none')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"保存图片时出错: {e}")
    
    # 显示图形
    plt.show()
    
    # 打印统计信息
    total = sum(values)
    print(f"\n=== 统计信息 ===")
    print(f"总计: {total}")
    print("各项占比:")
    for label, value in zip(labels, values):
        percentage = (value / total) * 100
        print(f"  {label}: {value} ({percentage:.1f}%)")

def simple_pie_chart(values, labels, title="饼图", save_path=None):
    """
    简化版饼图绘制函数
    
    参数:
    values: 数值列表
    labels: 标签名称列表
    title: 图表标题
    save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, shadow=True)
    
    # 设置百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    
    # 添加优化的图例
    ax.legend(wedges, labels, 
              loc="lower right", 
              bbox_to_anchor=(1.0, 0.0),
              fontsize=10,
              title_fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（如果提供了保存路径）
    if save_path:
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"图片已保存到: {save_path}")
        except Exception as e:
            print(f"保存图片时出错: {e}")
    
    plt.show()
    
    # 打印占比信息
    total = sum(values)
    print(f"\n=== 简单统计 ===")
    for label, value in zip(labels, values):
        print(f"{label}: {(value/total)*100:.1f}%")

def batch_save_charts(values, labels, title, base_filename):
    """
    批量保存不同格式的饼图
    
    参数:
    values: 数值列表
    labels: 标签列表
    title: 图表标题
    base_filename: 基础文件名（不含扩展名）
    """
    # 创建输出目录
    output_dir = "pie_charts_output"
    os.makedirs(output_dir, exist_ok=True)
    
    formats = ['png', 'jpg', 'pdf', 'svg']
    
    for fmt in formats:
        save_path = os.path.join(output_dir, f"{base_filename}.{fmt}")
        dpi_setting = 300 if fmt in ['png', 'jpg'] else None
        
        draw_pie_chart(values, labels, title=title, 
                      save_path=save_path, 
                      dpi=dpi_setting if dpi_setting else 300)

# 示例用法和测试代码
# if __name__ == "__main__":
    # # 示例数据1：职业分布
    # occupation_values = [38, 25, 18, 10, 9]
    # occupation_labels = ['<1% occupation', '1%-5% occupation', '5-10% occupation', 
    #                     '10%-20% occupation', '20%-45% occupation']
    
    # print("=== 示例1：职业分布饼图 ===")
    # draw_pie_chart(occupation_values, occupation_labels, 
    #                title="职业分布饼图",
    #                save_path="occupation_pie_chart.png")
    
    # # 示例数据2：销售数据
    # sales_values = [150, 200, 120, 80, 50]
    # sales_labels = ['产品A', '产品B', '产品C', '产品D', '产品E']
    
    # # 自定义颜色
    # custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    # print("\n=== 示例2：销售数据分布（自定义颜色）===")
    # draw_pie_chart(sales_values, sales_labels, 
    #                title="销售数据分布", 
    #                colors=custom_colors,
    #                autopct='%1.2f%%',
    #                save_path="sales_distribution.jpg",
    #                dpi=200)
    
    # # 示例数据3：简化版本
    # simple_values = [30, 25, 20, 15, 10]
    # simple_labels = ['类别A', '类别B', '类别C', '类别D', '类别E']
    
    # print("\n=== 示例3：简化版饼图 ===")
    # simple_pie_chart(simple_values, simple_labels, 
    #                  title="简化版饼图", 
    #                  save_path="simple_chart.png")
    
    # # 示例数据4：不同格式保存
    # print("\n=== 示例4：批量保存不同格式 ===")
    # batch_save_charts(occupation_values, occupation_labels, 
    #                  "职业分布图", "occupation_chart")

# # 使用说明和提示
# print("""
# === 使用说明 ===

# 1. 基本使用：
#    draw_pie_chart([10, 20, 30], ['A', 'B', 'C'], title="我的饼图")

# 2. 保存图片：
#    draw_pie_chart(values, labels, save_path="my_chart.png")

# 3. 自定义颜色：
#    colors = ['red', 'blue', 'green']
#    draw_pie_chart(values, labels, colors=colors)

# 4. 支持的保存格式：
#    - PNG: 网页显示推荐
#    - JPG: 文件较小
#    - PDF: 矢量格式，适合打印
#    - SVG: 矢量格式，适合网页

# 5. 高级设置：
#    - dpi: 图片分辨率（默认300）
#    - figsize: 图形大小，如 (12, 8)
#    - autopct: 百分比格式，如 '%1.2f%%'

# 6. 特点：
#    ✅ 自动计算比例并显示在扇形内
#    ✅ 自动生成不同颜色
#    ✅ 图例位于右下角，避免遮挡
#    ✅ 支持中文标签
#    ✅ 高质量图片输出
#    ✅ 自动创建保存目录
# """)
        
        
# 示例用法
if __name__ == "__main__":
    # 示例数据
    import os
    save_path = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images'
    save_path = os.path.join(save_path, 'Proportion of category quantity.png')
    # data_values = [568,373,459,361,381,364]
    data_values = [275,35,49,10,87,35]
    
    data_labels = ['black_burn', 
                   'curcuit_damage', 
                   'ink_pollution', 
                   'flue_resuidence',
                   'pi_over_exposure', 
                   'purple_circle']
    
    # custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    # 绘制饼图并保存
    # draw_pie_chart(data_values, data_labels, 
    #                title="train_data_pie_chart",
    #                save_path=save_path)
    draw_pie_chart(data_values,
                   data_labels, 
                   title="Proportion of category quantity",
                #    colors=custom_colors,
                   autopct='%1.2f%%',
                   save_path=save_path,
                   dpi=200)
    
#     # # 另一个示例
#     # sales_data = [150, 200, 120, 80, 50]
#     # sales_labels = ['产品A', '产品B', '产品C', '产品D', '产品E']
    
#     # # 自定义颜色和保存路径
#     # custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
#     # draw_pie_chart(sales_data, sales_labels, 
#     #                title="销售数据分布", 
#     #                colors=custom_colors,
#     #                autopct='%1.2f%%',
#     #                save_path="./charts/sales_distribution.jpg",  # 保存为JPG格式
#     #                dpi=150)  # 自定义分辨率
    
#     # # 使用简化版本并保存
#     # simple_pie_chart(data_values, data_labels, 
#     #                  title="简单饼图", 
#     #                  save_path="simple_chart.pdf")  # 保存为PDF格式