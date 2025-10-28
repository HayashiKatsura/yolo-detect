import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

class ImageReorganizer:
    def __init__(self, base_output_dir="reorganized_images"):
        """
        图像重组工具
        
        Args:
            base_output_dir: 输出目录的基础路径
        """
        self.base_output_dir = base_output_dir
        self.reorganization_log = []
        
    def get_image_paths(self, image_dir):
        """
        获取目录下所有图像文件的路径
        
        Args:
            image_dir: 图像目录路径
            
        Returns:
            list: 图像文件路径列表
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_paths = []
        
        print(f"扫描图像目录: {image_dir}")
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        print(f"找到 {len(image_paths)} 张图像")
        return sorted(image_paths)  # 排序确保顺序一致
    
    def load_clustering_results(self, labels_path, features_2d_path=None):
        """
        加载聚类结果
        
        Args:
            labels_path: 聚类标签文件路径
            features_2d_path: 2D特征文件路径（可选，用于生成可视化）
            
        Returns:
            tuple: (labels, features_2d)
        """
        print(f"加载聚类结果: {labels_path}")
        
        try:
            labels = np.load(labels_path)
            print(f"✅ 成功加载 {len(labels)} 个聚类标签")
            
            # 聚类统计
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"聚类分布:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(labels) * 100
                print(f"  Cluster {label}: {count} 张图像 ({percentage:.1f}%)")
            
            features_2d = None
            if features_2d_path and os.path.exists(features_2d_path):
                features_2d = np.load(features_2d_path)
                print(f"✅ 同时加载了2D特征数据")
            
            return labels, features_2d
            
        except Exception as e:
            print(f"❌ 加载聚类结果失败: {e}")
            return None, None
    
    def create_output_structure(self, labels, output_dir, copy_mode=True):
        """
        创建输出目录结构
        
        Args:
            labels: 聚类标签数组
            output_dir: 输出目录
            copy_mode: True=复制文件，False=移动文件
        """
        # 创建主输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个聚类创建子目录
        unique_labels = np.unique(labels)
        
        print(f"\n创建输出目录结构: {output_dir}")
        print(f"模式: {'复制' if copy_mode else '移动'} 文件")
        
        for label in unique_labels:
            cluster_dir = os.path.join(output_dir, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)
            print(f"  📁 {cluster_dir}")
        
        # 创建元数据目录
        metadata_dir = os.path.join(output_dir, "_metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        return unique_labels
    
    def reorganize_images(self, image_paths, labels, output_dir, copy_mode=True, 
                         confirm_before_start=True):
        """
        重新组织图像文件
        
        Args:
            image_paths: 图像文件路径列表
            labels: 聚类标签数组
            output_dir: 输出目录
            copy_mode: True=复制文件，False=移动文件
            confirm_before_start: 是否在开始前确认
        """
        if len(image_paths) != len(labels):
            print(f"❌ 错误：图像数量({len(image_paths)}) 与标签数量({len(labels)}) 不匹配")
            return False
        
        # 确认操作
        if confirm_before_start:
            print(f"\n⚠️  准备{'复制' if copy_mode else '移动'} {len(image_paths)} 张图像")
            print(f"源目录包含的图像将被重新组织到: {output_dir}")
            if not copy_mode:
                print("⚠️  移动模式会删除原始位置的文件！")
            
            # 在实际使用时取消注释下面这行
            # response = input("确认继续？(y/n): ")
            # if response.lower() != 'y':
            #     print("操作已取消")
            #     return False
        
        # 创建目录结构
        unique_labels = self.create_output_structure(labels, output_dir, copy_mode)
        
        # 开始重组
        print(f"\n开始重新组织图像...")
        successful_operations = 0
        failed_operations = []
        
        operation_func = shutil.copy2 if copy_mode else shutil.move
        operation_name = "复制" if copy_mode else "移动"
        
        for i, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), 
                                                  total=len(image_paths), 
                                                  desc=f"{operation_name}图像")):
            try:
                # 构造目标路径
                filename = os.path.basename(img_path)
                target_dir = os.path.join(output_dir, f"cluster_{label}")
                target_path = os.path.join(target_dir, filename)
                
                # 处理文件名冲突
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_path):
                        new_filename = f"{base}_{counter}{ext}"
                        target_path = os.path.join(target_dir, new_filename)
                        counter += 1
                
                # 执行操作
                if os.path.exists(img_path):
                    operation_func(img_path, target_path)
                    successful_operations += 1
                    
                    # 记录操作日志
                    self.reorganization_log.append({
                        'original_path': img_path,
                        'new_path': target_path,
                        'cluster': int(label),
                        'operation': operation_name,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    failed_operations.append(f"文件不存在: {img_path}")
                    
            except Exception as e:
                failed_operations.append(f"{img_path}: {str(e)}")
        
        # 操作完成报告
        print(f"\n{'='*50}")
        print(f"重新组织完成！")
        print(f"{'='*50}")
        print(f"✅ 成功{operation_name}: {successful_operations} 张图像")
        
        if failed_operations:
            print(f"❌ 失败: {len(failed_operations)} 张图像")
            print("失败详情:")
            for error in failed_operations[:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(failed_operations) > 5:
                print(f"  ... 还有 {len(failed_operations) - 5} 个错误")
        
        # 保存操作日志和统计信息
        self.save_metadata(output_dir, labels, successful_operations, failed_operations)
        
        return len(failed_operations) == 0
    
    def save_metadata(self, output_dir, labels, successful_count, failed_operations):
        """
        保存元数据和操作日志
        """
        metadata_dir = os.path.join(output_dir, "_metadata")
        
        # 1. 保存聚类统计
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_stats = {}
        
        for label, count in zip(unique_labels, counts):
            cluster_dir = os.path.join(output_dir, f"cluster_{label}")
            actual_count = len([f for f in os.listdir(cluster_dir) 
                              if not f.startswith('.')])
            cluster_stats[f"cluster_{label}"] = {
                'expected_count': int(count),
                'actual_count': actual_count,
                'directory': cluster_dir
            }
        
        # 保存统计信息
        stats_file = os.path.join(metadata_dir, "cluster_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
        
        # 2. 保存操作日志
        log_file = os.path.join(metadata_dir, "reorganization_log.json")
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(self.reorganization_log),
            'successful_operations': successful_count,
            'failed_operations': len(failed_operations),
            'failed_details': failed_operations,
            'operations': self.reorganization_log
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 3. 保存聚类标签
        labels_file = os.path.join(metadata_dir, "cluster_labels.npy")
        np.save(labels_file, labels)
        
        print(f"\n📄 元数据已保存到: {metadata_dir}")
        print(f"  - cluster_statistics.json: 聚类统计")
        print(f"  - reorganization_log.json: 操作日志")
        print(f"  - cluster_labels.npy: 聚类标签备份")
    
    def create_cluster_preview(self, output_dir, max_images_per_cluster=9):
        """
        为每个聚类创建预览图
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            print(f"\n生成聚类预览图...")
            
            cluster_dirs = [d for d in os.listdir(output_dir) 
                          if d.startswith('cluster_') and 
                          os.path.isdir(os.path.join(output_dir, d))]
            
            for cluster_dir in cluster_dirs:
                cluster_path = os.path.join(output_dir, cluster_dir)
                cluster_id = cluster_dir.replace('cluster_', '')
                
                # 获取该聚类中的图像
                image_files = [f for f in os.listdir(cluster_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not image_files:
                    continue
                
                # 随机选择几张图像作为预览
                import random
                preview_images = random.sample(image_files, 
                                             min(max_images_per_cluster, len(image_files)))
                
                # 创建预览图
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle(f'Cluster {cluster_id} 预览 ({len(image_files)} 张图像)', 
                           fontsize=16)
                
                for i, ax in enumerate(axes.flat):
                    if i < len(preview_images):
                        img_path = os.path.join(cluster_path, preview_images[i])
                        try:
                            img = Image.open(img_path)
                            ax.imshow(img)
                            ax.set_title(preview_images[i], fontsize=8)
                        except Exception as e:
                            ax.text(0.5, 0.5, f'加载失败\n{str(e)}', 
                                  ha='center', va='center', transform=ax.transAxes)
                    
                    ax.axis('off')
                
                # 保存预览图
                preview_path = os.path.join(output_dir, "_metadata", 
                                          f"cluster_{cluster_id}_preview.png")
                plt.savefig(preview_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"✅ 预览图已生成到: {os.path.join(output_dir, '_metadata')}")
            
        except ImportError:
            print("⚠️  缺少matplotlib或PIL，跳过预览图生成")
        except Exception as e:
            print(f"⚠️  生成预览图失败: {e}")

def reorganize_with_original_clustering():
    """
    使用原始6类聚类结果重新组织图像
    """
    print("=== 使用原始聚类结果重新组织图像 ===\n")
    
    # 配置参数
    image_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480souce"  # 替换为你的原始图像目录
    labels_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析/labels.npy"  # 原始6类标签
    output_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480v2"
    
    reorganizer = ImageReorganizer()
    
    # 1. 获取图像路径
    image_paths = reorganizer.get_image_paths(image_dir)
    
    # 2. 加载聚类结果
    labels, features_2d = reorganizer.load_clustering_results(labels_path)
    
    if labels is None or len(image_paths) == 0:
        print("❌ 无法加载数据，请检查路径")
        return
    
    # 3. 重新组织图像
    success = reorganizer.reorganize_images(
        image_paths, labels, output_dir, 
        copy_mode=True,  # True=复制，False=移动
        confirm_before_start=True
    )
    
    # 4. 生成预览图
    if success:
        reorganizer.create_cluster_preview(output_dir)
    
    return reorganizer

def reorganize_with_improved_clustering():
    """
    使用改进的8类聚类结果重新组织图像
    """
    print("=== 使用改进聚类结果重新组织图像 ===\n")
    
    # 配置参数
    image_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480souce"  # 替换为你的原始图像目录
    labels_path = "/home/panxiang/coding/kweilx/ultralytics/clustering_results/improved_labels.npy"  # 改进的8类标签
    output_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析_V2"
    
    reorganizer = ImageReorganizer()
    
    # 1. 获取图像路径
    image_paths = reorganizer.get_image_paths(image_dir)
    
    # 2. 加载聚类结果
    labels, features_2d = reorganizer.load_clustering_results(labels_path)
    
    if labels is None or len(image_paths) == 0:
        print("❌ 无法加载数据，请检查路径")
        return
    
    # 3. 重新组织图像
    success = reorganizer.reorganize_images(
        image_paths, labels, output_dir, 
        copy_mode=True,  # True=复制，False=移动
        confirm_before_start=True
    )
    
    # 4. 生成预览图
    if success:
        reorganizer.create_cluster_preview(output_dir)
    
    return reorganizer

def compare_both_organizations():
    """
    同时创建6类和8类的组织结果，便于对比
    """
    print("=== 创建两种聚类结果的对比 ===\n")
    
    # 1. 使用原始6类聚类
    print("1. 创建6类聚类组织...")
    reorganizer_6 = reorganize_with_original_clustering()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 使用改进8类聚类
    print("2. 创建8类聚类组织...")
    reorganizer_8 = reorganize_with_improved_clustering()
    
    print(f"\n{'='*50}")
    print("对比完成！")
    print(f"{'='*50}")
    print("你现在可以对比两个文件夹：")
    print("📁 reorganized_6_clusters/ - 6类聚类结果")
    print("📁 reorganized_8_clusters/ - 8类聚类结果")
    print("\n建议：")
    print("1. 查看各自的 _metadata/ 目录中的预览图")
    print("2. 对比聚类质量，选择更合适的版本")
    print("3. 删除不需要的版本")

# 使用示例
def main():
    """
    主函数 - 选择你需要的重组方式
    """
    print("图像重组工具")
    print("="*30)
    print("选择重组方式：")
    print("1. 仅使用原始6类聚类")
    print("2. 仅使用改进8类聚类") 
    print("3. 同时创建两种版本进行对比")
    
    # 在实际使用时，取消注释下面的交互选择
    # choice = input("请选择 (1/2/3): ")
    
    # 示例：直接运行对比版本
    choice = "3"
    
    if choice == "1":
        reorganize_with_original_clustering()
    elif choice == "2":
        reorganize_with_improved_clustering()
    elif choice == "3":
        compare_both_organizations()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()