import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def safe_save_results(improved_labels, base_dir="./"):
    """
    安全保存改进结果，并验证保存是否成功
    """
    print("=== 保存改进结果 ===")
    
    # 1. 确保目录存在
    save_dir = os.path.join(base_dir, "clustering_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {os.path.abspath(save_dir)}")
    
    # 2. 定义文件路径
    improved_labels_path = os.path.join(save_dir, "improved_labels.npy")
    
    try:
        # 3. 保存文件
        np.save(improved_labels_path, improved_labels)
        print(f"✅ 尝试保存到: {os.path.abspath(improved_labels_path)}")
        
        # 4. 验证保存是否成功
        if os.path.exists(improved_labels_path):
            # 尝试重新加载验证
            test_load = np.load(improved_labels_path)
            if len(test_load) == len(improved_labels):
                print(f"✅ 保存成功！文件大小: {len(test_load)} 个标签")
                print(f"✅ 文件位置: {os.path.abspath(improved_labels_path)}")
                return improved_labels_path
            else:
                print(f"❌ 保存失败：文件内容不匹配")
        else:
            print(f"❌ 保存失败：文件不存在")
            
    except Exception as e:
        print(f"❌ 保存出错: {e}")
    
    # 5. 备用保存方案：保存到当前目录
    backup_path = "improved_labels_backup.npy"
    try:
        np.save(backup_path, improved_labels)
        print(f"🔄 备用保存: {os.path.abspath(backup_path)}")
        return backup_path
    except Exception as e:
        print(f"❌ 备用保存也失败: {e}")
        return None

def compare_clustering_results(original_labels, improved_labels, features_pca):
    """
    对比原始和改进的聚类结果
    """
    print("\n=== 聚类结果对比 ===")
    
    # 1. 基本统计对比
    orig_clusters = len(np.unique(original_labels))
    impr_clusters = len(np.unique(improved_labels))
    
    print(f"原始聚类数: {orig_clusters}")
    print(f"改进聚类数: {impr_clusters}")
    
    # 2. 质量指标对比
    orig_silhouette = silhouette_score(features_pca, original_labels)
    impr_silhouette = silhouette_score(features_pca, improved_labels)
    
    print(f"\n质量对比:")
    print(f"原始轮廓系数: {orig_silhouette:.3f}")
    print(f"改进轮廓系数: {impr_silhouette:.3f}")
    
    if impr_silhouette > orig_silhouette:
        improvement = ((impr_silhouette - orig_silhouette) / orig_silhouette) * 100
        print(f"✅ 改进了 {improvement:.1f}%")
    else:
        decline = ((orig_silhouette - impr_silhouette) / orig_silhouette) * 100
        print(f"❌ 下降了 {decline:.1f}%")
    
    # 3. 聚类大小分布对比
    print(f"\n聚类大小分布对比:")
    print("原始:")
    orig_unique, orig_counts = np.unique(original_labels, return_counts=True)
    for cluster_id, count in zip(orig_unique, orig_counts):
        percentage = count / len(original_labels) * 100
        print(f"  Cluster {cluster_id}: {count} ({percentage:.1f}%)")
    
    print("改进:")
    impr_unique, impr_counts = np.unique(improved_labels, return_counts=True)
    for cluster_id, count in zip(impr_unique, impr_counts):
        percentage = count / len(improved_labels) * 100
        print(f"  Cluster {cluster_id}: {count} ({percentage:.1f}%)")

def create_comparison_visualization(original_labels, improved_labels, features_2d, save_path=None):
    """
    创建对比可视化图
    """
    print("\n=== 生成对比可视化 ===")
    
    # 创建并排对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始聚类图
    n_orig_clusters = len(np.unique(original_labels))
    colors1 = plt.cm.Set3(np.linspace(0, 1, n_orig_clusters))
    
    for i in range(n_orig_clusters):
        mask = original_labels == i
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors1[i]], label=f'Cluster {i}', alpha=0.7, s=30)
    
    ax1.set_title('原始聚类结果 (6类)', fontsize=14)
    ax1.set_xlabel('t-SNE dimension 1')
    ax1.set_ylabel('t-SNE dimension 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 改进聚类图
    n_impr_clusters = len(np.unique(improved_labels))
    colors2 = plt.cm.Set3(np.linspace(0, 1, n_impr_clusters))
    
    for i in range(n_impr_clusters):
        mask = improved_labels == i
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors2[i]], label=f'Cluster {i}', alpha=0.7, s=30)
    
    ax2.set_title(f'改进聚类结果 ({n_impr_clusters}类)', fontsize=14)
    ax2.set_xlabel('t-SNE dimension 1')
    ax2.set_ylabel('t-SNE dimension 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 对比图保存到: {os.path.abspath(save_path)}")
        except Exception as e:
            backup_path = "clustering_comparison.png"
            plt.savefig(backup_path, dpi=300, bbox_inches='tight')
            print(f"🔄 对比图备用保存: {os.path.abspath(backup_path)}")
    
    plt.show()

def use_improved_labels_workflow():
    """
    使用改进标签的完整工作流程
    """
    print("=== 改进标签使用指南 ===\n")
    
    try:
        # 1. 加载原始数据
        print("1. 加载原始数据...")
        original_labels = np.load('/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析/labels.npy')
        features_2d = np.load('/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析/features_2d.npy')
        features = np.load('/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析/features.npy')
        
        # 2. 重新生成PCA特征（如果需要）
        print("2. 处理特征数据...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        features_pca = pca.fit_transform(features_normalized)
        
        # 3. 生成改进的聚类（示例：使用8个聚类）
        print("3. 生成改进聚类...")
        improved_kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        improved_labels = improved_kmeans.fit_predict(features_pca)
        
        # 4. 安全保存
        print("4. 保存改进结果...")
        saved_path = safe_save_results(improved_labels)
        
        if saved_path:
            # 5. 对比分析
            print("5. 对比分析...")
            compare_clustering_results(original_labels, improved_labels, features_pca)
            
            # 6. 生成对比可视化
            print("6. 生成对比可视化...")
            create_comparison_visualization(
                original_labels, improved_labels, features_2d,
                save_path="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析/clustering_comparison.png"
            )
            
            print(f"\n{'='*50}")
            print("🎉 改进流程完成！")
            print(f"{'='*50}")
            print("生成的文件:")
            print(f"- 改进标签: {saved_path}")
            print(f"- 对比图: clustering_comparison.png")
            print("\n现在你可以:")
            print("1. 查看对比图，比较新旧聚类效果")
            print("2. 如果满意，使用新标签重新组织图像文件夹")
            print("3. 如果不满意，继续使用原始标签")
            
            return improved_labels, saved_path
        else:
            print("❌ 保存失败，请检查文件权限和路径")
            return None, None
            
    except FileNotFoundError as e:
        print(f"❌ 找不到原始数据文件: {e}")
        print("请确保以下文件存在:")
        print("- clustering_results/labels.npy")
        print("- clustering_results/features_2d.npy") 
        print("- clustering_results/features.npy")
        return None, None
    except Exception as e:
        print(f"❌ 处理过程出错: {e}")
        return None, None

def reorganize_images_with_improved_labels(improved_labels_path, original_image_paths, output_dir="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析"):
    """
    使用改进的标签重新组织图像文件
    """
    print(f"\n=== 使用改进标签重新组织图像 ===")
    
    try:
        # 加载改进的标签
        improved_labels = np.load(improved_labels_path)
        print(f"✅ 加载改进标签: {len(improved_labels)} 个标签")
        
        # 创建新的输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个新聚类创建目录
        unique_labels = np.unique(improved_labels)
        for label in unique_labels:
            cluster_dir = os.path.join(output_dir, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)
        
        # 复制图像到新的聚类目录
        import shutil
        for img_path, label in zip(original_image_paths, improved_labels):
            if os.path.exists(img_path):
                filename = os.path.basename(img_path)
                dst_path = os.path.join(output_dir, f"cluster_{label}", filename)
                shutil.copy2(img_path, dst_path)
        
        print(f"✅ 图像重新组织完成！")
        print(f"✅ 新的聚类文件夹: {os.path.abspath(output_dir)}")
        
        # 统计每个聚类的图像数量
        for label in unique_labels:
            cluster_dir = os.path.join(output_dir, f"cluster_{label}")
            count = len(os.listdir(cluster_dir))
            print(f"   Cluster {label}: {count} 张图像")
            
    except Exception as e:
        print(f"❌ 重新组织图像失败: {e}")

# 主函数：完整的使用流程
def main():
    """
    完整的改进聚类使用流程
    """
    print("开始改进聚类分析...\n")
    
    # 运行改进流程
    improved_labels, saved_path = use_improved_labels_workflow()
    
    if improved_labels is not None and saved_path is not None:
        print(f"\n成功！现在你有了改进的聚类结果。")
        print(f"改进标签文件: {saved_path}")
        
        # 询问是否要重新组织图像文件
        # reorganize = input("\n是否要用新标签重新组织图像文件夹？(y/n): ")
        # if reorganize.lower() == 'y':
        #     # 这里需要你提供原始图像路径列表
        #     # image_paths = [...] # 你的图像路径列表
        #     # reorganize_images_with_improved_labels(saved_path, image_paths)

if __name__ == "__main__":
    main()