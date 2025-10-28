import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
import seaborn as sns

class DINOv2FeatureExtractor:
    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        """
        初始化 DINOv2 特征提取器
        
        Args:
            model_name: 模型名称 ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 加载预训练的 DINOv2 模型
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理管道
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_features(self, image_paths, batch_size=32):
        """
        批量提取图像特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
            
        Returns:
            features: 特征矩阵 (n_images, feature_dim)
        """
        features = []
        
        print(f"开始提取 {len(image_paths)} 张图像的特征...")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # 加载并预处理批次图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"加载图像 {path} 失败: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # 转换为批次张量
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                features.append(batch_features.cpu().numpy())
        
        # 合并所有特征
        features = np.vstack(features)
        print(f"特征提取完成，特征维度: {features.shape}")
        
        return features

class ImageClusterAnalysis:
    def __init__(self, n_clusters=5):
        """
        图像聚类分析器
        
        Args:
            n_clusters: 聚类数量
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=50)  # 先用PCA降维到50维
        self.tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
    def cluster_features(self, features):
        """
        对特征进行聚类
        
        Args:
            features: 特征矩阵
            
        Returns:
            cluster_labels: 聚类标签
            cluster_centers: 聚类中心
        """
        print("开始聚类分析...")
        
        # 标准化特征
        features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # PCA降维
        print("进行PCA降维...")
        features_pca = self.pca.fit_transform(features_normalized)
        
        # K-means聚类
        print(f"进行K-means聚类，分为{self.n_clusters}类...")
        cluster_labels = self.kmeans.fit_predict(features_pca)
        
        # 获取聚类中心
        cluster_centers = self.kmeans.cluster_centers_
        
        print("聚类完成！")
        return cluster_labels, cluster_centers, features_pca
    
    def visualize_clusters(self, features_pca, labels, image_paths=None, save_path=None):
        """
        可视化聚类结果
        
        Args:
            features_pca: PCA降维后的特征
            labels: 聚类标签
            image_paths: 图像路径列表
            save_path: 保存路径
        """
        print("生成t-SNE可视化...")
        
        # t-SNE降维到2D用于可视化
        features_2d = self.tsne.fit_transform(features_pca)
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        
        # 使用不同颜色表示不同聚类
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        for i in range(self.n_clusters):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
        
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.title('DINOv2特征聚类可视化 (t-SNE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return features_2d
    
    def analyze_clusters(self, labels, image_paths):
        """
        分析聚类结果
        
        Args:
            labels: 聚类标签
            image_paths: 图像路径列表
        """
        print("\n=== 聚类分析结果 ===")
        
        # 统计每个聚类的样本数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels) * 100
            print(f"聚类 {label}: {count} 张图像 ({percentage:.1f}%)")
        
        # 创建聚类分布饼图
        plt.figure(figsize=(8, 6))
        plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
                autopct='%1.1f%%', startangle=90)
        plt.title('聚类分布')
        plt.axis('equal')
        plt.show()
        
        return dict(zip(unique_labels, counts))

def save_cluster_results(labels, image_paths, output_dir):
    """
    保存聚类结果，将图像按聚类分组
    
    Args:
        labels: 聚类标签
        image_paths: 图像路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个聚类创建子目录
    for cluster_id in np.unique(labels):
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)
    
    # 复制图像到对应聚类目录
    import shutil
    
    for img_path, label in zip(image_paths, labels):
        filename = os.path.basename(img_path)
        dst_path = os.path.join(output_dir, f'cluster_{label}', filename)
        shutil.copy2(img_path, dst_path)
    
    print(f"聚类结果已保存到: {output_dir}")

def main():
    """
    主函数：完整的聚类分析流程
    """
    # 配置参数
    image_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480souce"  # 替换为你的图像目录
    output_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/1480聚类分析"   # 聚类结果输出目录
    n_clusters = 6                      # 聚类数量
    model_name = 'dinov2_vitb14'       # DINOv2模型名称
    
    # 获取图像路径
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    if len(image_paths) == 0:
        print("未找到图像文件，请检查图像目录路径")
        return
    
    # 1. 初始化特征提取器
    print("初始化DINOv2特征提取器...")
    feature_extractor = DINOv2FeatureExtractor(model_name=model_name)
    
    # 2. 提取特征
    features = feature_extractor.extract_features(image_paths, batch_size=16)
    
    # 3. 聚类分析
    cluster_analyzer = ImageClusterAnalysis(n_clusters=n_clusters)
    labels, centers, features_pca = cluster_analyzer.cluster_features(features)
    
    # 4. 可视化结果
    features_2d = cluster_analyzer.visualize_clusters(
        features_pca, labels, image_paths, 
        save_path=os.path.join(output_dir, 'cluster_visualization.png')
    )
    
    # 5. 分析聚类结果
    cluster_stats = cluster_analyzer.analyze_clusters(labels, image_paths)
    
    # 6. 保存结果
    save_cluster_results(labels, image_paths, output_dir)
    
    # 7. 保存特征和标签
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    np.save(os.path.join(output_dir, 'features_2d.npy'), features_2d)
    
    print("聚类分析完成！")
    
    return features, labels, cluster_stats

# 示例用法
if __name__ == "__main__":
    # 运行完整的聚类分析流程
    features, labels, stats = main()
    
    # 你也可以单独使用各个组件
    # 例如，只进行特征提取：
    # extractor = DINOv2FeatureExtractor()
    # features = extractor.extract_features(your_image_paths)