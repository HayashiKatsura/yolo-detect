import os
import zipfile
from pathlib import Path
import rarfile
import py7zr
import logging
from loguru import logger


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class ArchiveExtractor:
    """
    多格式压缩文件解压工具
    支持 .zip, .rar, .7z 格式
    """
    
    def __init__(self):
        self.supported_formats = {
            '.zip': self._extract_zip,
            '.rar': self._extract_rar,
            '.7z': self._extract_7z
        }
    
    def extract(self, archive_path, target_dir=None):
        """
        解压压缩文件
        
        Args:
            archive_path (str): 压缩文件路径
            target_dir (str, optional): 目标目录，默认为压缩文件名（无扩展名）
        
        Returns:
            bool: 解压是否成功
        """
        archive_path = Path(archive_path)
        
        # 检查文件是否存在
        if not archive_path.exists():
            logger.info(f"错误: 文件 '{archive_path}' 不存在")
            return False
        
        # 获取文件扩展名
        file_ext = archive_path.suffix.lower()
        
        # 检查是否支持该格式
        if file_ext not in self.supported_formats:
            logger.info(f"错误: 不支持的文件格式 '{file_ext}'")
            logger.info(f"支持的格式: {', '.join(self.supported_formats.keys())}")
            return False
        
        # 确定目标目录
        if target_dir is None:
            target_dir = archive_path.stem  # 文件名（无扩展名）
        
        target_path = Path(target_dir)
        
        # 创建目标目录
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"目标目录: {target_path.absolute()}")
        except Exception as e:
            logger.info(f"错误: 无法创建目录 '{target_path}': {e}")
            return False
        
        # 调用相应的解压方法
        extract_method = self.supported_formats[file_ext]
        return extract_method(archive_path, target_path)
    
    def _extract_zip(self, archive_path, target_path):
        """解压 ZIP 文件"""
        try:
            logger.info(f"正在解压 ZIP 文件: {archive_path}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_path)
            logger.info("ZIP 文件解压完成!")
            return True
        except zipfile.BadZipFile:
            logger.info("错误: 损坏的 ZIP 文件")
            return False
        except Exception as e:
            logger.info(f"解压 ZIP 文件时出错: {e}")
            return False
    
    def _extract_rar(self, archive_path, target_path):
        """解压 RAR 文件"""

        
        try:
            logger.info(f"正在解压 RAR 文件: {archive_path}")
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(target_path)
            logger.info("RAR 文件解压完成!")
            return True
        except rarfile.BadRarFile:
            logger.info("错误: 损坏的 RAR 文件")
            return False
        except rarfile.RarCannotExec:
            logger.info("错误: 找不到 RAR 解压工具，请安装 WinRAR 或 unrar")
            return False
        except Exception as e:
            logger.info(f"解压 RAR 文件时出错: {e}")
            return False
    
    def _extract_7z(self, archive_path, target_path):
        """解压 7Z 文件"""
        
        try:
            logger.info(f"正在解压 7Z 文件: {archive_path}")
            with py7zr.SevenZipFile(archive_path, mode='r') as sz_ref:
                sz_ref.extractall(target_path)
            logger.info("7Z 文件解压完成!")
            return True
        except py7zr.Bad7zFile:
            logger.info("错误: 损坏的 7Z 文件")
            return False
        except Exception as e:
            logger.info(f"解压 7Z 文件时出错: {e}")
            return False
    
    def list_files(self, archive_path):
        """
        列出压缩文件中的内容
        
        Args:
            archive_path (str): 压缩文件路径
            
        Returns:
            list: 文件列表，如果出错返回None
        """
        archive_path = Path(archive_path)
        file_ext = archive_path.suffix.lower()
        
        if not archive_path.exists():
            logger.info(f"错误: 文件 '{archive_path}' 不存在")
            return None
        
        try:
            if file_ext == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
            elif file_ext == '.rar':
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    file_list = rar_ref.namelist()
            elif file_ext == '.7z':
                with py7zr.SevenZipFile(archive_path, mode='r') as sz_ref:
                    file_list = sz_ref.getnames()
            else:
                logger.info(f"错误: 不支持的文件格式或缺少依赖库")
                return None
            
            logger.info(f"\n压缩文件 '{archive_path}' 包含 {len(file_list)} 个文件/文件夹:")
            logger.info("-" * 50)
            for file_name in file_list:
                logger.info(file_name)
            logger.info("-" * 50)
            return file_list
            
        except Exception as e:
            logger.info(f"读取压缩文件时出错: {e}")
            return None


# 便捷函数
def extract_archive(archive_path, target_dir=None):
    """
    解压压缩文件的便捷函数
    
    Args:
        archive_path (str): 压缩文件路径
        target_dir (str, optional): 目标目录，默认为压缩文件名（无扩展名）
    
    Returns:
        bool: 解压是否成功
    """
    extractor = ArchiveExtractor()
    return extractor.extract(archive_path, target_dir)


def list_archive_files(archive_path):
    """
    列出压缩文件内容的便捷函数
    
    Args:
        archive_path (str): 压缩文件路径
        
    Returns:
        list: 文件列表，如果出错返回None
    """
    extractor = ArchiveExtractor()
    return extractor.list_files(archive_path)


def extract_multiple_archives(archive_paths, base_target_dir=None):
    """
    批量解压多个压缩文件
    
    Args:
        archive_paths (list): 压缩文件路径列表
        base_target_dir (str, optional): 基础目标目录
    
    Returns:
        dict: 解压结果字典 {文件路径: 是否成功}
    """
    extractor = ArchiveExtractor()
    results = {}
    
    for archive_path in archive_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理文件: {archive_path}")
        logger.info('='*60)
        
        # 确定目标目录
        target_dir = None
        if base_target_dir:
            archive_name = Path(archive_path).stem
            target_dir = f"{base_target_dir}/{archive_name}"
        
        success = extractor.extract(archive_path, target_dir)
        results[archive_path] = success
    
    # 输出结果统计
    success_count = sum(results.values())
    total_count = len(archive_paths)
    logger.info(f"\n{'='*60}")
    logger.info(f"批量解压完成: {success_count}/{total_count} 个文件解压成功")
    logger.info('='*60)
    
    return results


# 使用示例
if __name__ == "__main__":
    # 示例1: 解压单个文件到默认目录（文件名）
    # extract_archive("test.zip")
    
    # # 示例2: 解压到指定目录
    extract_archive("/Users/katsura/Downloads/2a05381d-a539-424a-97db-3e4a751bc38d.rar", 
                    "/Users/katsura/Downloads")
    
    # 示例3: 列出压缩文件内容
    # files = list_archive_files("/home/panxiang/coding/kweilx/ultralytics/_api/data/dataset_example/dataset.zip")
    # if files:
    #     logger.info(f"找到 {len(files)} 个文件")
    
    # 示例4: 批量解压
    # archive_list = ["file1.zip", "file2.rar", "file3.7z"]
    # results = extract_multiple_archives(archive_list, "extracted")
    
    # 示例5: 使用类的方式
    # extractor = ArchiveExtractor()
    # extractor.extract("test.zip", "output_folder")
    # extractor.list_files("test.zip")
    
    # logger.info("请调用相应的函数来使用解压功能")
    # logger.info("主要函数:")
    # logger.info("- extract_archive(archive_path, target_dir=None)")
    # logger.info("- list_archive_files(archive_path)")
    # logger.info("- extract_multiple_archives(archive_paths, base_target_dir=None)")