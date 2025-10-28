"""
输出每一层的参数,计算量
"""
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/12/yolo12.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()