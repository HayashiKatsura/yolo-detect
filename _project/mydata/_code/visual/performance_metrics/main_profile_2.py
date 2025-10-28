import torch, thop
from thop import profile
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')
from ultralytics import YOLO, RTDETR
from prettytable import PrettyTable

if __name__ == '__main__':
    batch_size, height, width = 1, 640, 640
    
    yaml_list = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060012_yolo12_Train2419_Val479/weights/best.pt",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508052210_yolo12-convnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt"
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/lamp/202508271425-prune-lamp-convnextv2-mutilscale36912-reg0.05-gt-sp1.5-finetune/weights/best.pt",
        "",
        "",
        "",
        "",
    ]
    
    for yaml_path in yaml_list:

        model = YOLO(yaml_path).model # select your model.pt path
        # model = RTDETR(r'ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml').model
        model.fuse()
        input = torch.randn(batch_size, 3, height, width)
        total_flops, total_params, layers = profile(model, [input], verbose=True, ret_layer_info=True)
        FLOPs, Params = thop.clever_format([total_flops * 2 / batch_size, total_params], "%.3f")
        table = PrettyTable()
        table.title = f'Model Flops:{FLOPs} Params:{Params}'
        table.field_names = ['Layer ID', "FLOPs", "Params"]
        for layer_id in layers['model'][2]:
            data = layers['model'][2][layer_id]
            FLOPs, Params = thop.clever_format([data[0] * 2 / batch_size, data[1]], "%.3f")
            table.add_row([layer_id, FLOPs, Params])
        print(table)