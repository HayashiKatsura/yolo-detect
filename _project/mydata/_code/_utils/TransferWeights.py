from logging import raiseExceptions
import os
import sys
from pprint import pprint
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

import torch
from copy import deepcopy
import torch
import gc

def transfer_weights(current_weight:str,
                     dict_range:int,
                     special_dict_range:list = None,
                     standard_weight:str='/home/panxiang/coding/kweilx/ultralytics/yolov8n.pt'):
    save_path = os.path.join(os.path.dirname(current_weight), 'transfered_weights.pt')
    
    standard_weight = torch.load(standard_weight)
    current_weight = torch.load(current_weight)
    
    yolov8_dict =standard_weight['model'].state_dict()
    # current_dict = current_weight['model'].state_dict()
    
    new_dict={}
    if special_dict_range:
        special_dict_range = list(sorted(special_dict_range))
    else:
        special_dict_range = []
    
    if dict_range>=special_dict_range[0]:
        raise ValueError("范围参数应比具体参数小")
    
    dict_range = [dict_range] + special_dict_range
    for key,value in yolov8_dict.items():
        key_lst =key.split(".")
        for item in dict_range:
            if int(key_lst[1])<item:
                new_dict[key]=deepcopy(value)
                continue
        if int(key_lst[1]) == 22: # 固定参数
            continue
        key_lst[1] = str(int(key_lst[1])+1)
        new_dict[".".join(key_lst)] = deepcopy(value)
    current_weight['model'].load_state_dict(new_dict,strict=False)
    torch.save(current_weight,save_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path




# yolov8n = torch.load('/home/panxiang/coding/kweilx/ultralytics/yolov8n.pt')
# yolov8n_best = torch.load('/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolov8m/_NewModules/_vitv0_202504171516/weights/best.pt')

# yolov8_dict =yolov8n['model'].state_dict()
# best_dict =yolov8n_best['model'].state_dict()
# print(len(yolov8_dict),len(best_dict))
# # pprint(list(yolov8_dict.keys())[:50])

# from copy import deepcopy
# new_dict={}
# for key,value in yolov8_dict.items():
#     key_lst =key.split(".")
#     if int(key_lst[1])<10:
#         new_dict[key]=deepcopy(value)
#         continue
#     if int(key_lst[1]) == 22:
#         continue
#     key_lst[1] = str(int(key_lst[1])+1)
#     new_dict[".".join(key_lst)] = deepcopy(value)
# yolov8n_best['model'].load_state_dict(new_dict,strict=False)
# torch.save(yolov8n_best,'new_v8_weights.pt')
# print()