import sys

sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

from ultralytics import YOLO
import os
import shutil
from _project.mydata._code.yolo._val import standard_val
from _project.mydata._code.yolo._train import standard_train



def batch_train_by_yamls_list(
    data_yaml:str,
    save_path:str,
    yaml_path:str):
    """
    训练同一份数据，遍历yaml文件下的所有yaml文件，依次训练并验证
    Args:
        data_yaml (str): _description_
        save_path (str): _description_
        yaml_path (str): 文件夹路径
    """
    yaml_path_list = []
    if os.path.isfile(yaml_path):
        yaml_path_list.append(yaml_path)
    else:
        for file in os.listdir(yaml_path):
            if file.endswith('.yaml'):
                yaml_path_list.append(os.path.join(yaml_path,file))
                
    for config_yaml in yaml_path_list:
        try:
            config_yaml_name = str(os.path.basename(config_yaml)).split('.')[0]
            # train
            train_save_path = standard_train(
                config_yaml_path=config_yaml,
                data_yaml=data_yaml,
                desc=config_yaml_name,
                save_path=save_path,
                epochs=200,
                batch_size = 32,
                )
            # val
            standard_val(
                val_data = {'model':train_save_path,'yaml_data':data_yaml},
                only_one = False
                ) # ->list

            # 训练结束，移至v8ok
            shutil.move(config_yaml,os.path.join('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/sussess',str(os.path.basename(config_yaml))))
        except Exception as e:
            with open('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/erro/erro_log.txt','a') as f:
                f.write(f"{os.path.basename(config_yaml)}\n")
                f.write(f"{e}\n")

                # 训练出错，移至v8erro
                # shutil.move(config_yaml,os.path.join('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/erro',str(os.path.basename(config_yaml))))
            

def batch_batch_train_by_yamls_list(
    data_yaml_list:str|list,
    save_path:str,
    yaml_path:str):
    """
    训练多份数据，遍历yaml文件下的所有yaml文件，依次训练并验证
    Args:
        data_yaml (str): _description_
        save_path (str): _description_
        yaml_path (str): 文件夹路径
    """
    data_yaml_list = [data_yaml_list] if isinstance(data_yaml_list,str) else data_yaml_list
    
    yaml_path_list = []
    if os.path.isfile(yaml_path):
        yaml_path_list.append(yaml_path)
    else:
        for file in os.listdir(yaml_path):
            if file.endswith('.yaml'):
                yaml_path_list.append(os.path.join(yaml_path,file))
    
    for data_yaml in data_yaml_list:    
        for config_yaml in yaml_path_list:
        # try:
            config_yaml_name = str(os.path.basename(config_yaml)).split('.')[0]
            # train
            train_save_path = standard_train(
                config_yaml_path=config_yaml,
                data_yaml=data_yaml,
                desc=config_yaml_name,
                save_path=save_path,
                epochs=200,
                batch_size = 32,
                )
            # val
            standard_val(
                val_data = {'model':train_save_path,'yaml_data':data_yaml},
                only_one = True
                ) # ->list
# 
            # 训练结束，移至v8ok
            # shutil.move(config_yaml,os.path.join('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/sussess',str(os.path.basename(config_yaml))))
        # except Exception as e:
        #     with open('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/erro/erro_log.txt','a') as f:
        #         f.write(f"{os.path.basename(config_yaml)}\n")
        #         f.write(f"{e}\n")

                # 训练出错，移至v8erro
                # shutil.move(config_yaml,os.path.join('/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/erro',str(os.path.basename(config_yaml))))
            



if __name__ == '__main__':
    data_yaml= [     
         "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/mydata.yaml",
    ]
    save_path = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes'
    yaml_path = '/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/Add_C'
    batch_batch_train_by_yamls_list(data_yaml,save_path,yaml_path)