source_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/s_anomaly/source_yolo'
save_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/s_anomaly/process_yolo'

import os
import shutil

cnt = 0
for files in os.listdir(source_folder):
    try:
        if str(files).lower().endswith('.txt'):
            _name = str(files).split('.')[0]
            shutil.move(os.path.join(source_folder, files), os.path.join(save_folder, f'syn_{files}'))
            shutil.move(os.path.join(source_folder, f'{_name}.png'), os.path.join(save_folder, f'syn_{_name}.png'))
            cnt += 1
    except:
        continue
print(f"共处理{cnt}对文件")
        