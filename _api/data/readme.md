# 表
## 1、camera_collect 表： 记录所有从前端摄像头采集的图像
### 字段：file_id,file_folder_id,file_path,file_folder_path,file_real_name,file_type,file_comment,is_detected,file_create_time,update_time
file_id：唯一文件标识id
file_folder_id：唯一文件夹标识id，所在的文件夹
file_path：实际的文件存储路径
file_folder_path：实际的所在文件夹路径
file_real_name：文件真实名称
file_type：文件类型
file_comment：文件备注
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间

## 2、weights_val 表： 模型权重文件验证信息
### 字段：file_id,file_folder_id,file_path,file_folder_path,file_real_name,file_type,file_comment,is_detected,file_create_time,update_time
file_id：唯一文件标识id
file_folder_id：唯一文件夹标识id，所在的文件夹
file_path：实际的文件存储路径
file_folder_path：实际的所在文件夹路径
file_real_name：文件真实名称
file_type：文件类型
file_comment：文件备注
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间

## 3、datasets 表： 数据集信息
### 字段：file_id,file_folder_id,file_path,file_folder_path,file_real_name,file_type,file_comment,is_detected,file_create_time,update_time
file_id：唯一文件标识id
file_folder_id：唯一文件夹标识id，所在的文件夹
file_path：实际的文件存储路径
file_folder_path：实际的所在文件夹路径
file_real_name：文件真实名称
file_type：文件类型
file_comment：文件备注
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间

## 4、upload_images 表： 上传图像的信息
### 字段：file_id,file_folder_id,file_path,file_folder_path,file_real_name,file_type,file_comment,is_detected,file_create_time,update_time
file_id：唯一文件标识id
file_folder_id：唯一文件夹标识id，所在的文件夹
file_path：实际的文件存储路径
file_folder_path：实际的所在文件夹路径
file_real_name：文件真实名称
file_type：文件类型
file_comment：文件备注
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间

## 5、other_files 表： 其他一般文件的信息
### 字段：file_id,file_folder_id,file_path,file_folder_path,file_real_name,file_type,file_comment,is_detected,file_create_time,update_time
file_id：唯一文件标识id
file_folder_id：唯一文件夹标识id，所在的文件夹
file_path：实际的文件存储路径
file_folder_path：实际的所在文件夹路径
file_real_name：文件真实名称
file_type：文件类型
file_comment：文件备注
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间

## 6、detect_informations 表： 检测信息
### 字段：file_id,weight_id,is_detected,file_create_time,update_time
file_id：唯一文件标识id
weight_id：模型权重id
is_detected：检测结果
file_create_time：创建时间
update_time：更新时间