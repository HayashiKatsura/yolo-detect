-- 创建数据库 anomalies_detect
DROP DATABASE IF EXISTS anomalies_detect;
CREATE DATABASE IF NOT EXISTS anomalies_detect;

-- 使用 anomalies_detect 数据库
USE anomalies_detect;

-- 创建 files 表
CREATE TABLE IF NOT EXISTS files (
                                     file_id VARCHAR(255) PRIMARY KEY NOT NULL UNIQUE COMMENT '唯一文件Id',
                                     file_path VARCHAR(255) COMMENT '实际文件路径',
                                     folder_path VARCHAR(255) COMMENT '实际文件所在文件夹路径',
                                     file_name VARCHAR(255) COMMENT '真实文件名',
                                     type VARCHAR(255) COMMENT '文件类型',
                                     comment VARCHAR(255) COMMENT '文件描述',
                                     is_detected VARCHAR(255) DEFAULT NULL COMMENT '检测信息',
                                     create_time VARCHAR(255) DEFAULT NULL COMMENT '创建时间',
                                     update_time VARCHAR(255) DEFAULT NULL COMMENT '更新时间',
                                     is_delete BOOLEAN DEFAULT FALSE COMMENT '是否删除'
);

-- 创建 detections 表
CREATE TABLE IF NOT EXISTS detections (
                                          file_id VARCHAR(255) PRIMARY KEY NOT NULL UNIQUE COMMENT '文件id',
                                          weight_id VARCHAR(255) COMMENT '权重id',
                                          details JSON COMMENT '检测信息细节',
                                          create_time VARCHAR(255) DEFAULT NULL COMMENT '创建时间',
                                          update_time VARCHAR(255) DEFAULT NULL COMMENT '更新时间'
);

-- 创建 datasets 表
CREATE TABLE IF NOT EXISTS datasets (
                                        file_id VARCHAR(255) PRIMARY KEY NOT NULL UNIQUE COMMENT '唯一文件id',
                                        file_path VARCHAR(255) COMMENT '文件路径',
                                        file_name VARCHAR(255) COMMENT '文件名称',
                                        yaml_path VARCHAR(255) COMMENT '配置文件路径',
                                        images_folder VARCHAR(255) COMMENT '图像文件夹路径',
                                        labels_folder VARCHAR(255) COMMENT '标签文件夹路径',
                                        train_counts INT DEFAULT 0 COMMENT '训练集数量',
                                        last_train_counts INT DEFAULT 0 COMMENT '上一次训练集数量',
                                        val_counts INT DEFAULT 0 COMMENT '验证集数量',
                                        last_val_counts INT DEFAULT 0 COMMENT '上一次验证集数量',
                                        create_time VARCHAR(255) DEFAULT NULL COMMENT '创建时间',
                                        update_time VARCHAR(255) DEFAULT NULL COMMENT '更新时间',
                                        is_delete BOOLEAN DEFAULT FALSE COMMENT '是否删除'
);

-- 创建 weights 表
CREATE TABLE IF NOT EXISTS weights (
                                       file_id VARCHAR(255) PRIMARY KEY NOT NULL UNIQUE COMMENT '唯一文件id',
                                       file_path VARCHAR(255) COMMENT '文件路径',
                                       folder_path VARCHAR(255) COMMENT '文件夹路径',
                                       file_name VARCHAR(255) COMMENT '文件名称',
                                       dataset_id VARCHAR(255) DEFAULT NULL COMMENT '数据集id',
                                       is_validated VARCHAR(255) DEFAULT NULL COMMENT '是否通过验证',
                                       session_id VARCHAR(255) DEFAULT NULL COMMENT '会话id',
                                       train_log VARCHAR(255) DEFAULT NULL COMMENT '训练日志',
                                       create_time VARCHAR(255) DEFAULT NULL COMMENT '创建时间',
                                       update_time VARCHAR(255) DEFAULT NULL COMMENT '更新时间',
                                       is_delete BOOLEAN DEFAULT FALSE COMMENT '是否删除'
);
