-- 1) 创建数据库（带字符集/排序规则）
DROP DATABASE IF EXISTS chips_yolo;
CREATE DATABASE chips_yolo
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE chips_yolo;

-- 2) files 表
CREATE TABLE `files` (
    `id` bigint unsigned NOT NULL AUTO_INCREMENT,
    `kind` varchar(50) NOT NULL,
    `content_type` varchar(100) DEFAULT NULL,
    `original_filename` varchar(255) NOT NULL,
    `storage_path` varchar(1024) NOT NULL,
    `size_bytes` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    `checksum_sha256` char(64) DEFAULT NULL,
    `media_annotations` json DEFAULT NULL,
    `model_metrics` json DEFAULT NULL,
    `remark` json DEFAULT NULL,
    `deleted_at` datetime(6) DEFAULT NULL,
    `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`id`)
) ENGINE = InnoDB AUTO_INCREMENT = 71 DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;

-- 3) datasets 表
CREATE TABLE `datasets` (
    `id` bigint unsigned NOT NULL AUTO_INCREMENT,
    `name` varchar(255) NOT NULL,
    `kind` varchar(50) NOT NULL DEFAULT 'yolo',
    `root_path` text NOT NULL,
    `yaml_path` text,
    `images_dir` varchar(255) DEFAULT NULL,
    `labels_dir` varchar(255) DEFAULT NULL,
    `train_counts` int unsigned NOT NULL DEFAULT '0',
    `last_train_counts` int unsigned NOT NULL DEFAULT '0',
    `val_counts` int unsigned NOT NULL DEFAULT '0',
    `last_val_counts` int unsigned NOT NULL DEFAULT '0',
    `test_counts` int unsigned NOT NULL DEFAULT '0',
    `last_test_counts` int unsigned NOT NULL DEFAULT '0',
    `size_bytes` bigint unsigned DEFAULT NULL,
    `remark` text,
    `deleted_at` datetime(6) DEFAULT NULL,
    `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`id`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;

-- 4) dataset_runs 表（训练记录）
CREATE TABLE `dataset_runs` (
    `id` bigint unsigned NOT NULL AUTO_INCREMENT,
    `dataset_id` bigint unsigned NOT NULL,
    `run_uuid` char(36) DEFAULT NULL,
    `model_file_id` bigint unsigned DEFAULT NULL,
    `status` varchar(32) NOT NULL DEFAULT 'queued',
    `started_at` datetime(6) DEFAULT NULL,
    `finished_at` datetime(6) DEFAULT NULL,
    `metrics` json DEFAULT NULL,
    `artifacts` json DEFAULT NULL,
    `params` json DEFAULT NULL,
    `remark` text,
    `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`id`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;