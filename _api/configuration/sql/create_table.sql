-- 1) 创建数据库（带字符集/排序规则）
DROP DATABASE IF EXISTS chips_yolo;
CREATE DATABASE chips_yolo
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE chips_yolo;

-- 2) files 表
CREATE TABLE `files` (
  `id`                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `kind`              VARCHAR(50) NOT NULL,          -- e.g. image/upload, image/camera, weights, dataset
  `content_type`      VARCHAR(100) NULL,             -- MIME
  `original_filename` VARCHAR(255) NOT NULL,
  `storage_path`      TEXT NOT NULL,                 -- 本地相对/绝对路径（建议相对）
  `size_bytes`        BIGINT UNSIGNED NULL,

  `media_annotations` JSON NULL,                     -- 图像/视频检测结果
  `model_metrics`     JSON NULL,                     -- 权重评估指标

  `remark`            TEXT NULL,
  `deleted_at`        DATETIME(6) NULL,
  `created_at`        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at`        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 3) datasets 表
CREATE TABLE `datasets` (
  `id`                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `name`              VARCHAR(255) NOT NULL,
  `kind`              VARCHAR(50)  NOT NULL DEFAULT 'yolo',   -- yolo/coco/voc/custom...
  `root_path`         TEXT NOT NULL,                           -- 建议相对路径
  `yaml_path`         TEXT NULL,
  `images_dir`        VARCHAR(255) NULL,
  `labels_dir`        VARCHAR(255) NULL,

  `train_counts`      INT UNSIGNED NOT NULL DEFAULT 0,
  `last_train_counts` INT UNSIGNED NOT NULL DEFAULT 0,
  `val_counts`        INT UNSIGNED NOT NULL DEFAULT 0,
  `last_val_counts`   INT UNSIGNED NOT NULL DEFAULT 0,
  `test_counts`       INT UNSIGNED NOT NULL DEFAULT 0,
  `last_test_counts`  INT UNSIGNED NOT NULL DEFAULT 0,

  `size_bytes`        BIGINT UNSIGNED NULL,
  `remark`            TEXT NULL,

  `deleted_at`        DATETIME(6) NULL,
  `created_at`        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at`        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 4) dataset_runs 表（训练记录）
CREATE TABLE `dataset_runs` (
  `id`            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `dataset_id`    BIGINT UNSIGNED NOT NULL,          -- 逻辑外键：指向 datasets.id（不加 FK 约束）
  `run_uuid`      CHAR(36) NULL,                     -- 可选：对外 ID
  `model_file_id` BIGINT UNSIGNED NULL,              -- 逻辑外键：指向 files.id（使用或产出的权重）
  `status`        VARCHAR(32) NOT NULL DEFAULT 'queued',  -- queued/running/succeeded/failed
  `started_at`    DATETIME(6) NULL,
  `finished_at`   DATETIME(6) NULL,

  `metrics`       JSON NULL,                         -- mAP、precision/recall 等
  `artifacts`     JSON NULL,                         -- 产物位置：权重相对路径、可视化图、日志等
  `params`        JSON NULL,                         -- 训练超参：epochs/imgsz/augmentations...
  `remark`        TEXT NULL,

  `created_at`    DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at`    DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
