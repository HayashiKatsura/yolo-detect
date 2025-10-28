import csv
from genericpath import isdir, isfile
import os
from datetime import timedelta

def frame_detect_csv(save_folder, target_frames):
    
    # 同一目标(同类别+连续帧）的检测段,仅保留一个中间时间写入CSV
    
    # 步骤1：按帧索引排序,确保帧顺序正确
    target_frames.sort(key=lambda x: x["frame_idx"])
    if not target_frames:
        return None

    # 步骤2：识别“同一目标(类别+连续帧）”的检测段
    target_segments = []  # 存储目标段：[{start_idx, end_idx, class, start_ms, end_ms, mid_frame}, ...]
    
    # 初始化第一个目标段(按帧内每个目标分别初始化）
    first_frame = target_frames[0]
    for det in first_frame["detections"]:
        target_segments.append({
            "class": det["class"],  # 目标类别(核心：按类别区分同一目标）
            "start_idx": first_frame["frame_idx"],
            "end_idx": first_frame["frame_idx"],
            "start_ms": first_frame["timestamp_ms"],
            "end_ms": first_frame["timestamp_ms"],
            "frames": [first_frame]  # 存储该目标段的所有帧
        })

    # 遍历后续帧,扩展或新增目标段
    for frame in target_frames[1:]:
        current_frame_idx = frame["frame_idx"]
        current_frame_dets = {det["class"]: det for det in frame["detections"]}  # 按类别存储当前帧目标

        # 1. 处理已有目标段：判断当前帧是否有同类别目标且帧连续
        updated_segments = []
        for seg in target_segments:
            seg_class = seg["class"]
            # 判定：当前帧有同类别目标 + 帧连续(当前帧索引=段结束帧+1）
            if seg_class in current_frame_dets and current_frame_idx == seg["end_idx"] + 1:
                # 扩展目标段：更新结束帧、结束时间、添加当前帧
                seg["end_idx"] = current_frame_idx
                seg["end_ms"] = frame["timestamp_ms"]
                seg["frames"].append(frame)
                updated_segments.append(seg)
                # 从当前帧目标中移除已匹配的类别(避免重复处理）
                del current_frame_dets[seg_class]
            else:
                # 目标段不连续或无同类别目标,保留原段
                updated_segments.append(seg)
        target_segments = updated_segments

        # 2. 处理当前帧中未匹配的新目标(新增目标段）
        for det_class, det in current_frame_dets.items():
            target_segments.append({
                "class": det_class,
                "start_idx": current_frame_idx,
                "end_idx": current_frame_idx,
                "start_ms": frame["timestamp_ms"],
                "end_ms": frame["timestamp_ms"],
                "frames": [frame]
            })

    # 步骤3：过滤持续时间 < 1秒(1000毫秒）的目标段
    filtered_segments = []
    for seg in target_segments:
        seg_duration_ms = seg["end_ms"] - seg["start_ms"]
        if seg_duration_ms >= 1000:  # 仅保留持续≥1秒的目标段
            filtered_segments.append(seg)
    if not filtered_segments:
        # print(f"任务 {save_folder}:无持续≥1秒的目标段,不生成CSV")
        return None

    # 步骤4：时间戳格式转换(毫秒→分:秒.毫秒）
    def format_time(ms):
        td = timedelta(milliseconds=ms)
        minutes = int(td.total_seconds() // 60)
        seconds = int(td.total_seconds() % 60)
        ms_remain = td.microseconds // 1000
        return f"{minutes:02d}:{seconds:02d}.{ms_remain:03d}"

    # 步骤5：生成CSV(同一目标段仅写入一个中间时间）
    # csv_filename = f"{save_folder}_target_timeline.csv" 
    
    if os.path.isfile(save_folder):
        save_folder = f'{os.path.splitext(save_folder)[0]}_detected.csv'
    elif os.path.isdir(save_folder):
        save_folder = f'{os.path.join(save_folder, "detected.csv")}'
    # csv_path = os.path.join(output_folder, save_folder)

    with open(save_folder, mode="w", newline="", encoding="utf-8") as f:
        # fieldnames = [
        #     "目标段ID", "目标类别", "中间帧时间", 
        #     "中间帧索引", "目标置信度", "持续时间(秒)"
        # ]
        fieldnames = [
            "seg_id", "class", "mid_frame_time", 
            "mid_frame_idx", "confidence", "duration_sec"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历过滤后的目标段,每个段仅写入一条中间时间记录
        for seg_id, seg in enumerate(filtered_segments, 1):
            # 计算当前目标段的中间帧
            mid_idx = (seg["start_idx"] + seg["end_idx"]) // 2
            # 找到中间帧数据(取中间帧中该类别的目标置信度）
            mid_frame = next(f for f in seg["frames"] if f["frame_idx"] == mid_idx)
            mid_det = next(d for d in mid_frame["detections"] if d["class"] == seg["class"])
            # 计算中间时间和持续时间
            mid_time = format_time(mid_frame["timestamp_ms"])
            duration_sec = round((seg["end_ms"] - seg["start_ms"]) / 1000, 2)

            # 同一目标段仅写入一条记录(含唯一中间时间）
            writer.writerow({
                "seg_id": seg_id,
                "class": seg["class"],
                "mid_frame_time": mid_time,  # 同一目标段仅一个中间时间
                "mid_frame_idx": mid_idx,
                "confidence": round(mid_det["confidence"], 2),
                "duration_sec": duration_sec
            })

    # print(f"任务 {save_folder}:CSV生成完成,共{len(filtered_segments)}个目标段(每段一个中间时间）,路径：{csv_path}")

    return str(save_folder)