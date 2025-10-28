import os, re, glob
from typing import List, Tuple, Optional, Dict, Iterable, Literal
import numpy as np
import matplotlib.pyplot as plt

# 文件名解析：<step>_sl_<sparsity>.png
DEFAULT_NAME_RE = re.compile(
    r"(?P<step>\d+)_sl_(?P<sl>\d+(?:\.\d+)?).*?\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)

def parse_dir(
    img_dir: str,
    name_re: re.Pattern = DEFAULT_NAME_RE,
    dedup: Literal["last", "mean", "max"] = "last",
) -> List[Tuple[int, float]]:
    """从单个目录解析 (step, sparsity) 序列"""
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    files.sort()

    bucket: Dict[int, List[float]] = {}
    for fp in files:
        m = name_re.search(os.path.basename(fp))
        if not m:
            continue
        step = int(m.group("step"))
        sl = float(m.group("sl"))
        bucket.setdefault(step, []).append(sl)

    steps = sorted(bucket.keys())
    seq = []
    for s in steps:
        vals = bucket[s]
        if dedup == "last":
            seq.append((s, vals[-1]))
        elif dedup == "mean":
            seq.append((s, float(np.mean(vals))))
        elif dedup == "max":
            seq.append((s, float(np.max(vals))))
        else:
            raise ValueError("Unknown dedup mode")
    return seq

def ema_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def movavg(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) == 0:
        return y.astype(float)
    k = min(k, len(y))
    csum = np.cumsum(np.r_[0.0, y])
    out = (csum[k:] - csum[:-k]) / k
    pad = y[:k-1].astype(float)
    return np.r_[pad, out]

def plot_sparsity_from_dirs(
    dirs: Iterable[str],
    labels: List[str],
    *,
    name_re: re.Pattern = DEFAULT_NAME_RE,
    dedup: Literal["last", "mean", "max"] = "last",
    smooth: Optional[Tuple[str, float]] = None,
    figsize=(10, 4),
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    从多个目录解析并绘制稀疏率折线图。

    参数:
      dirs    : 目录列表
      labels  : 与 dirs 对应的标签列表（必须等长）
      name_re : 文件名解析正则
      dedup   : 同步去重策略: 'last' | 'mean' | 'max'
      smooth  : 可选平滑 ("ema", alpha) 或 ("ma", k)
      figsize : 图尺寸
      ylim    : y轴范围
      save_path: 保存路径（如 'sparsity.png'）
      show    : 是否 plt.show()
    """
    dirs = list(dirs)
    if len(labels) != len(dirs):
        raise ValueError("labels 数量必须和 dirs 数量一致")

    series = []
    for d, lb in zip(dirs, labels):
        pairs = parse_dir(d, name_re=name_re, dedup=dedup)
        if not pairs:
            continue
        steps, y = map(np.asarray, zip(*pairs))
        if smooth is not None:
            mode, param = smooth
            if mode == "ema":
                y = ema_smooth(y, float(param))
            elif mode == "ma":
                y = movavg(y, int(param))
            else:
                raise ValueError("smooth 仅支持 ('ema', alpha) 或 ('ma', k)")
        series.append({"step": steps, "y": y.astype(float), "label": lb})

    if not series:
        raise RuntimeError("没有解析到任何 (step, sparsity) 数据")

    plt.figure(figsize=figsize)
    for s in series:
        plt.plot(s["step"], s["y"], label=s["label"], linewidth=1.6)
    plt.xlabel("step")
    plt.ylabel("sparsity ratio")
    plt.title("Sparsity vs Step")
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return series

if __name__ == '__main__':
    plot_sparsity_from_dirs(
    dirs=["/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/202508312129-prune-slim-convnextv2-mutilscale36912-reg0.01-gt-sp2.0-prune/visual", 
          "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/slim/202508240240-prune-slim-convnextv2-mutilscale36912-reg0.05-prune/visual",
          "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/slim/202508272254-prune-slim-convnextv2-mutilscale36912-reg0.1-gt-sp2.0-prune/visual",
          "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/202509010746-prune-slim-convnextv2-mutilscale36912-reg0.001-gt-sp2.0-prune/visual"],
    labels=["lambda=0.01",
            "lambda=0.05", 
            "lambda=0.1",
            "lambda=0.001"],
    smooth=("ema", 0.2),         # 可选：EMA 平滑
    ylim=(0.0, 0.35),
    save_path="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/sparsity_compare.png"
)

