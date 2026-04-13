#!/usr/bin/env python3
"""
监控脚本 - 每10分钟汇报实验进展
每10分钟运行一次，检查results目录状态并发送汇报
"""
import os, json, time, glob
from pathlib import Path

RESULTS_DIR = "/home/fandy/workplace/thesis/results"
STATUS_FILE = "/home/fandy/workplace/thesis/.monitor_status.json"

# 数据集定义
DATASETS = {
    "pamap2": {"best": "v3", "initial": "v2"},
    "kuhar": {"best": "v2", "initial": "v1"},
    "uci_har": {"best": "v2", "initial": "v1"},
    "harth": {"best": "v2", "initial": "v1"},
    "gait": {"best": "v2", "initial": "v1"},
    "uci_har_new": {"best": "v2", "initial": "v1"},
    "motionsense": {"best": "v2", "initial": "v1"},
    "wisdm": {"best": "pure_cnn", "initial": None},
    "motionsense_dm": {"best": "pure_cnn", "initial": None},
}

# 预期结果（用于比较）
EXPECTED = {
    "pamap2": {"v3": 95.93, "v2": 95.02, "v1": 93.10},
    "kuhar": {"v2": 85.02, "v1": 81.01},
    "uci_har": {"v2": 96.50, "v1": 96.20},
    "harth": {"v2": 96.10},
    "gait": {"v2": 98.37},
    "uci_har_new": {"v2": 93.99},
    "motionsense": {"v2": 99.40},
    "wisdm": {"pure_cnn": 99.60},
    "motionsense_dm": {"pure_cnn": 99.58},
}

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"started_at": time.time(), "completed": [], "running": None, "last_update": None}

def save_status(s):
    with open(STATUS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def check_running():
    """检查正在运行的实验"""
    import subprocess
    result = subprocess.run(["pgrep", "-f", "distill_"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout:
        pids = result.stdout.decode().strip().split()
        return len(pids)
    return 0

def get_result_file(dataset, version):
    pattern = f"{RESULTS_DIR}/{dataset}_v{version}*.json"
    files = glob.glob(pattern)
    # 也检查旧路径
    if not files:
        pattern2 = f"/home/fandy/workplace/thesis/new_results_v2/{dataset}*.json"
        files = glob.glob(pattern2)
    if not files:
        pattern3 = f"/home/fandy/workplace/thesis/new_results/{dataset}*.json"
        files = glob.glob(pattern3)
    return files[0] if files else None

def read_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def check_log_file(dataset, version):
    """检查日志文件获取进度"""
    patterns = [
        f"{RESULTS_DIR}/{dataset}_v{version}_log.txt",
        f"/home/fandy/workplace/thesis/new_results_v2/{dataset}*log.txt",
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            try:
                with open(files[0]) as f:
                    lines = f.readlines()
                    return lines[-1].strip() if lines else ""
            except:
                pass
    return None

def format_report(status):
    lines = ["📊 **HAR蒸馏实验进展汇报**\n"]
    
    elapsed = time.time() - status.get("started_at", time.time())
    mins = int(elapsed / 60)
    lines.append(f"⏱️ 已运行: {mins}分钟\n")
    
    # 检查正在运行的
    running_count = check_running()
    if running_count > 0:
        lines.append(f"🔄 正在运行: {running_count}个进程\n")
    
    lines.append("─" * 30)
    lines.append("\n**数据集状态:**\n")
    
    for ds, versions in DATASETS.items():
        best_v = versions.get("best")
        initial_v = versions.get("initial")
        
        best_done = best_v and get_result_file(ds, best_v)
        initial_done = initial_v and get_result_file(ds, initial_v)
        
        if best_done and (not initial_v or initial_done):
            icon = "✅"
        elif best_done:
            icon = "⏳"
        else:
            icon = "⏸️"
        
        best_str = f"**{best_v.upper()}**={best_done['v3_kd'] if 'v3_kd' in best_done else best_done.get('v2_kd', best_done.get('pure_cnn', '?'))}%🔥" if best_done else f"{best_v.upper()}❌"
        
        if initial_v and initial_v != "v1":
            initial_str = f"{initial_v.upper()}={'✅' if initial_done else '⏳'}"
        elif initial_v:
            initial_str = f"v1={'✅' if initial_done else '⏳'}"
        else:
            initial_str = ""
        
        lines.append(f"{icon} {ds.upper()}: {best_str} {initial_str}")
    
    lines.append("\n─" * 30)
    
    completed = len(status.get("completed", []))
    total = sum(1 for v in DATASETS.values() if v.get("initial") or v.get("best"))
    lines.append(f"\n完成: {completed}/{total} 个版本")
    
    return "\n".join(lines)

if __name__ == "__main__":
    status = load_status()
    report = format_report(status)
    print(report)
    print(f"\n--- RAW STATUS ---")
    print(json.dumps(status, indent=2))
