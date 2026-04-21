#!/usr/bin/env python3
"""
自动训练链条管理器
检查当前训练状态，自动启动下一个训练任务
"""
import subprocess, time, os, sys

BASE = "/home/fandy/workplace/thesis"
os.chdir(BASE)

# 训练链条顺序
CHAIN = [
    # (dataset, version)
    ("gait", "pure_cnn"),
    ("gait", "v1"),
    ("gait", "v2"),
    ("gait", "v3"),
    ("motionsense_dm", "pure_cnn"),
    ("motionsense_dm", "v1"),
    ("motionsense_dm", "v2"),
    ("motionsense_dm", "v3"),
    ("uci_har_new", "pure_cnn"),
    ("uci_har_new", "v1"),
    ("uci_har_new", "v2"),
    ("uci_har_new", "v3"),
    ("kuhar", "pure_cnn"),
    ("kuhar", "v1"),
    ("kuhar", "v2"),
    ("kuhar", "v3"),
    ("wisdm", "pure_cnn"),
    ("wisdm", "v1"),
    ("wisdm", "v2"),
    ("wisdm", "v3"),
]

def get_running_pid():
    """检查是否有训练进程在运行"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_distill.py"],
            capture_output=True, text=True
        )
        pids = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return pids[0] if pids else None
    except:
        return None

def is_checkpoint_done(dataset, version):
    """检查某个训练任务是否已完成"""
    ckpt = f"{BASE}/results/checkpoints/{dataset}_{version}_best.pt"
    return os.path.exists(ckpt)

def start_training(dataset, version):
    """启动一个新的训练任务"""
    log_file = f"/tmp/train_{dataset}_{version}.log"
    cmd = [
        "nohup", "python3", "-u", "run_distill.py",
        dataset, version
    ]
    with open(log_file, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=BASE)
    print(f"🚀 启动训练: {dataset} {version} (log: {log_file})")
    return True

def get_last_checkpoint():
    """获取已完成链表中最后一个"""
    completed = [(ds, v) for ds, v in CHAIN if is_checkpoint_done(ds, v)]
    return completed[-1] if completed else None

def get_next_task():
    """获取下一个待训练任务"""
    for ds, v in CHAIN:
        if not is_checkpoint_done(ds, v):
            return ds, v
    return None

if __name__ == "__main__":
    running_pid = get_running_pid()
    
    if running_pid:
        print(f"✅ 训练进行中 (PID {running_pid})，无需操作")
        sys.exit(0)
    
    next_task = get_next_task()
    if next_task is None:
        print("✅ 所有训练任务已完成！")
        sys.exit(0)
    
    ds, v = next_task
    started = start_training(ds, v)
    if started:
        print(f"✅ 已启动: {ds} {v}")
    else:
        print(f"❌ 启动失败: {ds} {v}")
        sys.exit(1)
