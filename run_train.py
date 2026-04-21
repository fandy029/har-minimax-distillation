#!/usr/bin/env python3
"""
训练参数校验 + 启动脚本
用法: python run_train.py <dataset> <version> [--resume]
"""
import sys

DATASETS = ['pamap2', 'kuhar', 'uci_har', 'harth', 'uci_har_new', 'motionsense', 'gait', 'wisdm', 'motionsense_dm']
VERSIONS = ['pure_cnn', 'v1', 'v2', 'v3']

DATASET_INFO = {
    'pamap2':         (5,  '3D CNN',  '6通道 128步'),
    'kuhar':          (18, '3D CNN',  '8通道 128步'),
    'uci_har':        (6,  '2D MLP',  '561维特征'),
    'harth':          (6,  '3D CNN',  '3通道 128步'),
    'uci_har_new':    (12, '2D MLP',  '561维特征'),
    'motionsense':    (6,  '3D CNN',  '3通道 128步'),
    'gait':           (4,  '3D CNN',  '6通道 128步'),
    'wisdm':          (6,  '3D CNN',  '3通道 128步'),  # 改用 raw 时间序列 CNN
    'motionsense_dm': (6, '3D CNN',  '3通道 128步'),
}

VERSION_PARAMS = {
    'v1': {'T': 3.0,  'ALPHA': 0.6, 'epochs': 300},
    'v2': {'T': 2.5,  'ALPHA': 0.8, 'epochs': 300},
    'v3': {'T': 1.5,  'ALPHA': 0.85,'epochs': 300},
}

def main():
    if len(sys.argv) < 3:
        print("用法: python run_train.py <dataset> <version> [--resume]")
        print(f"数据集: {', '.join(DATASETS)}")
        print(f"版本:   {', '.join(VERSIONS)}")
        print("\n示例:")
        print("  python run_train.py pamap2 v2")
        print("  python run_train.py wisdm v3")
        print("  python run_train.py uci_har_new v1 --resume")
        print("  python run_train.py kuhar pure_cnn --resume")
        sys.exit(1)

    dataset = sys.argv[1]
    version = sys.argv[2]
    resume = '--resume' in sys.argv

    # 校验数据集
    if dataset not in DATASETS:
        print(f"❌ 未知数据集: {dataset}")
        print(f"   可用: {', '.join(DATASETS)}")
        sys.exit(1)

    # 校验版本
    if version not in VERSIONS:
        print(f"❌ 未知版本: {version}")
        print(f"   可用: {', '.join(VERSIONS)}")
        sys.exit(1)

    n_cls, model_type, data_desc = DATASET_INFO[dataset]
    params = VERSION_PARAMS.get(version, {})

    # 检查软标签文件
    import os
    soft_file = f"/home/fandy/workplace/thesis/results/soft_labels/{dataset}_soft.npy"
    checkpoint_file = f"/home/fandy/workplace/thesis/results/checkpoints/{dataset}_{version}_best.pt"
    soft_exists = os.path.exists(soft_file)
    ckpt_exists = os.path.exists(checkpoint_file)
    soft_info = ""
    if version == 'pure_cnn':
        soft_info = "(纯训练模式，不需要软标签)"
    elif soft_exists:
        import numpy as np
        arr = np.load(soft_file)
        is_onehot = (arr > 0.99).sum(axis=1) == 1
        real = int(np.sum((arr.sum(axis=1) > 0) & ~is_onehot))
        soft_info = f"✅ 存在 ({real} 真软标签)"
    else:
        soft_info = "⚠️ 软标签不存在!"

    # 显示确认信息
    print("=" * 60)
    print(f"  训练任务 - 参数确认")
    print("=" * 60)
    print(f"  数据集:      {dataset}")
    print(f"  版本:        {version}")
    print(f"  类别数:      {n_cls}")
    print(f"  模型类型:    {model_type}")
    print(f"  数据描述:    {data_desc}")
    print(f"  断点续传:    {'是 (--resume)' if resume else '否'}")
    print("-" * 60)
    if version != 'pure_cnn':
        print(f"  软标签:      {soft_info}")
        print(f"  Temperature: {params['T']}")
        print(f"  Alpha:       {params['ALPHA']}")
        print(f"  Epochs:      {params['epochs']}")
    else:
        print(f"  软标签:      (pure_cnn 模式)")
    if ckpt_exists:
        print(f"  断点文件:    ✅ 存在")
    else:
        print(f"  断点文件:    ⏸️ 不存在 (将从头开始)")
    print("=" * 60)

    # 版本兼容性检查
    issues = []
    if version != 'pure_cnn' and not soft_exists:
        issues.append("软标签文件不存在，训练可能失败")

    if issues:
        print("\n⚠️  问题:")
        for issue in issues:
            print(f"  - {issue}")
        confirm = input("\n仍要继续? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("已取消")
            sys.exit(1)

    confirm = input("\n✅ 确认启动? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        sys.exit(0)

    # 调用训练脚本
    import subprocess
    script = "/home/fandy/workplace/thesis/run_distill.py"
    cmd = [sys.executable, "-u", script, dataset, version]
    if resume:
        cmd.append('--resume')

    log_file = f"/tmp/train_{dataset}_{version}.log"
    print(f"\n🚀 启动: {' '.join(cmd)}")
    print(f"   日志文件: {log_file}")
    print(f"   断点续传: {'启用' if resume else '禁用'}")
    print(f"   使用 'tail -f {log_file}' 查看实时日志\n")

    # 日志写入文件，同时输出到终端
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"   进程PID: {proc.pid}")
    print(f"   查看日志: tail -f {log_file}")

if __name__ == "__main__":
    main()
