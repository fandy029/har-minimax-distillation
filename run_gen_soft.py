#!/usr/bin/env python3
"""
软标签生成参数校验 + 启动脚本
用法: python run_gen_soft.py <dataset> [samples_per_class]

新逻辑：每类软标签数量 = 该类训练样本数 × 25%，上限200
手动指定 samples_per_class 会覆盖动态计算（慎用）。
"""
import sys

DATASETS = ['pamap2', 'kuhar', 'uci_har', 'harth', 'uci_har_new', 'motionsense', 'gait', 'wisdm', 'motionsense_dm']

DATASET_INFO = {
    'pamap2':         (5,  '3D (128,6)'),
    'kuhar':          (18, '3D (128,6)'),
    'uci_har':        (6,  '2D (561)'),
    'harth':          (6,  '3D (128,3)'),
    'uci_har_new':    (12, '2D (561)'),
    'motionsense':    (6,  '3D (128,3)'),
    'gait':           (4,  '3D (128,6)'),
    'wisdm':          (6,  '3D (128,3)'),   # 改用 raw 时间序列
    'motionsense_dm': (6,  '3D (128,6)'),
}

def main():
    if len(sys.argv) < 2:
        print("用法: python run_gen_soft.py <dataset> [samples_per_class]")
        print(f"可用数据集: {', '.join(DATASETS)}")
        print("\n说明: 软标签数量 = 每类样本数 × 25%，上限200")
        print("      手动指定 samples_per_class 会覆盖动态计算（慎用）")
        print("\n示例:")
        print("  python run_gen_soft.py wisdm          # 自动计算每类软标签数")
        print("  python run_gen_soft.py wisdm 200      # 强制每类200（覆盖默认逻辑）")
        print("  python run_gen_soft.py wisdm --force   # 强制从头开始")
        sys.exit(1)

    dataset = sys.argv[1]

    # 校验数据集名
    if dataset not in DATASETS:
        print(f"❌ 未知数据集: {dataset}")
        print(f"   可用: {', '.join(DATASETS)}")
        sys.exit(1)

    n_cls, data_shape = DATASET_INFO[dataset]
    manual_spc = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '--force' else None
    force_restart = '--force' in sys.argv

    if manual_spc is not None:
        # 手动指定每类样本数（覆盖动态计算）
        if manual_spc < 1 or manual_spc > 500:
            print(f"❌ samples_per_class 超出范围: {manual_spc} (有效: 1-500)")
            sys.exit(1)
        mode_desc = f"手动指定每类 {manual_spc}（覆盖动态计算）"
    else:
        mode_desc = "动态计算: 每类样本数×25%，上限200"

    # 显示确认信息
    print("=" * 55)
    print(f"  软标签生成 - 参数确认")
    print("=" * 55)
    print(f"  数据集:        {dataset}")
    print(f"  类别数:        {n_cls}")
    print(f"  计算模式:      {mode_desc}")
    print(f"  数据形状:      {data_shape}")
    if force_restart:
        print(f"  强制重启:      是（忽略已有进度）")
    print("=" * 55)
    print(f"\n  脚本将自动加载数据集并计算各class样本数。")
    print(f"  实际每类软标签数取决于数据集实际情况。")

    confirm = input("\n✅ 确认启动? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        sys.exit(0)

    # 调用生成脚本（后台运行）
    import subprocess, os
    script = "/home/fandy/workplace/thesis/gen_soft_labels_unified.py"
    cmd = [sys.executable, "-u", script, dataset]
    if manual_spc is not None:
        cmd.append(str(manual_spc))
    if force_restart:
        cmd.append('--force')
    
    log_file = f"/tmp/gen_{dataset}.log"
    
    # --force 时删除旧日志
    if force_restart:
        import os as _os
        if _os.path.exists(log_file):
            _os.remove(log_file)
        err_log = f"/home/fandy/workplace/thesis/results/logs/gen_{dataset}_errors.log"
        if _os.path.exists(err_log):
            _os.remove(err_log)
        fresh_log = f"/home/fandy/workplace/thesis/results/logs/gen_{dataset}_fresh.log"
        if _os.path.exists(fresh_log):
            _os.remove(fresh_log)
        other_log = f"/home/fandy/workplace/thesis/results/logs/gen_{dataset}.log"
        if _os.path.exists(other_log):
            _os.remove(other_log)
    
    print(f"\n🚀 启动: {' '.join(cmd)}")
    print(f"   日志: {log_file}")
    
    # 后台运行，输出写入日志文件
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"   进程PID: {proc.pid}")
    print(f"   查看日志: tail -f {log_file}")


if __name__ == "__main__":
    main()
