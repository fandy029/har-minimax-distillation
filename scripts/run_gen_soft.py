#!/usr/bin/env python3
"""
软标签生成参数校验 + 启动脚本
用法: python run_gen_soft.py <dataset> [--ratio RATIO] [--limit LIMIT] [--force]

参数：
  <dataset>          数据集名称（必填）
  --ratio RATIO      每类采样率，默认 0.40（40%）
  --limit LIMIT      每类软标签上限，默认 400
  --force            强制从头开始（忽略已有进度）
"""
import sys, os

THESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RATIO = 0.40
DEFAULT_LIMIT = 400

DATASETS = ['pamap2', 'kuhar', 'uci_har', 'harth', 'uci_har_new', 'motionsense', 'gait', 'wisdm', 'motionsense_dm']

DATASET_INFO = {
    'pamap2':         (5,  '3D (128,6)'),
    'kuhar':          (18, '3D (128,6)'),
    'uci_har':        (6,  '2D (561)'),
    'harth':          (6,  '3D (128,3)'),
    'uci_har_new':    (12, '2D (561)'),
    'motionsense':     (6,  '3D (128,3)'),
    'gait':           (4,  '3D (128,6)'),
    'wisdm':          (6,  '3D (128,3)'),
    'motionsense_dm':  (6,  '3D (128,6)'),
}

def main():
    if len(sys.argv) < 2:
        print(f"用法: python run_gen_soft.py <dataset> [--ratio {DEFAULT_RATIO}] [--limit {DEFAULT_LIMIT}] [--force]")
        print(f"可用数据集: {', '.join(DATASETS)}")
        print("\n参数：")
        print("  <dataset>          数据集名称（必填）")
        print(f"  --ratio {DEFAULT_RATIO}      每类采样率，默认 {DEFAULT_RATIO}（{int(DEFAULT_RATIO*100)}%）")
        print(f"  --limit {DEFAULT_LIMIT}       每类软标签上限，默认 {DEFAULT_LIMIT}")
        print("  --force           强制从头开始（忽略已有进度）")
        print("\n示例:")
        print("  python run_gen_soft.py wisdm                    # 自动计算每类软标签数")
        print("  python run_gen_soft.py pamap2 --limit 500      # 上限500")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description='软标签生成启动器', add_help=False)
    parser.add_argument('dataset', nargs='?', default=None)
    parser.add_argument('--ratio', type=float, default=DEFAULT_RATIO)
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT)
    parser.add_argument('--force', action='store_true')
    args, unknown = parser.parse_known_args()

    dataset = args.dataset

    # 校验数据集
    if dataset is None:
        print("❌ 缺少数据集参数")
        print(f"可用: {', '.join(DATASETS)}")
        sys.exit(1)
    if dataset not in DATASETS:
        print(f"❌ 未知数据集: {dataset}")
        print(f"   可用: {', '.join(DATASETS)}")
        sys.exit(1)

    n_cls, data_shape = DATASET_INFO[dataset]
    force_restart = args.force

    # 显示确认信息
    print("=" * 55)
    print(f"  软标签生成 - 参数确认")
    print("=" * 55)
    print(f"  数据集:        {dataset}")
    print(f"  类别数:        {n_cls}")
    print(f"  采样率:        {args.ratio*100:.0f}%（每类样本数 × {args.ratio}）")
    print(f"  每类上限:      {args.limit}")
    print(f"  数据形状:      {data_shape}")
    if force_restart:
        print(f"  强制重启:      是（忽略已有进度）")
    print("=" * 55)

    confirm = input("\n✅ 确认启动? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        sys.exit(0)

    # 调用生成脚本
    import subprocess
    import os as _os_  # avoid shadowing outer scope
    script = _os_.path.join(THESIS_DIR, 'scripts/gen_soft_labels_unified.py')
    cmd = [sys.executable, "-u", script, dataset,
           "--ratio", str(args.ratio),
           "--limit", str(args.limit)]
    if force_restart:
        cmd.append('--force')

    log_dir = _os_.path.join(THESIS_DIR, 'results/logs')
    log_file = f"{log_dir}/gen_{dataset}.log"

    # --force 时删除旧日志
    if force_restart:
        import os as _os
        if _os.path.exists(log_file):
            _os.remove(log_file)
        err_log = f"{log_dir}/gen_{dataset}_errors.log"
        if _os.path.exists(err_log):
            _os.remove(err_log)

    print(f"\n🚀 启动: {' '.join(cmd)}")
    print(f"   日志: {log_file}")

    # 后台运行，输出写入 results/logs/
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"   进程PID: {proc.pid}")
    print(f"   查看日志: tail -f {log_file}")


if __name__ == "__main__":
    main()
