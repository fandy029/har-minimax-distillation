#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本 - 训练所有数据集的所有版本
用法:
  python3 run_train_all.py              # 训练所有数据集（跳过正在生成软标签的）
  python3 run_train_all.py --datasets pamap2 uci_har  # 只训练指定数据集
  python3 run_train_all.py --check-only # 只检查状态，不训练
  python3 run_train_all.py --version v2  # 只训练v2版本（调试用）
  python3 run_train_all.py --resume     # 继续之前中断的训练
"""
import subprocess, argparse, os, time, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # thesis/ root
THESIS_DIR = os.path.dirname(SCRIPT_DIR)  # parent of scripts/ = thesis/
DISTILL_SCRIPT = 'run_distill.py'

ALL_DATASETS = ['pamap2', 'uci_har', 'harth', 'uci_har_new',
                'motionsense', 'motionsense_dm', 'gait', 'wisdm', 'kuhar']
VERSIONS = ['pure_cnn', 'v1', 'v2', 'v3']

def get_running_procs():
    """返回正在运行的软标签生成进程"""
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        procs = []
        for line in r.stdout.strip().split('\n'):
            if ('gen_soft_labels_unified' in line or 'gen_kuhar_parallel' in line) and 'python' in line:
                procs.append(line)
        return procs
    except:
        return []

def is_gen_running(ds):
    """检查某数据集的软标签生成是否在运行"""
    running = get_running_procs()
    for line in running:
        if ds == 'kuhar':
            if 'gen_kuhar_parallel' in line:
                return True
        else:
            if f'gen_soft_labels_unified {ds}' in line:
                return True
    return False

def get_training_status():
    """获取所有数据集的训练状态（已完成的结果）"""
    history_dir = os.path.join(THESIS_DIR, 'results', 'history')
    status = {}
    for ds in ALL_DATASETS:
        status[ds] = {}
        for version in VERSIONS:
            # 查找历史文件
            hist_file = os.path.join(history_dir, f'{ds}_{version}_history.json')
            if os.path.exists(hist_file):
                try:
                    with open(hist_file, 'r') as f:
                        data = json.load(f)
                    key = 'pure_cnn' if version == 'pure_cnn' else f'{version}_kd'
                    test_acc = data.get(key, 'N/A')
                    status[ds][version] = test_acc
                except:
                    status[ds][version] = '损坏'
            else:
                status[ds][version] = None
    return status

def main():
    parser = argparse.ArgumentParser(description='训练所有数据集的所有版本')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='指定数据集，不指定则全部')
    parser.add_argument('--version', default=None,
                        help='只训练指定版本（调试用）')
    parser.add_argument('--check-only', action='store_true',
                        help='只检查状态，不进行训练')
    parser.add_argument('--resume', action='store_true',
                        help='继续之前中断的训练（跳过已完成）')
    parser.add_argument('--force', action='store_true',
                        help='强制重新训练（忽略已有结果）')
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS
    versions = [args.version] if args.version else VERSIONS

    print(f'=' * 60)
    print(f' 训练配置')
    print(f' 数据集: {", ".join(datasets)}')
    print(f' 版本: {", ".join(versions)}')
    print(f' 模式: {"检查状态" if args.check_only else "训练"}')
    if args.resume:
        print(f' 恢复训练: 是（跳过已完成）')
    if args.force:
        print(f' 强制重新训练: 是（忽略已有结果）')
    print(f'=' * 60)

    # 状态检查
    running = get_running_procs()
    if running:
        print(f'\n检测到 {len(running)} 个软标签生成进程正在运行:')
        for p in running:
            print(f'  {p[:100]}')
        print()
    else:
        print('\n无运行中的软标签生成进程')

    status = get_training_status()

    print('\n当前训练状态:')
    print(f'{"数据集":<18} {"pure_cnn":>10} {"v1":>10} {"v2":>10} {"v3":>10} {"生成中":>6}')
    print('-' * 66)
    for ds in datasets:
        gen_running = is_gen_running(ds)
        gen_marker = '🔄' if gen_running else ''
        pure = status[ds].get('pure_cnn', 'N/A')
        v1 = status[ds].get('v1', 'N/A')
        v2 = status[ds].get('v2', 'N/A')
        v3 = status[ds].get('v3', 'N/A')
        pure_s = f'{pure:.2f}%' if isinstance(pure, float) else str(pure)
        v1_s = f'{v1:.2f}%' if isinstance(v1, float) else str(v1)
        v2_s = f'{v2:.2f}%' if isinstance(v2, float) else str(v2)
        v3_s = f'{v3:.2f}%' if isinstance(v3, float) else str(v3)
        print(f'{ds:<18} {pure_s:>10} {v1_s:>10} {v2_s:>10} {v3_s:>10} {gen_marker:>6}')

    if args.check_only:
        print('\n--check-only 模式，仅检查状态')
        return

    print('\n' + '=' * 66)
    print(' 开始训练...')
    print('=' * 66)

    trained = 0
    skipped_gen = 0
    skipped_done = 0

    for ds in datasets:
        gen_running = is_gen_running(ds)
        if gen_running:
            print(f'\n[{ds}] 🔄 软标签生成中，跳过')
            skipped_gen += 1
            continue

        for version in versions:
            # 检查是否已完成
            pure_status = status[ds].get('pure_cnn')
            # history文件key是v1_kd/v2_kd/v3_kd，需要映射
            status_key = 'pure_cnn' if version == 'pure_cnn' else f'{version}_kd'
            v_status = status[ds].get(version)
            history_file = os.path.join(THESIS_DIR, 'results', 'history', f'{ds}_{version}_history.json')

            if not args.force and os.path.exists(history_file):
                print(f'\n[{ds}][{version}] ✅ 已完成，跳过 (结果: {v_status})')
                skipped_done += 1
                continue

            if args.resume and v_status is not None and v_status != 'N/A':
                print(f'\n[{ds}][{version}] ✅ 已完成，跳过 (结果: {v_status})')
                skipped_done += 1
                continue

            print(f'\n[{ds}][{version}] 🚀 开始训练...')
            cmd = [sys.executable, DISTILL_SCRIPT, ds, version]
            print(f'  命令: {" ".join(cmd)}')

            result = subprocess.run(cmd, cwd=SCRIPT_DIR)
            trained += 1

            if result.returncode == 0:
                print(f'[{ds}][{version}] ✅ 训练完成')
            else:
                print(f'[{ds}][{version}] ❌ 训练失败 (exit={result.returncode})')

            # 更新状态
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        data = json.load(f)
                    key = 'pure_cnn' if version == 'pure_cnn' else f'{version}_kd'
                    test_acc = data.get(key, 'N/A')
                    print(f'  测试准确率: {test_acc}')
                except:
                    pass

    print('\n' + '=' * 66)
    print(f' 训练完成！')
    print(f' 本轮训练: {trained} 个')
    print(f' 跳过（生成中）: {skipped_gen} 个')
    print(f' 跳过（已完成）: {skipped_done} 个')
    print('=' * 66)

if __name__ == '__main__':
    main()
