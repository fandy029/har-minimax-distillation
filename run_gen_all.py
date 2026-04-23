#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一软标签生成脚本 - 并行启动所有9个数据集
用法:
  python3 run_gen_all.py                      # 后台运行（默认）
  python3 run_gen_all.py --sequential         # 前台运行
  python3 run_gen_all.py --ratio 0.40       # 40%采样率
  python3 run_gen_all.py --limit 500        # 每类上限500
  python3 run_gen_all.py --datasets pamap2  # 只生成指定数据集
"""
import subprocess, argparse, os, time, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED = 'scripts/gen_soft_labels_unified.py'
KUHAR  = 'scripts/gen_kuhar_parallel.py'

ALL_DATASETS = ['pamap2', 'uci_har', 'harth', 'uci_har_new',
                'motionsense', 'motionsense_dm', 'gait', 'wisdm', 'kuhar']

def get_running_procs():
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        procs = []
        for line in r.stdout.strip().split('\n'):
            if ('gen_soft_labels' in line or 'gen_kuhar_parallel' in line) and 'python' in line:
                procs.append(line)
        return procs
    except:
        return []

def is_gen_running(ds):
    try:
        r = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split('\n'):
            if ds == 'kuhar':
                if 'gen_kuhar_parallel' in line and 'python' in line:
                    return True
            else:
                if f'gen_soft_labels_unified {ds}' in line:
                    return True
    except:
        pass
    return False

def main():
    parser = argparse.ArgumentParser(description='软标签生成 - 并行启动所有数据集')
    parser.add_argument('--ratio', type=float, default=0.35,
                        help='每类采样率，默认 0.35 (35%%)')
    parser.add_argument('--limit', type=int, default=400,
                        help='每类软标签上限，默认 400')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='指定数据集，不指定则全部')
    parser.add_argument('--sequential', action='store_true',
                        help='前台运行，顺序监控（调试用）')
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS
    ratio = args.ratio
    limit = args.limit
    is_daemon = not args.sequential

    print(f'============================================================')
    print(f' 软标签生成配置')
    print(f' 每类采样率: {ratio*100:.0f}%')
    print(f' 每类上限: {limit}')
    print(f' 数据集: {", ".join(datasets)}')
    print(f' 模式: {"后台" if is_daemon else "前台"}')
    print(f'============================================================')

    running = get_running_procs()
    if running:
        print(f'\n检测到 {len(running)} 个正在运行的生成进程:')
        for p in running:
            print(f'  {p[:120]}')
        print()

    procs = []

    # Fork 到后台（在启动任何 subprocess 之前）
    if is_daemon:
        pid = os.fork()
        if pid > 0:
            # 父进程：打印PID并退出
            print(f'后台运行中，PID: {pid}', flush=True)
            os._exit(0)
        # 子进程：创建新会话，成为守护进程
        os.setsid()
        # 重定向 stdout/stderr 到日志文件
        log_path = os.path.join(SCRIPT_DIR, 'run_gen_all.log')
        log_file = open(log_path, 'a', buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())
        sys.stdin.close()
        # 在日志中重新打印配置
        print(f'============================================================')
        print(f' 软标签生成配置')
        print(f' 每类采样率: {ratio*100:.0f}%')
        print(f' 每类上限: {limit}')
        print(f' 数据集: {", ".join(datasets)}')
        print(f' 模式: 后台')
        print(f'============================================================')
        running2 = get_running_procs()
        if running2:
            print(f'\n检测到 {len(running2)} 个正在运行的生成进程:')
            for p in running2:
                print(f'  {p[:120]}')
            print()

    # 启动所有数据集（除kuhar外），每个间隔30秒
    for ds in datasets:
        if ds == 'kuhar':
            continue
        if is_gen_running(ds):
            print(f'[{ds}] 已在运行中，跳过')
            continue
        cmd = [sys.executable, UNIFIED, ds, '--ratio', str(ratio), '--limit', str(limit)]
        print(f'启动: python3 {UNIFIED} {ds} --ratio {ratio} --limit {limit}')
        p = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
        procs.append((ds, p))
        if is_daemon:
            time.sleep(30)  # 后台模式间隔30秒

    # 启动 KuHar（并行脚本）
    if 'kuhar' in datasets:
        if is_gen_running('kuhar'):
            print('[kuhar] 并行生成已在运行中，跳过')
        else:
            cmd = [sys.executable, KUHAR, 'start', '--ratio', str(ratio), '--limit', str(limit)]
            print(f'启动: python3 {KUHAR} start --ratio {ratio} --limit {limit}')
            p = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
            procs.append(('kuhar', p))
            if is_daemon:
                time.sleep(30)

    started = len(procs)
    print(f'\n共启动 {started} 个进程，监控中（每30秒检查一次）...\n')

    # 监控直到全部完成
    while procs:
        time.sleep(30)
        remaining = []
        for ds, p in procs:
            ret = p.poll()
            if ret is None:
                remaining.append((ds, p))
            else:
                print(f'[{ds}] 完成 (exit={ret})')
        procs = remaining
        if procs:
            running_ds = [ds for ds, _ in procs]
            print(f'  剩余: {", ".join(running_ds)}')
        else:
            print('\n所有数据集软标签生成完成！')
            break

if __name__ == '__main__':
    main()
