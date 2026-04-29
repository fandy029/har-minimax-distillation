#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软标签生成 - 批量启动所有9个数据集
用法:
  python3 run_gen_all.py                      # 后台运行（默认）
  python3 run_gen_all.py --ratio 0.40       # 采样率
  python3 run_gen_all.py --limit 500        # 每类上限500
  python3 run_gen_all.py --force             # 强制从头开始
  python3 run_gen_all.py --kill              # 杀掉所有正在运行的生成进程
  python3 run_gen_all.py --status           # 查看正在运行的进程状态
"""
import subprocess, argparse, os, time, sys, json, signal

DEFAULT_RATIO = 0.40
DEFAULT_LIMIT = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # thesis/ root (where script is run from)
THESIS_DIR = os.path.dirname(SCRIPT_DIR)  # parent of scripts/ = thesis/
UNIFIED = 'gen_soft_labels_unified.py'
KUHAR  = 'gen_kuhar_parallel.py'
PID_FILE = 'results/.run_gen_all_pids.json'

ALL_DATASETS = ['pamap2', 'uci_har', 'harth', 'uci_har_new',
                'motionsense', 'motionsense_dm', 'gait', 'wisdm', 'kuhar']

def load_pid_file():
    pf = os.path.join(SCRIPT_DIR, PID_FILE)
    if os.path.exists(pf):
        with open(pf) as f:
            return json.load(f)
    return {"launch_time": None, "processes": {}}

def save_pid_file(data):
    pf = os.path.join(SCRIPT_DIR, PID_FILE)
    with open(pf, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def remove_pid_file():
    pf = os.path.join(SCRIPT_DIR, PID_FILE)
    if os.path.exists(pf):
        os.remove(pf)

def is_pid_alive(pid):
    """检查进程是否存在"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_running_procs():
    """获取所有正在运行的生成进程（排除zombie）"""
    import psutil
    procs = []
    for p in psutil.process_iter(['pid', 'cmdline', 'status']):
        try:
            cmdline = ' '.join(p.info['cmdline'] or [])
            status = p.info['status']
            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                continue
            if ('gen_soft_labels' in cmdline or 'gen_kuhar_parallel' in cmdline) and 'python' in cmdline:
                procs.append(cmdline[:120])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs

def is_gen_running(ds):
    """检查指定数据集的进程是否在运行（使用psutil，排除zombie）"""
    import psutil
    for p in psutil.process_iter(['pid', 'cmdline', 'status']):
        try:
            cmdline = ' '.join(p.info['cmdline'] or [])
            status = p.info['status']
            # 排除 zombie/defunct
            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                continue
            if ds == 'kuhar':
                if 'gen_kuhar_parallel' in cmdline and 'python' in cmdline:
                    return True
            else:
                if f'gen_soft_labels_unified {ds}' in cmdline and 'python' in cmdline:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def cmd_kill():
    """杀掉所有正在运行的生成进程"""
    pid_data = load_pid_file()
    procs_info = pid_data.get("processes", {})
    killed = []
    not_found = []
    errors = []

    if not procs_info:
        print("PID 文件为空或不存在，尝试通过进程扫描杀进程...")

    # 先用 PID 文件中的记录杀
    for ds, pid in procs_info.items():
        if is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(f"{ds}(PID={pid})")
            except OSError as e:
                errors.append(f"{ds}(PID={pid}): {e}")
        else:
            not_found.append(f"{ds}(PID={pid})")

    # 如果 PID 文件没有记录，用 ps 扫描杀
    running = get_running_procs()
    for line in running:
        try:
            parts = line.split()
            pid = int(parts[1])
            # 避免重复杀
            if str(pid) not in [str(v) for v in procs_info.values()]:
                os.kill(pid, signal.SIGTERM)
                killed.append(f"扫描发现(PID={pid})")
        except (ValueError, OSError):
            pass

    remove_pid_file()

    if killed:
        print(f"✅ 已发送 SIGTERM 信号: {', '.join(killed)}")
    if not_found:
        print(f"⚠️ 进程不存在（已结束）: {', '.join(not_found)}")
    if errors:
        print(f"❌ 杀死进程失败: {', '.join(errors)}")
    if not killed:
        print("没有找到正在运行的生成进程")

def cmd_status():
    """查看正在运行的进程状态"""
    pid_data = load_pid_file()
    procs_info = pid_data.get("processes", {})
    launch_time = pid_data.get("launch_time", "未知")

    print(f"============================================================")
    print(f" 软标签生成进程状态")
    print(f" 启动时间: {launch_time}")
    print(f"============================================================")

    if not procs_info:
        print("  (PID 文件为空或不存在)")

    alive = []
    dead = []
    for ds, pid in procs_info.items():
        if is_pid_alive(pid):
            alive.append(f"{ds}(PID={pid})")
        else:
            dead.append(f"{ds}(PID={pid})")

    if alive:
        print(f"\n  🟢 运行中 ({len(alive)}):")
        for p in alive:
            print(f"    {p}")
    else:
        print(f"\n  没有运行中的进程")

    if dead:
        print(f"\n  ⚪ 已结束 ({len(dead)}):")
        for p in dead:
            print(f"    {p}")

    # 也扫描一下有没有漏网的
    running = get_running_procs()
    if running:
        print(f"\n  🔍 扫描到其他生成进程:")
        for line in running:
            parts = line.split()
            pid = parts[1]
            cmdline = ' '.join(parts[10:13])
            print(f"    PID={pid}: {cmdline}")

    print(f"\n  总计运行中: {len(alive)}")
    print(f"============================================================")

    # 如果全部死了，删除 PID 文件
    if not alive and procs_info:
        remove_pid_file()
        print("  PID 文件已清理（所有进程已结束）")

def main():
    parser = argparse.ArgumentParser(description='软标签生成 - 批量启动所有数据集')
    parser.add_argument('--ratio', type=float, default=DEFAULT_RATIO,
                        help=f'每类采样率，默认 {DEFAULT_RATIO} ({int(DEFAULT_RATIO*100)}%%)')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'每类软标签上限，默认 {DEFAULT_LIMIT}')
    parser.add_argument('--force', action='store_true',
                        help='强制从头开始（删除已有软标签和旧日志）')
    parser.add_argument('--kill', action='store_true',
                        help='杀掉所有正在运行的生成进程')
    parser.add_argument('--status', action='store_true',
                        help='查看正在运行的进程状态')
    args = parser.parse_args()

    # --kill 和 --status 是独立的，不和其他参数混用
    if args.kill:
        cmd_kill()
        return
    if args.status:
        cmd_status()
        return

    datasets = ALL_DATASETS
    ratio = args.ratio
    limit = args.limit

    print(f'============================================================')
    print(f' 软标签生成配置')
    print(f' 每类采样率: {ratio*100:.0f}%')
    print(f' 每类上限: {limit}')
    print(f' 数据集: {", ".join(datasets)}')
    if args.force:
        print(f' 强制重生成: 是（删除已有软标签和旧日志）')
    print(f'============================================================')

    # --force: 删除已有软标签和旧日志
    if args.force:
        for ds in datasets:
            soft_file = f"{THESIS_DIR}/results/soft_labels/{ds}_soft.npy"
            if os.path.exists(soft_file):
                os.remove(soft_file)
                print(f'  🗑️ 删除已有软标签: {os.path.basename(soft_file)}')
            err_log = f"{THESIS_DIR}/results/logs/gen_{ds}_errors.log"
            if os.path.exists(err_log):
                os.remove(err_log)
            stdout_log = f"{THESIS_DIR}/results/logs/gen_{ds}.log"
            if os.path.exists(stdout_log):
                os.remove(stdout_log)
        for pid in range(6):
            for ext in ('', '_err'):
                f = f"{THESIS_DIR}/results/logs/gen_kuhar_{pid}{ext}.log"
                if os.path.exists(f):
                    os.remove(f)
                    print(f'  🗑️ 删除: {os.path.basename(f)}')
        print()

    running = get_running_procs()
    if running:
        print(f'\n检测到 {len(running)} 个正在运行的生成进程:')
        for p in running:
            print(f'  {p[:120]}')
        print()

    procs = {}  # {dataset: pid}

    # Fork 到后台
    # Fork 两次：避免僵尸中间进程
    LOG_FILE = os.path.join(THESIS_DIR, 'results', 'logs', 'daemon_gen.log')
    logf = open(LOG_FILE, 'a')
    
    # 第一次 fork
    pid1 = os.fork()
    if pid1 > 0:
        # 父进程：等中间进程退出后报告 PID
        for _ in range(50):
            result = os.waitpid(pid1, os.WNOHANG)
            if result[0] != 0:
                break
            time.sleep(0.1)
        print(f'后台运行中，PID: {pid1}', flush=True)
        os._exit(0)
    
    # 中间进程：创建新会话，成为无终端守护进程
    os.setsid()
    
    # 第二次 fork（防止获取控制终端）
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    
    # 孙子进程：真正的 daemon，重定向 stdout/stderr
    os.dup2(logf.fileno(), sys.stdout.fileno())
    os.dup2(logf.fileno(), sys.stderr.fileno())
    os.close(logf.fileno())
    sys.stdin.close()
    
    import traceback
    try:
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
            procs[ds] = p  # 存 Popen 对象，方便 wait()
            time.sleep(30)
            if p.poll() is not None:
                try:
                    p.wait()
                except Exception as e:
                    print(f'[{ds}] wait() 失败: {e}')

        # 启动 KuHar（并行脚本）
        if is_gen_running('kuhar'):
            print('[kuhar] 并行生成已在运行中，跳过')
        else:
            cmd = [sys.executable, KUHAR, 'start', '--ratio', str(ratio), '--limit', str(limit)]
            print(f'启动: python3 {KUHAR} start --ratio {ratio} --limit {limit}')
            p = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
            procs['kuhar'] = p  # 存 Popen 对象，方便 wait()
            time.sleep(30)
            if p.poll() is not None:
                try:
                    p.wait()
                except Exception as e:
                    print(f'[kuhar] wait() 失败: {e}')

        # 保存 PIDs 到文件（从 Popen 对象提取 PID）
        pid_data = {
            "launch_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "processes": {ds: p.pid for ds, p in procs.items()}
        }
        save_pid_file(pid_data)

        started = len(procs)
        print(f'\n共启动 {started} 个进程，监控中（每30秒检查一次）...\n')

        # 监控直到全部完成
        while procs:
            time.sleep(30)
            remaining = {}
            for ds, p in list(procs.items()):
                ret = p.poll()
                if ret is None:
                    remaining[ds] = p
                else:
                    try:
                        p.wait()
                    except Exception as e:
                        print(f'[{ds}] wait() 失败: {e} (exit={ret})')
                    else:
                        print(f'[{ds}] 完成 (exit={ret})')
            procs = remaining
            if procs:
                running_ds = list(procs.keys())
                print(f'  剩余: {", ".join(running_ds)}')
            else:
                print('\n所有数据集软标签生成完成！')
                remove_pid_file()
                break
    except Exception:
        traceback.print_exc()
        print(f'DAEMON ERROR - see {LOG_FILE}')
    finally:
        logf.close()

if __name__ == '__main__':
    main()
