#!/bin/bash
# 软标签生成 - 全部数据集（全量生成，后台运行）
# 用法: ./gen_soft.sh [--force]
cd "$(dirname "$0")"
LOG="./gen_soft.log"

# 清除所有代理环境变量，避免 Python subprocess 变慢
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy no_proxy

# 用 nohup 方式启动，绕过 exec tool 的环境继承问题
nohup env -i PATH="$PATH" HOME="$HOME" USER="$USER" bash -c 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy no_proxy; python3 -u scripts/run_gen_all.py "$@"' _ "$@" >> "$LOG" 2>&1 &
echo "[后台] 软标签生成已启动 (PID=$!)"
echo "工作目录: $(pwd)"
echo "启动日志: $LOG"
echo "子进程日志: results/logs/gen_full_<dataset>.log"
echo "查看启动日志: tail -f $LOG"
echo "杀进程: ./kill_all.sh"
