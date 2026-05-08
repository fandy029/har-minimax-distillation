#!/bin/bash
# run_all_gen.sh - 启动所有软标签生成脚本
# 用法: ./run_all_gen.sh [start|stop|status]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    "gait:gait_gen.py"
    "harth:harth_gen.py"
    "kuhar:kuhar_gen.py"
    "motionsense:motionsense_gen.py"
    "pamap2:pamap2_gen.py"
    "uci_har:uci_har_gen.py"
)

PID_DIR="$SCRIPT_DIR/.run_pids"
mkdir -p "$PID_DIR"

start() {
    echo "启动所有软标签生成脚本..."
    for item in "${SCRIPTS[@]}"; do
        name="${item%%:*}"
        script="${item##*:}"
        script_path="$SCRIPT_DIR/scripts/$name/$script"
        pid_file="$PID_DIR/${name}.pid"

        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "  [SKIP] $name (已在运行, PID=$(cat "$pid_file"))"
            continue
        fi

        echo "  [START] $name${FORCE_ARG:+(传递 --force)}"
        cd "$SCRIPT_DIR"
        nohup python3 "$script_path" $FORCE_ARG > /dev/null 2>&1 &
        echo $! > "$pid_file"
        echo "  [PID] $name started with PID $(cat "$pid_file")"
    done
    echo "全部启动完成"
}

stop() {
    echo "停止所有软标签生成脚本..."
    for item in "${SCRIPTS[@]}"; do
        name="${item%%:*}"
        pid_file="$PID_DIR/${name}.pid"

        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  [STOP] $name (PID=$pid)"
                kill "$pid" 2>/dev/null
                # 等待进程结束
                for i in {1..5}; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                # 强制杀死
                kill -9 "$pid" 2>/dev/null
            else
                echo "  [SKIP] $name (未运行)"
            fi
            rm -f "$pid_file"
        else
            echo "  [SKIP] $name (无PID文件)"
        fi
    done
    echo "全部停止完成"
}

status() {
    echo "软标签生成脚本运行状态:"
    any_running=0
    for item in "${SCRIPTS[@]}"; do
        name="${item%%:*}"
        pid_file="$PID_DIR/${name}.pid"

        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  [RUN] $name (PID=$pid)"
                any_running=1
            else
                echo "  [DOWN] $name (PID文件存在但进程已退出)"
                rm -f "$pid_file"
            fi
        else
            echo "  [STOP] $name"
        fi
    done
    if [[ $any_running -eq 0 ]]; then
        echo "没有脚本在运行"
    fi
}

FORCE_FLAG=""
FORCE_ARG=""

case "${1:-start}" in
    start)
        start
        ;;
    force)
        echo "强制重启所有软标签生成脚本 (传递 --force)..."
        FORCE_FLAG="1"
        FORCE_ARG="--force"
        stop
        sleep 2
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    *)
        echo "用法: $0 {start|stop|status|restart|force}"
        exit 1
        ;;
esac
