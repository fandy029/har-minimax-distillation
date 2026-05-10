#!/bin/bash
# 一键启动/停止所有6个数据集的软标签生成
# 用法:
#   bash run_all.sh            全量生成
#   bash run_all.sh --quick    每类50样本测试
#   bash run_all.sh --stop     停止所有
#   bash run_all.sh --status   查看状态

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

DATA_DIRS=("kuhar" "gait" "pamap2" "uci_har" "motionsense" "harth")

# ===== 停止 =====
if [ "$1" = "--stop" ]; then
    echo "═══ 停止所有数据集生成 ═══"
    for ds in "${DATA_DIRS[@]}"; do
        bash "$SCRIPT_DIR/$ds/${ds}_run.sh" --stop 2>/dev/null
    done
    # 强制清理残留
    PIDS=$(pgrep -f "_gen.py")
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null
        sleep 1
    fi
    echo "全部已停止"
    exit 0
fi

# ===== 状态 =====
if [ "$1" = "--status" ]; then
    echo "═══ 数据集生成状态 ═══"
    printf "%-15s %s\n" "数据集" "进程数"
    echo "─────────────────────────"
    for ds in "${DATA_DIRS[@]}"; do
        count=$(pgrep -fc "${ds}_gen.py" 2>/dev/null || echo 0)
        printf "%-15s %d\n" "$ds" "$count"
    done
    exit 0
fi

# ===== 启动 =====
FLAGS="${@}"
echo "╔══════════════════════════════════════╗"
echo "║  全部数据集软标签生成                 ║"
echo "║  参数: $FLAGS"
echo "╚══════════════════════════════════════╝"
echo ""

echo "启动日志写入各数据集 output/logs/ 目录"
echo ""
for ds in "${DATA_DIRS[@]}"; do
    echo "  $ds ..."
    bash "$SCRIPT_DIR/$ds/${ds}_run.sh" $FLAGS > /dev/null 2>&1 &
    sleep 5
done

echo ""
echo "全部 6 个数据集已启动 (后台运行,输出已重定向)"
echo "查看状态: bash run_all.sh --status"
echo "停止:      bash run_all.sh --stop"
