#!/bin/bash
# HAR蒸馏实验进度汇报脚本
# 每20分钟执行一次

THESIS_DIR="/home/fandy/workplace/thesis"
RESULTS_DIR="$THESIS_DIR/results"
LOG_FILE="$RESULTS_DIR/batch_distill_log.txt"
TARGET="o9cq808v47CqMX7YHkVozFRlnGvk@im.wechat"

# 计算运行时长
calc_duration() {
    local start_str="$1"
    local start_epoch=$(date -d "$start_str" +%s 2>/dev/null)
    local now_epoch=$(date +%s)
    local diff=$((now_epoch - start_epoch))
    local hours=$((diff / 3600))
    local mins=$(((diff % 3600) / 60))
    echo "${hours}h ${mins}m"
}

# 获取已完成实验数
COMPLETED=$(grep "完成:" "$LOG_FILE" 2>/dev/null | grep -v "失败" | wc -l)
TOTAL=36

# 获取当前运行中的实验
CURRENT=$(ps aux | grep "run_distill.py" | grep -v grep | head -1)
if [ -n "$CURRENT" ]; then
    CURRENT_NAME=$(echo "$CURRENT" | grep -oP "run_distill\.py \K\w+" || echo "未知")
    CURRENT_PID=$(echo "$CURRENT" | awk '{print $2}')
    CURRENT_START=$(ps -o lstart= -p "$CURRENT_PID" 2>/dev/null || echo "未知")
else
    CURRENT_NAME="无"
    CURRENT_PID="-"
    CURRENT_START="-"
fi

# 获取最后一行日志
LAST_LOG=$(tail -1 "$LOG_FILE" 2>/dev/null)

# 获取已完成实验列表
echo "=== HAR蒸馏实验进度 ==="
echo "已完成: $COMPLETED/$TOTAL"
echo "当前: $CURRENT_NAME (PID: $CURRENT_PID)"
echo "最后日志: $LAST_LOG"

# 构建消息
MSG="📊 **HAR蒸馏实验进度** [$(date '+%H:%M')]

✅ **已完成: $COMPLETED/36**

正在运行:
- $CURRENT_NAME (PID: $CURRENT_PID)

最后日志:
\`$LAST_LOG\`

⏰ 下次报告: $(date -d '+20 minutes' '+%H:%M')"

# 发送微信
openclaw message send --channel openclaw-weixin --target "$TARGET" --message "$MSG"
