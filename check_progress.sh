#!/bin/bash
cd /home/fandy/workplace/thesis

# 检查运行状态
RUNNING=$(pgrep -f "distill_" | wc -l)
if [ $RUNNING -eq 0 ]; then
    echo "[$(date)] 没有实验在运行"
    exit 0
fi

# 读取当前运行的实验
STATUS_FILE="results/.monitor_status.json"
CURRENT=""
if [ -f "$STATUS_FILE" ]; then
    CURRENT=$(python3 -c "import json; s=json.load(open('$STATUS_FILE')); print(s.get('running',{}).get('dataset','unknown'))" 2>/dev/null)
fi

echo "[$(date)] 正在运行实验: $CURRENT (PID数: $RUNNING)"

# 检查results目录的最新文件
for f in results/*.json; do
    if [ -f "$f" ]; then
        echo "  - $(basename $f): $(date -r $f '+%H:%M:%S')"
    fi
done
