#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAIT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET="$GAIT_DIR/output"
if [ -d "$TARGET" ]; then
    echo "将删除: $TARGET"
    read -p "确认删除? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET"
        echo "已删除"
    else
        echo "取消"
    fi
else
    echo "output/ 目录不存在"
fi
