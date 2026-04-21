#!/bin/bash
# ============================================================
# 软标签生成 + 训练 完整流程脚本
# 自动依次处理9个数据集，每步详细记录
# ============================================================
set -e

SCRIPT_DIR="/home/fandy/workplace/thesis"
SOFT_SCRIPT="$SCRIPT_DIR/gen_soft_labels_unified.py"
TRAIN_SCRIPT="$SCRIPT_DIR/run_distill.py"
LOG_BASE="/home/fandy/workplace/thesis/results/logs"
DONE_FILE="/tmp/pipeline_done"

# 9个数据集（顺序）
DATASETS=("pamap2" "kuhar" "uci_har" "harth" "uci_har_new" "motionsense" "gait" "wisdm" "motionsense_dm")

# 每个数据集的目标样本数（每类样本数×类别数）
declare -A TARGETS
TARGETS["pamap2"]=150
TARGETS["kuhar"]=3600
TARGETS["uci_har"]=1200
TARGETS["harth"]=1200
TARGETS["uci_har_new"]=1548
TARGETS["motionsense"]=1200
TARGETS["gait"]=116
TARGETS["wisdm"]=1200
TARGETS["motionsense_dm"]=1200

log() { echo "[$(date '+%m-%d %H:%M:%S')] $1"; }

is_dataset_done() {
    ds=$1
    tgt=${TARGETS[$ds]}
    f="$SCRIPT_DIR/results/soft_labels/${ds}_soft.npy"
    [ ! -f "$f" ] && return 1
    real=$(python3 -c "
import numpy as np
arr = np.load('$f')
is_onehot = (arr > 0.99).sum(axis=1) == 1
real = int(np.sum((arr.sum(axis=1) > 0) & ~is_onehot))
print(real)
" 2>/dev/null)
    [ "$real" -ge "$tgt" ]
}

is_training_done() {
    ds=$1; ver=$2
    ckpt="$SCRIPT_DIR/results/checkpoints/${ds}_${ver}_best.pt"
    [ -f "$ckpt" ]
}

wait_and_check() {
    log "等待进程结束..."
    while pgrep -f "gen_soft_labels_unified\|run_distill" > /dev/null 2>&1; do
        sleep 30
    done
    log "进程已结束，检查结果..."
}

# 主循环
for ds in "${DATASETS[@]}"; do
    tgt=${TARGETS[$ds]}
    log "========== 处理数据集: $ds (目标 $tgt 个真软标签) =========="
    
    # 1. 软标签生成
    if ! is_dataset_done "$ds"; then
        log "[$ds] 开始生成软标签..."
        cd "$SCRIPT_DIR"
        PYTHONUNBUFFERED=1 nohup python3 -u gen_soft_labels_unified.py "$ds" > "$LOG_BASE/gen_${ds}.log" 2>&1 &
        GEN_PID=$!
        log "[$ds] 软标签生成进程 PID=$GEN_PID"
        
        # 等待完成（每30秒检查一次）
        while kill -0 $GEN_PID 2>/dev/null; do
            sleep 30
            # 每分钟打印进度
            real=$(python3 -c "
import numpy as np
arr = np.load('$SCRIPT_DIR/results/soft_labels/${ds}_soft.npy')
is_onehot = (arr > 0.99).sum(axis=1) == 1
real = int(np.sum((arr.sum(axis=1) > 0) & ~is_onehot))
print(real)
" 2>/dev/null || echo "0")
            log "[$ds] 当前进度: $real / $tgt"
        done
        
        wait $GEN_PID
        log "[$ds] 软标签生成完成!"
    else
        log "[$ds] 软标签已存在，跳过生成"
    fi
    
    # 2. 训练4个版本
    for ver in pure_cnn v1 v2 v3; do
        if ! is_training_done "$ds" "$ver"; then
            log "[$ds] 训练 $ver..."
            cd "$SCRIPT_DIR"
            PYTHONUNBUFFERED=1 nohup python3 -u run_distill.py "$ds" "$ver" > "$LOG_BASE/train_${ds}_${ver}.log" 2>&1 &
            TRAIN_PID=$!
            log "[$ds] $ver 训练进程 PID=$TRAIN_PID"
            
            while kill -0 $TRAIN_PID 2>/dev/null; do
                sleep 30
            done
            
            wait $TRAIN_PID
            log "[$ds] $ver 训练完成!"
        else
            log "[$ds] $ver 已训练过，跳过"
        fi
    done
    
    log "========== $ds 处理完毕 =========="
done

log "全部9个数据集处理完毕！"
touch "$DONE_FILE"
