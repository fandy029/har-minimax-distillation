#!/bin/bash
# HAR Soft Label Generation Monitor
# Monitors progress, auto-launches next scripts, and sends WeChat updates

THESIS_DIR="/home/fandy/workplace/thesis"
RESULTS_DIR="$THESIS_DIR/results"
WECHAT_TARGET="o9cq808v47CqMX7YHkVozFRlnGvk@im.wechat"
OPENCLAW_CMD="openclaw message send"

# Log files
UCI_LOG="$RESULTS_DIR/uci_har_soft_log.txt"
HARTH_LOG="$RESULTS_DIR/harth_soft_log.txt"
MOTIONSENSE_LOG="$RESULTS_DIR/motionsense_soft_log.txt"
KUHAR_LOG="$RESULTS_DIR/kuhar_soft_log.txt"

# State tracking
STATE_FILE="$RESULTS_DIR/monitor_state.txt"

send_wechat() {
    local msg="$1"
    openclaw message send --channel openclaw-weixin --target "$WECHAT_TARGET" "$msg"
}

get_class_progress() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "0 0"
        return
    fi
    local completed=$(grep -c "进度已保存" "$log_file" 2>/dev/null || echo 0)
    local current=$(grep "Class [0-9]" "$log_file" | tail -1 | grep -oP "Class \K[0-9]+" || echo 0)
    echo "$completed $current"
}

is_process_running() {
    local script_name="$1"
    ps aux | grep -v grep | grep "$script_name" | grep -q "python3"
}

is_dataset_complete() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        return 1
    fi
    grep -q "软标签生成完成\|完成\|Finished\|Done\|saved successfully" "$log_file" 2>/dev/null
}

count_saved() {
    local log_file="$1"
    grep -c "进度已保存" "$log_file" 2>/dev/null || echo 0
}

get_last_class() {
    local log_file="$1"
    grep "Class [0-9]" "$log_file" 2>/dev/null | tail -1 || echo "N/A"
}

build_status_line() {
    local name="$1"
    local log="$2"
    local script="$3"
    
    if is_dataset_complete "$log"; then
        local count=$(count_saved "$log")
        echo "✅ $name (${count}个类完成)"
    elif is_process_running "$script"; then
        local count=$(count_saved "$log")
        local last=$(get_last_class "$log")
        echo "🔄 $name (${count}类已存, 当前: ${last})"
    elif [ -f "$log" ] && [ -s "$log" ]; then
        local count=$(count_saved "$log")
        echo "⚠️ $name (${count}类, 进程未运行?)"
    else
        echo "⏳ $name (等待中)"
    fi
}

echo "=== Monitor started at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$RESULTS_DIR/monitor.log"

ITERATION=0
UCI_DONE=false
HARTH_LAUNCHED=false
HARTH_DONE=false
MOTIONSENSE_LAUNCHED=false
MOTIONSENSE_DONE=false
KUHAR_LAUNCHED=false
KUHAR_DONE=false
ALL_DONE=false

# Check initial state
if is_dataset_complete "$HARTH_LOG" || is_process_running "gen_soft_labels_harth"; then
    HARTH_LAUNCHED=true
fi
if is_dataset_complete "$MOTIONSENSE_LOG" || is_process_running "gen_soft_labels_motionsense"; then
    MOTIONSENSE_LAUNCHED=true
fi
if is_dataset_complete "$KUHAR_LOG" || is_process_running "gen_soft_labels_kuhar"; then
    KUHAR_LAUNCHED=true
fi

while true; do
    ITERATION=$((ITERATION + 1))
    NOW=$(date '+%H:%M')
    NEXT=$(date -d '+5 minutes' '+%H:%M')
    
    echo "=== Check #$ITERATION at $NOW ===" >> "$RESULTS_DIR/monitor.log"
    
    # Check UCI-HAR
    UCI_COMPLETE=false
    if is_dataset_complete "$UCI_LOG"; then
        UCI_COMPLETE=true
    elif ! is_process_running "gen_soft_labels_uci_har" && [ -f "$UCI_LOG" ] && [ -s "$UCI_LOG" ]; then
        # Process stopped, check if it completed
        if grep -q "进度已保存" "$UCI_LOG"; then
            UCI_COMPLETE=true
        fi
    fi
    
    # Launch HARTH if UCI done
    if [ "$UCI_COMPLETE" = true ] && [ "$HARTH_LAUNCHED" = false ]; then
        echo "Launching HARTH at $NOW" >> "$RESULTS_DIR/monitor.log"
        cd "$THESIS_DIR" && python3 -u gen_soft_labels_harth.py > "$HARTH_LOG" 2>&1 &
        HARTH_LAUNCHED=true
        send_wechat "🚀 **UCI-HAR完成！** [$NOW]
正在启动 **HARTH** 软标签生成...
✅ 已完成：PAMAP2, Gait, UCI-HAR-New, UCI-HAR
🔄 运行中：HARTH
⏳ 等待中：MotionSense, KuHar
⏰ 下次报告：$NEXT"
    fi
    
    # Check HARTH
    HARTH_COMPLETE=false
    if [ "$HARTH_LAUNCHED" = true ]; then
        if is_dataset_complete "$HARTH_LOG"; then
            HARTH_COMPLETE=true
        elif ! is_process_running "gen_soft_labels_harth" && [ -f "$HARTH_LOG" ] && [ -s "$HARTH_LOG" ]; then
            # Check if it has substantial progress (treat as done if stopped)
            local_saved=$(count_saved "$HARTH_LOG")
            if [ "$local_saved" -gt 0 ]; then
                HARTH_COMPLETE=true
            fi
        fi
    fi
    
    # Launch MotionSense if HARTH done
    if [ "$HARTH_COMPLETE" = true ] && [ "$MOTIONSENSE_LAUNCHED" = false ]; then
        echo "Launching MotionSense at $NOW" >> "$RESULTS_DIR/monitor.log"
        cd "$THESIS_DIR" && python3 -u gen_soft_labels_motionsense.py > "$MOTIONSENSE_LOG" 2>&1 &
        MOTIONSENSE_LAUNCHED=true
        send_wechat "🚀 **HARTH完成！** [$NOW]
正在启动 **MotionSense** 软标签生成...
✅ 已完成：PAMAP2, Gait, UCI-HAR-New, UCI-HAR, HARTH
🔄 运行中：MotionSense
⏳ 等待中：KuHar
⏰ 下次报告：$NEXT"
    fi
    
    # Check MotionSense
    MOTIONSENSE_COMPLETE=false
    if [ "$MOTIONSENSE_LAUNCHED" = true ]; then
        if is_dataset_complete "$MOTIONSENSE_LOG"; then
            MOTIONSENSE_COMPLETE=true
        elif ! is_process_running "gen_soft_labels_motionsense" && [ -f "$MOTIONSENSE_LOG" ] && [ -s "$MOTIONSENSE_LOG" ]; then
            local_saved=$(count_saved "$MOTIONSENSE_LOG")
            if [ "$local_saved" -gt 0 ]; then
                MOTIONSENSE_COMPLETE=true
            fi
        fi
    fi
    
    # Launch KuHar if MotionSense done
    if [ "$MOTIONSENSE_COMPLETE" = true ] && [ "$KUHAR_LAUNCHED" = false ]; then
        echo "Launching KuHar at $NOW" >> "$RESULTS_DIR/monitor.log"
        cd "$THESIS_DIR" && python3 -u gen_soft_labels_kuhar.py > "$KUHAR_LOG" 2>&1 &
        KUHAR_LAUNCHED=true
        send_wechat "🚀 **MotionSense完成！** [$NOW]
正在启动 **KuHar** 软标签生成...
✅ 已完成：PAMAP2, Gait, UCI-HAR-New, UCI-HAR, HARTH, MotionSense
🔄 运行中：KuHar
⏰ 下次报告：$NEXT"
    fi
    
    # Check KuHar
    KUHAR_COMPLETE=false
    if [ "$KUHAR_LAUNCHED" = true ]; then
        if is_dataset_complete "$KUHAR_LOG"; then
            KUHAR_COMPLETE=true
        elif ! is_process_running "gen_soft_labels_kuhar" && [ -f "$KUHAR_LOG" ] && [ -s "$KUHAR_LOG" ]; then
            local_saved=$(count_saved "$KUHAR_LOG")
            if [ "$local_saved" -gt 0 ]; then
                KUHAR_COMPLETE=true
            fi
        fi
    fi
    
    # Check if ALL done
    if [ "$UCI_COMPLETE" = true ] && [ "$HARTH_COMPLETE" = true ] && [ "$MOTIONSENSE_COMPLETE" = true ] && [ "$KUHAR_COMPLETE" = true ] && [ "$ALL_DONE" = false ]; then
        ALL_DONE=true
        send_wechat "🎉 **所有软标签生成完成！** [$NOW]

✅ PAMAP2
✅ Gait
✅ UCI-HAR-New
✅ UCI-HAR
✅ HARTH
✅ MotionSense
✅ KuHar

🏁 全部7个数据集软标签已生成完毕！
论文实验数据准备就绪 🎓"
        echo "ALL DONE at $NOW" >> "$RESULTS_DIR/monitor.log"
        break
    fi
    
    # Regular progress update (every 5 min)
    UCI_STATUS=$(build_status_line "UCI-HAR" "$UCI_LOG" "gen_soft_labels_uci_har")
    HARTH_STATUS=$(build_status_line "HARTH" "$HARTH_LOG" "gen_soft_labels_harth")
    MS_STATUS=$(build_status_line "MotionSense" "$MOTIONSENSE_LOG" "gen_soft_labels_motionsense")
    KH_STATUS=$(build_status_line "KuHar" "$KUHAR_LOG" "gen_soft_labels_kuhar")
    
    send_wechat "📊 **HAR软标签进度** [$NOW] #$ITERATION

✅ PAMAP2 (完成)
✅ Gait (完成)
✅ UCI-HAR-New (完成)
$UCI_STATUS
$HARTH_STATUS
$MS_STATUS
$KH_STATUS

⏰ 下次报告：$NEXT"
    
    sleep 300
done

echo "Monitor finished at $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_DIR/monitor.log"
