#!/bin/bash
# HAR Soft Label Monitor - Auto-launch + WeChat notifications

THESIS="/home/fandy/workplace/thesis"
RESULTS="$THESIS/results"

UCI_LOG="$RESULTS/uci_har_soft_log.txt"
HARTH_LOG="$RESULTS/harth_soft_log.txt"
MS_LOG="$RESULTS/motionsense_soft_log.txt"
KH_LOG="$RESULTS/kuhar_soft_log.txt"
MON_LOG="$RESULTS/monitor.log"
WECHAT="o9cq808v47CqMX7YHkVozFRlnGvk@im.wechat"

log() { echo "[$(date '+%H:%M:%S')] $1" >> "$MON_LOG"; }
say() { openclaw message send --channel openclaw-weixin --target "$WECHAT" --message "$1" >> "$MON_LOG" 2>&1; }

running() { ps aux | grep -v grep | grep -q "$1"; }

saved_count() { [ -f "$1" ] && grep -c "进度已保存" "$1" 2>/dev/null; }
last_class() { grep "Class [0-9]" "$1" 2>/dev/null | tail -1 | sed 's/^[ ]*//' || echo "无"; }
has_done_msg() { grep -q "软标签生成完成\|All.*done\|完全完成\|全部完成\|已保存\|完成\|Finished\|Done" "$1" 2>/dev/null; }

# State
uci_done=false; harth_launched=false; harth_done=false
ms_launched=false; ms_done=false; kh_launched=false; kh_done=false

# Detect already-running processes
running "gen_soft_labels_harth" && harth_launched=true
running "gen_soft_labels_motionsense" && ms_launched=true && harth_launched=true
running "gen_soft_labels_kuhar" && kh_launched=true && ms_launched=true && harth_launched=true

# Detect completed datasets (file exists and has done message, or process done with progress)
if has_done_msg "$HARTH_LOG"; then harth_launched=true; harth_done=true; fi
if has_done_msg "$MS_LOG"; then ms_launched=true; harth_launched=true; ms_done=true; fi
if has_done_msg "$KH_LOG"; then kh_launched=true; ms_launched=true; harth_launched=true; kh_done=true; fi

log "Monitor started. harth_launched=$harth_launched ms_launched=$ms_launched kh_launched=$kh_launched"
say "🔄 HAR软标签监控已重启 [$NOW]
当前状态:
- UCI-HAR: 已完成
- HARTH: 已完成  
- MotionSense: $([ "$ms_done" = true ] && echo "已完成" || echo "运行中")
- KuHar: 等待中"

ITER=0
while true; do
    ITER=$((ITER+1))
    NOW=$(date '+%H:%M')
    NEXT=$(date -d '+5 minutes' '+%H:%M' 2>/dev/null)

    log "=== Check $ITER at $NOW ==="

    # UCI-HAR status
    uci_count=$(saved_count "$UCI_LOG")
    uci_last=$(last_class "$UCI_LOG")
    if running "gen_soft_labels_uci_har"; then
        uci_status="🔄 UCI-HAR (${uci_count}类完成, 当前:${uci_last})"
    elif has_done_msg "$UCI_LOG" || [ "$uci_count" -ge 6 ]; then
        uci_done=true; uci_status="✅ UCI-HAR (完成)"
    elif [ "$uci_count" -gt 0 ]; then
        uci_status="⚠️ UCI-HAR (${uci_count}类, 进程已停止)"
        [ "$uci_count" -ge 6 ] && uci_done=true && uci_status="✅ UCI-HAR (完成)"
    else
        uci_status="⏳ UCI-HAR (未开始)"
    fi

    # Launch HARTH if UCI done
    if [ "$uci_done" = true ] && [ "$harth_launched" = false ]; then
        log "LAUNCH: HARTH"
        cd "$THESIS" && python3 -u gen_soft_labels_harth.py > "$HARTH_LOG" 2>&1 &
        harth_launched=true
        say "🚀 UCI-HAR完成！正在启动HARTH [$NOW]
✅ PAMAP2 ✅ Gait ✅ UCI-HAR-New ✅ UCI-HAR
🔄 HARTH 启动中...
⏳ 等待中: MotionSense, KuHar
⏰ 下次报告: $NEXT"
    fi

    # HARTH status
    harth_count=$(saved_count "$HARTH_LOG")
    harth_last=$(last_class "$HARTH_LOG")
    if running "gen_soft_labels_harth"; then
        harth_status="🔄 HARTH (${harth_count}类完成, 当前:${harth_last})"
    elif [ "$harth_launched" = false ]; then
        harth_status="⏳ HARTH (等待中)"
    elif has_done_msg "$HARTH_LOG"; then
        harth_done=true; harth_status="✅ HARTH (完成)"
    elif [ "$harth_count" -gt 0 ]; then
        harth_status="🔄 HARTH (${harth_count}类, 处理中)"
    else
        harth_status="🔄 HARTH (启动中)"
    fi

    # Launch MotionSense if HARTH done
    if [ "$harth_done" = true ] && [ "$ms_launched" = false ]; then
        log "LAUNCH: MotionSense"
        cd "$THESIS" && python3 -u gen_soft_labels_motionsense.py > "$MS_LOG" 2>&1 &
        ms_launched=true
        say "🚀 HARTH完成！正在启动MotionSense [$NOW]
✅ PAMAP2 ✅ Gait ✅ UCI-HAR-New ✅ UCI-HAR ✅ HARTH
🔄 MotionSense 启动中...
⏳ 等待中: KuHar
⏰ 下次报告: $NEXT"
    fi

    # MotionSense status
    ms_count=$(saved_count "$MS_LOG")
    ms_last=$(last_class "$MS_LOG")
    if running "gen_soft_labels_motionsense"; then
        ms_status="🔄 MotionSense (${ms_count}类完成, 当前:${ms_last})"
    elif [ "$ms_launched" = false ]; then
        ms_status="⏳ MotionSense (等待中)"
    elif has_done_msg "$MS_LOG"; then
        ms_done=true; ms_status="✅ MotionSense (完成)"
    elif [ "$ms_count" -gt 0 ]; then
        ms_status="🔄 MotionSense (${ms_count}类, 处理中)"
    else
        ms_status="🔄 MotionSense (启动中)"
    fi

    # Launch KuHar if MotionSense done
    if [ "$ms_done" = true ] && [ "$kh_launched" = false ]; then
        log "LAUNCH: KuHar"
        cd "$THESIS" && python3 -u gen_soft_labels_kuhar.py > "$KH_LOG" 2>&1 &
        kh_launched=true
        say "🚀 MotionSense完成！正在启动KuHar [$NOW]
✅ PAMAP2 ✅ Gait ✅ UCI-HAR-New ✅ UCI-HAR ✅ HARTH ✅ MotionSense
🔄 KuHar 启动中...
⏰ 下次报告: $NEXT"
    fi

    # KuHar status
    kh_count=$(saved_count "$KH_LOG")
    kh_last=$(last_class "$KH_LOG")
    if running "gen_soft_labels_kuhar"; then
        kh_status="🔄 KuHar (${kh_count}类完成, 当前:${kh_last})"
    elif [ "$kh_launched" = false ]; then
        kh_status="⏳ KuHar (等待中)"
    elif has_done_msg "$KH_LOG"; then
        kh_done=true; kh_status="✅ KuHar (完成)"
    elif [ "$kh_count" -gt 0 ]; then
        kh_status="🔄 KuHar (${kh_count}类, 处理中)"
    else
        kh_status="🔄 KuHar (启动中)"
    fi

    # ALL DONE?
    if [ "$uci_done" = true ] && [ "$harth_done" = true ] && [ "$ms_done" = true ] && [ "$kh_done" = true ]; then
        say "🎉 全部HAR软标签生成完毕！[$NOW]

✅ PAMAP2
✅ Gait  
✅ UCI-HAR-New
✅ UCI-HAR
✅ HARTH
✅ MotionSense
✅ KuHar

🏁 7个数据集软标签全部完成！
论文实验数据就绪 🎓"
        log "ALL DONE at $NOW"
        break
    fi

    # Regular 5-min update
    say "📊 HAR软标签进度 [$NOW] #$ITER

✅ PAMAP2 (完成)
✅ Gait (完成)
✅ UCI-HAR-New (完成)
$uci_status
$harth_status
$ms_status
$kh_status

⏰ 下次报告: $NEXT"

    log "Update sent. Sleeping 300s..."
    sleep 300
done

log "Monitor exited normally."
