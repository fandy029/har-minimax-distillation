#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提示模板结构图 - 无乱码，清晰布局"""
from PIL import Image, ImageDraw, ImageFont
import os, unicodedata

W, H = 1600, 900
img = Image.new('RGBA', (W, H), (255, 255, 255, 255))
d = ImageDraw.Draw(img)

CN = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ENB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def fc(sz): return ImageFont.truetype(CN, sz)
def fe(sz, bold=False): return ImageFont.truetype(ENB if bold else EN, sz)

def is_cjk(ch):
    if not ch.strip(): return False
    try:
        cat = unicodedata.category(ch)
        if cat in ('Mn', 'Pc'): return True
        return 'CJK' in unicodedata.name(ch, '')
    except: return False

def getw(t, f):
    b = d.textbbox((0,0), t, font=f); return b[2]-b[0]
def geth(t, f):
    b = d.textbbox((0,0), t, font=f); return b[3]-b[1]

def draw_at(cx, cy, text, sc, se, color='black'):
    """在(cx,cy)居中渲染混排文本"""
    parts = []; cur = ''; lang = None
    for ch in text:
        l = 'cjk' if is_cjk(ch) else 'en'
        if lang and l != lang:
            parts.append((lang, cur)); cur = ch; lang = l
        else:
            cur += ch; lang = l
    if cur: parts.append((lang, cur))
    total = sum(getw(t, fc(sc) if l=='cjk' else fe(se)) for l,t in parts)
    x = cx - total // 2
    for l, t in parts:
        f = fc(sc) if l == 'cjk' else fe(se)
        h = geth(t, f)
        d.text((x, cy - h//2 - 2), t, fill=color, font=f)
        x += getw(t, f)

def rbox(cx, cy, w, h, r=8, fill='white', outline='#AAAAAA', lw=2):
    x1,y1 = cx-w//2, cy-h//2; x2,y2 = cx+w//2, cy+h//2
    d.rounded_rectangle([x1,y1,x2,y2], radius=r, fill=fill, outline=outline, width=lw)

def arr_down(cx, y1, y2, color='#555', lw=2):
    d.line([(cx,y1),(cx,y2)], fill=color, width=lw)
    if y2>y1: d.polygon([(cx,y2),(cx-7,y2-12),(cx+7,y2-12)], fill=color)

def txt_l(x, y, text, sc, se, color='#333333'):
    draw_at(x, y, text, sc, se, color)

# 边框
d.rectangle([6,6,W-6,H-6], outline='#AAAAAA', width=2)

# ===== 标题 =====
draw_at(W//2, 50, '提示模板结构图', 26, 26, '#2C3E50')

# ===== 四个组件框（横向）=====
ITEMS = [
    ('任务说明',     '明确告知大模型当前任务是\n分析IMU传感器窗口数据\n并判断对应的活动类型',   '#2980B9'),
    ('特征描述',     '以结构化文本形式列出\n物理特征数值\n加速度/角速度/频率等',       '#27AE60'),
    ('约束条件',     '要求基于物理规律推理\n而非统计模式匹配\n确保推理过程符合科学原则',   '#E67E22'),
    ('输出格式',     '要求JSON格式输出\n各类别的概率分布\n便于后续程序解析利用',         '#9B59B6'),
]

Y1 = 240
BW = 300; BH = 150
GAP = 70
total_w = 4*BW + 3*GAP
START_X = (W - total_w) // 2 + BW//2

for i, (title, desc, color) in enumerate(ITEMS):
    cx = START_X + i*(BW+GAP)
    
    # 框
    rbox(cx, Y1, BW, BH, 12, color, 'white', 2)
    
    # 标题
    draw_at(cx, Y1-45, title, 18, 18, 'white')
    
    # 描述（分行）
    lines = desc.split('\n')
    for li, line in enumerate(lines):
        draw_at(cx, Y1-5 + li*28, line, 13, 13, 'white')

# ===== 向下箭头 ======
arr_down(W//2, Y1+BH//2+5, Y1+BH//2+55)

# ===== 下方输出示例 =====
Y2 = Y1 + BH//2 + 90
rbox(W//2, Y2, 1050, 130, 12, '#F0F4F8', '#2980B9', 2)

draw_at(W//2, Y2-42, 'JSON输出示例', 15, 15, '#2C3E50')

# JSON内容（分行）
json_lines = [
    '{ "站立": 0.82,  "坐着": 0.05,  "行走": 0.08,  "跑步": 0.03,',
    '  "上楼梯": 0.01, "下楼梯": 0.01, "骑自行车": 0.01 }',
]
for li, line in enumerate(json_lines):
    draw_at(W//2, Y2-5 + li*22, line, 13, 13, '#555555')

# 说明
Y3 = Y2 + 90
d.text((80, Y3), '物理特征：加速度幅值均值/标准差/峰值、角速度幅值均值/标准差/峰值、主频率、信号能量',
        fill='#666666', font=fe(13))
d.text((80, Y3+28), '约束条件：判断应基于物理规律（如力学原理、运动学原理），而非简单的数据模式匹配或统计规律',
        fill='#666666', font=fe(13))

out = '/home/fandy/workplace/thesis/figures/prompt_structure.png'
os.makedirs('/home/fandy/workplace/thesis/figures', exist_ok=True)
img.convert('RGB').save(out, 'PNG')
print(f'OK: {out}')
