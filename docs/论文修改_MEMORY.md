# 论文生成 Memory（2026-05-12）

## 论文路径
- 原始 docx：`/home/fandy/workplace/thesis/docs/论文5_11.docx`
- 新生成章节：`/home/fandy/workplace/thesis/docs/` 下的 md 文件
  - `chapter1_绪论.md`（完整）
  - `chapter2_预处理与Prompt设计.md`（完整）
  - `chapter3_训练方案设计.md`（完整，不含具体训练代码和结果）
  - `chapter4_实验设计与结果.md`（框架完成，结果表格为空）

## 关键决策

### 数据集
- **确定用 5 个**：Gait、PAMAP2、MotionSense、HARTH、KuHar
- wisdm、uci_har、uci_har_new、motionsense_dm 暂不使用

### 学生模型架构
- 三种：Pure CNN、CNN-Residual、Transformer
- 第 3 章已写各架构的设计思路和理论依据
- 具体代码和训练还未跑，第 4 章表格留空

### 蒸馏参数
- 框架：L = α·CE(hard) + (1-α)·T²·KL(soft || logits/T)
- **不使用 V1/V2/V3 命名框架**，直接搜索 (T, α) 组合
- T 范围：1.0~3.5，α 范围：0.1~0.9
- **软标签版本不在论文中提及 A/B/C**，最终只体现一版（merge 后直接使用）

### 写作要求
- **不提及软标签生成的重试策略、hint 自校正等内部细节**
- **不提及 A/B/C 三版软标签**
- AI 味降低：短句、具体数据、去掉套话

## 待完成事项

### 软标签生成（进行中）
- Gait: ✅ 100%，MotionSense: ✅ 100%
- PAMAP2: 🔄 ~87%，KuHar: 🔄 ~76%，HARTH: 🔄 ~53%

### 训练（未开始）
- [ ] 各数据集软标签 merge（只输出最终一版）
- [ ] Pure CNN / CNN-Residual / Transformer 无蒸馏基线
- [ ] 三种架构 × 多组 (T, α) 蒸馏训练
- [ ] 汇总结果填入第 4 章表格

### 论文补充
- [ ] 第 4 章填入实验结果和分析
- [ ] 第 5 章（总结与展望）根据最终结果撰写
- [ ] 第 6 章参考文献补充 ResNet、Transformer 相关引用
- [ ] 第 7 章致谢（保持不变）
- [ ] 第 8 章附录更新
- [ ] 最终合并输出完整 docx

## 公式速查
- 蒸馏 loss：L = α·CE(hard) + (1-α)·T²·KL(soft || logits/T)
- Focal Loss：FL = -(1-pt)^γ · log(pt)，γ=2
- α 含义：α 越高越侧重硬标签，α=1.0 退化为纯 CE
- T 含义：T 越大软标签越平坦
