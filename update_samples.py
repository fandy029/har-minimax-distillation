import re

# Per-class sample counts: 25% of average per-class train samples, capped at 200
configs = {
    'gen_soft_labels_pamap2.py': 141,
    'gen_soft_labels_gait.py': 29,
    'gen_soft_labels_uci_har_new.py': 129,
    'gen_soft_labels_uci_har.py': 200,
    'gen_soft_labels_harth.py': 200,
    'gen_soft_labels_motionsense.py': 200,
    'gen_soft_labels_kuhar.py': 200,
}

for fname, count in configs.items():
    with open(fname, 'r') as f:
        content = f.read()
    content = re.sub(r'SAMPLES_PER_CLASS = \d+', f'SAMPLES_PER_CLASS = {count}', content)
    content = re.sub(r'time\.sleep\([\d.]+\)', 'time.sleep(0.12)', content)
    with open(fname, 'w') as f:
        f.write(content)
    print(f'{fname}: SAMPLES_PER_CLASS = {count}')
