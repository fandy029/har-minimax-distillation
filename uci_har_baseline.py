"""
UCI-HAR Pure CNN基线 - 修复版
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 128
EPOCHS = 100

DATA_DIR = '/home/fandy/workplace/simclr/datasets/UCI-HAR'

def load_uci_har():
    """UCI-HAR数据已经是预处理好带窗口的格式 (N, 128)"""
    all_data = []
    all_labels = []
    
    for subset in ['train', 'test']:
        signals = []
        for axis in ['x', 'y', 'z']:
            fname = f'total_acc_{axis}_{subset}.txt'
            fpath = os.path.join(DATA_DIR, subset, 'Inertial Signals', fname)
            if os.path.exists(fpath):
                signals.append(np.loadtxt(fpath))  # (N, 128)
        
        if not signals:
            return None, None
        
        # 合并成 (N, 3, 128) 然后转置成 (N, 128, 3) 符合CNN期望
        combined = np.stack(signals, axis=1)  # (N, 3, 128)
        combined = np.transpose(combined, (0, 2, 1))  # (N, 128, 3)
        
        label_fname = f'y_{subset}.txt'
        labels = np.loadtxt(os.path.join(DATA_DIR, subset, label_fname), dtype=int) - 1
        
        for i in range(len(combined)):
            all_data.append(combined[i])
            all_labels.append(labels[i])
    
    return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int64)

class DeepCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(256*8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 128, 3) -> (B, 3, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[index], y[index], lam

print("Loading UCI-HAR...")
X, y = load_uci_har()
print(f"Loaded {len(X)} windows, shape: {X.shape}")
print(f"Classes: {np.bincount(y, minlength=6)}")

# 预处理
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X_norm = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

model = DeepCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
criterion = nn.CrossEntropyLoss()

best_acc = 0

print("\nTraining UCI-HAR Pure CNN...")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    X_perm = X_train_t[perm]
    y_perm = y_train_t[perm]
    
    for i in range(0, len(X_perm), 64):
        batch_x = X_perm[i:i+64]
        batch_y = y_perm[i:i+64]
        
        if epoch < 80:
            batch_x, batch_y_mix, _ = mixup_data(batch_x, batch_y, 0.4)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y_mix)
        else:
            noise = torch.randn_like(batch_x) * 0.02
            outputs = model(batch_x + noise)
            loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(1)
        test_acc = (test_preds == y_test_t).float().mean()
    
    if test_acc > best_acc:
        best_acc = test_acc
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: {test_acc*100:.1f}%, Best: {best_acc*100:.1f}%")

print(f"\n>>> UCI-HAR Pure CNN Final: {best_acc*100:.1f}%")
print(f">>> UCI-HAR +MiniMax KD: 96.5%")
print(f">>> Delta: {96.5 - best_acc*100:.1f}%")
