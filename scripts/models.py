#!/usr/bin/env python3
"""论文第3章定义的三种学生模型架构"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureCNN(nn.Module):
    """3.2.1 Pure CNN 基线模型
    4层卷积块 → AdaptiveAvgPool1d(8) → FC 2048→128→64→n_cls
    """
    def __init__(self, in_channels, n_cls, dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)))


class ResidualBlock(nn.Module):
    """残差块: Conv→BN→ReLU→Conv→BN + skip"""
    def __init__(self, channels, kernel_size=3, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r, inplace=True)


class CNNResidual(nn.Module):
    """3.2.2 CNN-Residual"""
    def __init__(self, in_channels, n_cls, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.res1 = nn.Sequential(ResidualBlock(32, 3, dropout), ResidualBlock(32, 3, dropout))
        self.downsample = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(ResidualBlock(64, 3, dropout), ResidualBlock(64, 3, dropout))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        x = self.stem(x); x = self.res1(x); x = self.downsample(x); x = self.res2(x)
        return self.fc(self.pool(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """3.2.3 Transformer (Encoder-Only)"""
    def __init__(self, in_channels, n_cls, d_model=64, nhead=4, num_layers=3, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x); x = self.pos_encoder(x); x = self.encoder(x)
        return self.fc(x.mean(dim=1))


def build_model(name, in_channels, n_cls, **kwargs):
    models = {'purecnn': PureCNN, 'cnnres': CNNResidual, 'transformer': TransformerModel}
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](in_channels=in_channels, n_cls=n_cls, **kwargs)
