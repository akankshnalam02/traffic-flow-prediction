# =============================================================
# 🚦 PEMS-BAY Traffic Forecasting Dashboard - FIXED VERSION
# MODEL: MS-GWN-A | Built with Streamlit
# Run: streamlit run dashboard_complete.py
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime, date, time, timedelta

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="PEMS-BAY Traffic Forecasting",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6, .stSubheader {
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 1px;
        color: #000000 !important;
    }
    
    .main { 
        background-color: #ffffff !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    .stApp > header {
        background-color: #ffffff !important;
    }
    
    .stApp [data-testid="stDecoration"] {
        background-color: #ffffff !important;
    }
    
    .stApp [data-testid="stStatusWidget"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: #ffffff !important;
    }
    
    header, header * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }

    .alert-red {
        background: linear-gradient(135deg, #ffe5e5 0%, #ffcccc 100%);
        border-left: 5px solid #ff3333;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
        color: #cc0000;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(255, 51, 51, 0.2);
    }
    .alert-green {
        background: linear-gradient(135deg, #e5ffe5 0%, #ccffcc 100%);
        border-left: 5px solid #00e676;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
        color: #008800;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(0, 230, 118, 0.2);
    }
    .route-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #d1e7ff 100%);
        border: 1px solid #90caf9;
        border-left: 4px solid #29b6f6;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 6px 0;
        color: #0d47a1;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(41, 182, 246, 0.15);
    }
    .route-card b { color: #01579b; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #dee2e6;
        margin: 20px 0;
    }
    .forecast-banner {
        background: linear-gradient(135deg, #f3e5ff 0%, #e8d5ff 100%);
        border-left: 5px solid #9c6cff;
        padding: 14px 20px;
        border-radius: 8px;
        margin: 8px 0;
        color: #5a189a;
        font-weight: 500;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(156, 108, 255, 0.2);
    }
    .forecast-banner b { color: #3c096c; }
    .mode-badge-hist {
        display: inline-block;
        background: #e5ffed;
        color: #008822;
        border: 1px solid #00e676;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .mode-badge-fore {
        display: inline-block;
        background: #f3e5ff;
        color: #5a189a;
        border: 1px solid #9c6cff;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    p, span, div, label, .stMarkdown {
        color: #000000 !important;
    }
    
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        color: #000000 !important;
    }
    
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] div {
        color: #000000 !important;
    }
    
    [role="listbox"], 
    [role="option"],
    ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover {
        background-color: #f0f0f0 !important;
    }
    
    input, select, textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* DATE INPUT AND CALENDAR STYLING - WHITE THEME - ENHANCED */
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stDateInput label,
    .stTimeInput label {
        color: #000000 !important;
    }
    
    /* Calendar popup styling - COMPREHENSIVE FIX */
    [data-baseweb="calendar"] {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
    }
    
    [data-baseweb="calendar"] *,
    [data-baseweb="calendar"] div,
    [data-baseweb="calendar"] button,
    [data-baseweb="calendar"] span {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] header,
    [data-baseweb="calendar"] header * {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [aria-label*="previous"],
    [data-baseweb="calendar"] [aria-label*="next"] {
        background-color: transparent !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [role="columnheader"],
    [data-baseweb="calendar"] thead th {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    [data-baseweb="calendar"] [role="button"],
    [data-baseweb="calendar"] button[role="button"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [role="button"]:hover,
    [data-baseweb="calendar"] button[role="button"]:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [aria-selected="true"],
    [data-baseweb="calendar"] button[aria-selected="true"] {
        background-color: #1976d2 !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="calendar"] [data-highlighted="true"] {
        background-color: #e3f2fd !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [disabled],
    [data-baseweb="calendar"] button[disabled] {
        background-color: #f5f5f5 !important;
        color: #999999 !important;
    }
    
    [data-baseweb="calendar"] [data-baseweb="select"],
    [data-baseweb="calendar"] select {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    [data-baseweb="calendar"] [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [data-range-highlight] {
        background-color: #e3f2fd !important;
    }
    
    [data-baseweb="calendar"] td,
    [data-baseweb="calendar"] td > div,
    [data-baseweb="calendar"] td > button {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="calendar"] [style*="background-color: rgb(0, 0, 0)"],
    [data-baseweb="calendar"] [style*="background: rgb(0, 0, 0)"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stSelectbox label {
        color: #000000 !important;
    }
    
    .stRadio > div {
        background-color: transparent !important;
    }
    
    .stRadio label {
        color: #000000 !important;
    }
    
    .stRadio > div > label > div {
        color: #000000 !important;
    }
    
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    .stSlider > div > div > div {
        background-color: #f0f0f0 !important;
    }
    
    .stSlider label {
        color: #000000 !important;
    }
    
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stButton > button:hover {
        background-color: #f0f0f0 !important;
        border-color: #999999 !important;
    }
    
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1565c0 !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #eeeeee !important;
        color: #000000 !important;
    }
    
    details[open] > summary {
        border-bottom: 1px solid #dee2e6 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #666666 !important;
    }
    
    .stAlert {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    div[data-testid="stMarkdownContainer"] > div[data-testid="stAlert"] {
        background-color: #ffffff !important;
    }
    
    .stCodeBlock {
        background-color: #f5f5f5 !important;
    }
    
    code {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }
    
    .stCaptionContainer,
    [data-testid="stCaptionContainer"],
    small {
        color: #666666 !important;
    }
    
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #f0f0f0 !important;
    }
    
    .stSpinner > div {
        border-top-color: #1976d2 !important;
    }
    
    .stSuccess {
        background-color: #e5ffe5 !important;
        color: #008800 !important;
    }
    
    .stWarning {
        background-color: #fff4e5 !important;
        color: #996300 !important;
    }
    
    .stError {
        background-color: #ffe5e5 !important;
        color: #cc0000 !important;
    }
    
    .stInfo {
        background-color: #e3f2fd !important;
        color: #0d47a1 !important;
    }

    /* DATAFRAME FIX */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] *,
    [data-testid="stDataFrameResizable"] *,
    .stDataFrame *,
    div[data-testid="stDataFrame"] *,
    div[data-testid="stDataFrameResizable"] * {
        color: #000000 !important;
    }
    
    .gdg-cell,
    .gdg-cell > *,
    .gdg-cell span,
    .gdg-cell div,
    .gdg-cell-inner,
    .gdg-cell text {
        color: #000000 !important;
        background-color: #ffffff !important;
        fill: #000000 !important;
    }
    
    .gdg-header-cell,
    .gdg-header-cell > *,
    .gdg-header-cell span,
    .gdg-header-cell div,
    .gdg-header-cell text {
        color: #000000 !important;
        background-color: #e9ecef !important;
        font-weight: bold !important;
        fill: #000000 !important;
    }
    
    .gdg-cell {
        border-right: 1px solid #dee2e6 !important;
        border-bottom: 1px solid #dee2e6 !important;
    }
    
    .gdg-header-cell {
        border-right: 1px solid #adb5bd !important;
        border-bottom: 2px solid #6c757d !important;
    }
    
    [data-testid="stDataFrame"] canvas,
    [data-testid="stDataFrameResizable"] canvas {
        background-color: #ffffff !important;
    }
    
    .dvn-scroller, 
    .dvn-underlay, 
    .dvn-scroll-inner,
    .dvn-container {
        background-color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] svg text,
    [data-testid="stDataFrameResizable"] svg text {
        fill: #000000 !important;
    }
    
    div[class*="glide"] *,
    div[class*="glide"] text,
    div[class*="glide"] span,
    canvas + div {
        color: #000000 !important;
    }
    
    /* HTML table styling */
    .dataframe-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #ffffff;
        color: #000000;
    }
    
    .dataframe-table th {
        background-color: #e9ecef;
        color: #000000;
        padding: 10px;
        border: 1px solid #dee2e6;
        font-weight: bold;
        text-align: left;
    }
    
    .dataframe-table td {
        color: #000000;
        padding: 8px;
        border: 1px solid #dee2e6;
    }
    
    .dataframe-table tr:hover {
        background-color: #f8f9fa;
    }
    
    </style>
""", unsafe_allow_html=True)

# =============================================================
# MODEL ARCHITECTURE (must match saved model exactly)
# =============================================================
class NodeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key   = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.scale = math.sqrt(channels)

    def forward(self, x):
        B, C, N, T = x.shape
        x_pool = x.mean(dim=-1).permute(0, 2, 1)
        Q = self.query(x_pool)
        K = self.key(x_pool)
        V = self.value(x_pool)
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, V).permute(0, 2, 1)
        return out.unsqueeze(-1).expand(B, C, N, T)

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, adj_fixed):
        super().__init__()
        self.register_buffer('adj_fixed', adj_fixed)
        self.adj_learned = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        self.alpha       = nn.Parameter(torch.tensor(0.5))

    def forward(self):
        adj_l = F.relu(self.adj_learned)
        adj_l = adj_l / (adj_l.sum(dim=1, keepdim=True) + 1e-8)
        alpha = torch.sigmoid(self.alpha)
        return alpha * self.adj_fixed + (1 - alpha) * adj_l

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=(1,3), dilation=(1,1), padding=(0,1))
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=(1,3), dilation=(1,2), padding=(0,2))
        self.conv_4 = nn.Conv2d(channels, channels, kernel_size=(1,3), dilation=(1,4), padding=(0,4))
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=(1,1))

    def forward(self, x):
        x1 = F.relu(self.conv_1(x))
        x2 = F.relu(self.conv_2(x))
        x3 = F.relu(self.conv_4(x))
        return self.fusion(torch.cat([x1, x2, x3], dim=1))

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x = torch.bmm(adj.unsqueeze(0).expand(B*T, N, N), x)
        x = self.lin(x).reshape(B, T, N, -1).permute(0, 3, 2, 1)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.attn_weights = nn.Parameter(torch.ones(pred_len) / pred_len)

    def forward(self, x):
        weights = F.softmax(self.attn_weights, dim=0).view(1, -1, 1)
        return x * weights

class MS_GWN_A(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, adj_fixed):
        super().__init__()
        channels   = 48
        num_blocks = 3

        self.adaptive_adj    = AdaptiveAdjacency(num_nodes, adj_fixed)
        self.input_proj      = nn.Conv2d(in_dim, channels, kernel_size=(1,1))
        self.temporal_blocks = nn.ModuleList([MultiScaleTemporalBlock(channels) for _ in range(num_blocks)])
        self.graph_convs     = nn.ModuleList([GraphConvolution(channels, channels) for _ in range(num_blocks)])
        self.node_attentions = nn.ModuleList([NodeAttention(channels) for _ in range(num_blocks)])
        self.skip_convs      = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=(1,1)) for _ in range(num_blocks)])
        self.temporal_pool   = nn.AdaptiveAvgPool2d((num_nodes, 1))
        self.output_proj     = nn.Sequential(
            nn.Linear(channels, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, out_dim)
        )
        self.temporal_attn   = TemporalAttention(out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        adj          = self.adaptive_adj()
        x            = self.input_proj(x)
        skip_outputs = []
        for t_block, g_conv, n_attn, s_conv in zip(
            self.temporal_blocks, self.graph_convs,
            self.node_attentions, self.skip_convs
        ):
            residual = x
            x        = t_block(x)
            x        = F.relu(g_conv(x, adj))
            x        = x + n_attn(x)
            x        = x + residual
            skip_outputs.append(s_conv(x))
        x   = torch.stack(skip_outputs, dim=0).sum(dim=0)
        x   = self.temporal_pool(x).squeeze(-1).permute(0, 2, 1)
        out = self.output_proj(x).permute(0, 2, 1)
        return self.temporal_attn(out)


# =============================================================
# LOAD DATA + MODEL (cached)
# =============================================================
@st.cache_resource
def load_everything():
    device = "cpu"

    csv_path = "pems_bay_final_with_extra_features.csv"
    adj_path = "adj_mx_PEMS-BAY.pkl"

    print("=" * 60)
    print("🔄 LOADING DATA WITH TRAINING NORMALIZATION")
    print("=" * 60)

    df = pd.read_csv(csv_path, index_col="timestamp", low_memory=False)
    df.index = pd.to_datetime(df.index, format="%d-%m-%Y %H:%M")
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    sensor_cols = [c for c in df.columns if c.isdigit()]
    time_cols   = [c for c in df.columns if c not in sensor_cols]

    traffic   = df[sensor_cols].to_numpy(dtype=np.float32)
    time_feat = df[time_cols].to_numpy(dtype=np.float32)

    mean_path = "train_mean.npy"
    std_path = "train_std.npy"
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        print("✅ Loaded normalization from train_mean.npy and train_std.npy")
    else:
        raise FileNotFoundError(
            "❌ train_mean.npy and train_std.npy not found!\n"
            "Please run the training script or create these files using:\n"
            "python save_normalization.py"
        )
    
    traffic_norm = ((traffic - mean) / std).astype(np.float32)

    with open(adj_path, "rb") as f:
        adj_data = pickle.load(f, encoding="latin1")
    A = adj_data[2].astype(np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    adj_mx = torch.tensor(A_norm, dtype=torch.float32)

    T, N   = traffic_norm.shape
    F_time = time_feat.shape[1]
    traffic_exp   = traffic_norm[..., None]
    time_exp      = np.broadcast_to(time_feat[:, None, :], (T, N, F_time))
    data          = np.concatenate([traffic_exp, time_exp], axis=2).astype(np.float32)

    split     = int(len(data) * 0.8)
    test_data = data[split:]

    all_timestamps  = df.index.to_numpy()
    test_timestamps = all_timestamps[split:]
    A_numpy         = adj_data[2].astype(np.float32)

    in_dim    = data.shape[2]
    num_nodes = N
    out_dim   = 3

    model = MS_GWN_A(num_nodes=num_nodes, in_dim=in_dim,
                     out_dim=out_dim, adj_fixed=adj_mx)
    model.load_state_dict(torch.load("ms_gwn_a_best.pth", map_location="cpu"))
    model.eval()

    return (model, test_data, mean, std, sensor_cols, adj_mx, A_numpy,
            test_timestamps, data, all_timestamps, device)


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def predict_all_sensors_from_all_data(model, all_data, mean, std, window_idx, num_nodes, device, pred_steps=3):
    SEQ_LEN = 24
    MODEL_PRED_STEPS = 3
    y_np = all_data[window_idx : window_idx + pred_steps, :, 0]
    
    if pred_steps <= MODEL_PRED_STEPS:
        x_np = all_data[window_idx - SEQ_LEN : window_idx]
        pred_real = predict_all_sensors_from_data(model, x_np, mean, std, num_nodes, pred_steps)
    else:
        predictions = []
        current_window = all_data[window_idx - SEQ_LEN : window_idx].copy()
        steps_remaining = pred_steps
        while steps_remaining > 0:
            x_t = torch.from_numpy(current_window).permute(2, 1, 0).unsqueeze(0)
            with torch.no_grad():
                pred = model(x_t)
            steps_to_take = min(MODEL_PRED_STEPS, steps_remaining)
            pred_np_batch = pred[0, :steps_to_take].numpy()
            pred_real_batch = pred_np_batch * std[0] + mean[0]
            predictions.append(pred_real_batch)
            if steps_remaining > MODEL_PRED_STEPS:
                pred_norm = (pred_real_batch - mean[0]) / std[0]
                time_features = current_window[-1, :, 1:]
                new_steps = np.zeros((steps_to_take, current_window.shape[1], current_window.shape[2]))
                new_steps[:, :, 0] = pred_norm
                new_steps[:, :, 1:] = time_features
                current_window = np.concatenate([current_window[steps_to_take:], new_steps], axis=0)
            steps_remaining -= steps_to_take
        pred_real = np.concatenate(predictions, axis=0)
    
    true_real = y_np * std[0] + mean[0]
    return pred_real, true_real


def get_nearby_sensors(sensor_idx, A_numpy, sensor_cols, top_k=5):
    row = A_numpy[sensor_idx].copy()
    row[sensor_idx] = 0
    top_indices = np.argsort(row)[::-1][:top_k]
    return [(i, sensor_cols[i], row[i]) for i in top_indices if row[i] > 0]


def find_matching_windows(all_timestamps, all_data, target_dt, n=5):
    SEQ_LEN = 24
    target_hour = target_dt.hour
    target_dow  = target_dt.weekday()
    matches = []
    for i in range(SEQ_LEN, len(all_timestamps) - 27):
        ts = pd.Timestamp(all_timestamps[i])
        if ts.hour == target_hour and ts.weekday() == target_dow:
            matches.append(i)
        if len(matches) >= n:
            break
    return matches


def predict_all_sensors_from_data(model, data_slice, mean, std, num_nodes, pred_steps=3):
    MODEL_PRED_STEPS = 3
    if pred_steps <= MODEL_PRED_STEPS:
        x_t = torch.from_numpy(data_slice).permute(2, 1, 0).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_t)
        pred_np = pred[0, :pred_steps].numpy()
        pred_real = pred_np * std[0] + mean[0]
    else:
        predictions = []
        current_window = data_slice.copy()
        steps_remaining = pred_steps
        while steps_remaining > 0:
            x_t = torch.from_numpy(current_window).permute(2, 1, 0).unsqueeze(0)
            with torch.no_grad():
                pred = model(x_t)
            steps_to_take = min(MODEL_PRED_STEPS, steps_remaining)
            pred_np_batch = pred[0, :steps_to_take].numpy()
            pred_real_batch = pred_np_batch * std[0] + mean[0]
            predictions.append(pred_real_batch)
            if steps_remaining > MODEL_PRED_STEPS:
                pred_norm = (pred_real_batch - mean[0]) / std[0]
                time_features = current_window[-1, :, 1:]
                new_steps = np.zeros((steps_to_take, current_window.shape[1], current_window.shape[2]))
                new_steps[:, :, 0] = pred_norm
                new_steps[:, :, 1:] = time_features
                current_window = np.concatenate([current_window[steps_to_take:], new_steps], axis=0)
            steps_remaining -= steps_to_take
        pred_real = np.concatenate(predictions, axis=0)
    return pred_real


def dt_to_window(selected_dt, all_timestamps):
    target = pd.Timestamp(selected_dt)
    all_ts_pd = pd.DatetimeIndex(all_timestamps)
    diffs = np.abs((all_ts_pd - target).total_seconds())
    idx = int(np.argmin(diffs))
    idx = max(24, min(idx, len(all_timestamps) - 30))
    return idx


def create_styled_table(df):
    html = '<div style="height: 400px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 8px;">'
    html += '<table class="dataframe-table">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            value = row[col]
            if col == "Status":
                if "Congested" in str(value):
                    html += f'<td style="color: #cc0000; font-weight: bold;">{value}</td>'
                else:
                    html += f'<td style="color: #008800; font-weight: bold;">{value}</td>'
            else:
                html += f'<td>{value}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


# =============================================================
# LOAD
# =============================================================
(model, test_data, mean, std, sensor_cols,
 adj_mx, A_numpy, test_timestamps, all_data,
 all_timestamps, device) = load_everything()

num_nodes = len(sensor_cols)
DATA_START = pd.Timestamp(all_timestamps[24])
DATA_END   = pd.Timestamp(all_timestamps[-27])
CONGESTION_THRESHOLD = 50

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("🔄 FORCE REFRESH DATA", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared! Refreshing...")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 View Mode**")
view_mode = st.sidebar.radio(
    "What do you want to see?",
    ["Current Traffic (Actual)", "Future Prediction"],
    index=0
)

if view_mode == "Future Prediction":
    st.sidebar.markdown("**⏱️ Prediction Time Window**")
    st.sidebar.markdown("How many minutes ahead?")
    time_option = st.sidebar.radio(
        "",
        ["5 minutes", "10 minutes", "15 minutes"],
        index=2,
        label_visibility="collapsed"
    )
    time_window_minutes = int(time_option.split()[0])
    pred_steps = time_window_minutes // 5
    st.sidebar.caption(f"Predicting {pred_steps} steps × 5 min = {time_window_minutes} min ahead")
else:
    time_window_minutes = 15
    pred_steps = 3

st.sidebar.markdown("---")
selected_sensor_name = st.sidebar.selectbox(
    "🔍 Select Sensor (all 325)",
    sensor_cols,
    index=0
)
sensor_idx = sensor_cols.index(selected_sensor_name)

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 Select Date & Time**")
st.sidebar.caption(f"Dataset covers **Jan 1 – Jun 30, 2017**. Dates outside this range use historical pattern matching.")

selected_date = st.sidebar.date_input(
    "Date",
    value=date(2017, 5, 1),
    min_value=date(2017, 1, 1)
)

hour_options   = list(range(24))
minute_options = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
sel_hour   = st.sidebar.selectbox("Hour",   hour_options,   index=8,  format_func=lambda h: f"{h:02d}:00")
sel_minute = st.sidebar.selectbox("Minute", minute_options, index=0,  format_func=lambda m: f"{m:02d}")

selected_dt = datetime(selected_date.year, selected_date.month, selected_date.day, sel_hour, sel_minute)
in_dataset  = (DATA_START.to_pydatetime().date() <= selected_date <= DATA_END.to_pydatetime().date())

if in_dataset:
    window_idx = dt_to_window(selected_dt, all_timestamps)
    is_forecast_mode = False
else:
    is_forecast_mode = True
    window_idx = None

st.sidebar.markdown("---")
st.sidebar.markdown("**🔎 Auto-Find Congested Moment**")
if st.sidebar.button("Find Congested Window (Historical)"):
    with st.spinner("Scanning historical windows..."):
        found_w = None
        for w in range(24, min(1000, len(all_data) - 27), 10):
            slice_data = all_data[w - 24 : w].astype(np.float32)
            p = predict_all_sensors_from_data(model, slice_data, mean, std, num_nodes, 3)
            p_sensor = p[:, sensor_idx].mean()
            if p_sensor < CONGESTION_THRESHOLD:
                found_w = w
                break
        if found_w is not None:
            found_ts = pd.Timestamp(all_timestamps[found_w])
            st.sidebar.success(f"✅ Found congestion at {found_ts.strftime('%d %b %Y  %H:%M')}")
            st.sidebar.info("👆 Set date/time to the timestamp above to view it.")
        else:
            st.sidebar.warning("No congestion found in first 1000 windows for this sensor.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Congestion Threshold:** < {CONGESTION_THRESHOLD} mph 🔴")
st.sidebar.markdown("**Model:** MS-GWN-A")
st.sidebar.markdown("**Dataset:** PEMS-BAY · 325 sensors · 5-min intervals")

# =============================================================
# HEADER
# =============================================================
st.title("🚦 PEMS-BAY Traffic Speed Forecasting")

if view_mode == "Current Traffic (Actual)":
    st.markdown(
        "**Mode: Current Traffic** &nbsp;|&nbsp; Showing actual traffic at selected time "
        f"&nbsp;|&nbsp; Congestion Threshold: **{CONGESTION_THRESHOLD} mph**"
    )
else:
    st.markdown(
        "**Mode: Future Prediction** &nbsp;|&nbsp; Model: **MS-GWN-A** &nbsp;|&nbsp; "
        f"Prediction Horizon: **{time_window_minutes} min ({pred_steps} steps × 5 min)** &nbsp;|&nbsp; Congestion Threshold: **{CONGESTION_THRESHOLD} mph**"
    )

time_display = selected_dt.strftime("%A, %d %b %Y · %H:%M")

if view_mode == "Current Traffic (Actual)":
    st.markdown(
        f"🕐 **Current Time:** {time_display} &nbsp;&nbsp;"
        f'<span class="mode-badge-hist">📊 CURRENT TRAFFIC — Showing actual conditions at this time</span>',
        unsafe_allow_html=True
    )
    st.info("💡 **Viewing actual traffic conditions at the selected time.** Switch to 'Future Prediction' mode to see forecasts.")
else:
    if is_forecast_mode:
        st.markdown(
            f"🕐 **Base Time:** {time_display} &nbsp;&nbsp;"
            f'<span class="mode-badge-fore">🔮 FORECAST MODE — Pattern-based prediction</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div class="forecast-banner">
            📌 <b>Forecast Mode:</b> The selected date is outside the dataset (Jan–Jun 2017).
            The model will predict using <b>historical traffic patterns</b> for the same
            hour and day-of-week from the dataset. Predicting <b>{time_window_minutes} minutes</b> ahead.
            Actual ground-truth values are <b>not available</b> for this date.
        </div>
        """, unsafe_allow_html=True)
    else:
        prediction_times = ", ".join([
            (selected_dt + timedelta(minutes=5*i)).strftime('%H:%M')
            for i in range(1, pred_steps + 1)
        ])
        st.markdown(
            f"🕐 **Base Time:** {time_display} &nbsp;→&nbsp; **Predicting:** {prediction_times} &nbsp;&nbsp;"
            f'<span class="mode-badge-hist">📊 FUTURE PREDICTION — Next {time_window_minutes} minutes</span>',
            unsafe_allow_html=True
        )
        st.info(f"💡 **Using traffic from {(selected_dt - timedelta(hours=2)).strftime('%H:%M')} - {selected_dt.strftime('%H:%M')} to predict {prediction_times}**")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================
# RUN INFERENCE
# =============================================================
if view_mode == "Current Traffic (Actual)":
    if not in_dataset:
        st.error("⚠️ Current Traffic mode only works for dates within the dataset (Jan 1 - Jun 30, 2017). Switch to Future Prediction mode or select a date in range.")
        st.stop()
    
    traffic_sequence      = all_data[window_idx : window_idx + pred_steps, :, 0]
    traffic_sequence_real = traffic_sequence * std[0] + mean[0]
    pred_all  = traffic_sequence_real
    true_all  = traffic_sequence_real
    pred_real = pred_all[:, sensor_idx]
    true_real = true_all[:, sensor_idx]
    pred_avg  = float(pred_real.mean())
    true_avg  = float(true_real.mean())
    is_congested = pred_avg < CONGESTION_THRESHOLD
    match_label  = "Showing actual traffic from dataset (not a prediction)"

elif is_forecast_mode:
    matching_windows = find_matching_windows(all_timestamps, all_data, selected_dt, n=3)
    if matching_windows:
        preds_list = []
        for w in matching_windows:
            slice_data = all_data[w - 24 : w].astype(np.float32)
            p = predict_all_sensors_from_data(model, slice_data, mean, std, num_nodes, pred_steps)
            preds_list.append(p)
        pred_all    = np.mean(preds_list, axis=0)
        true_all    = None
        match_label = (
            f"Based on {len(matching_windows)} historical "
            f"{'windows' if len(matching_windows)>1 else 'window'} "
            f"matching {selected_dt.strftime('%A %H:%M')} pattern"
        )
    else:
        st.error("No matching historical pattern found. Try a different time.")
        st.stop()
    pred_real    = pred_all[:, sensor_idx]
    pred_avg     = float(pred_real.mean())
    is_congested = pred_avg < CONGESTION_THRESHOLD
    true_real    = None
    true_avg     = None

else:
    pred_all, true_all = predict_all_sensors_from_all_data(
        model, all_data, mean, std, window_idx, num_nodes, device, pred_steps
    )
    match_label  = None
    pred_real    = pred_all[:, sensor_idx]
    pred_avg     = float(pred_real.mean())
    is_congested = pred_avg < CONGESTION_THRESHOLD
    if true_all is not None:
        true_real = true_all[:, sensor_idx]
        true_avg  = float(true_real.mean())
    else:
        true_real = None
        true_avg  = None

# =============================================================
# ROW 1: KPI METRICS
# CHANGED: Removed MAE and RMSE — now only 3 metrics shown
# =============================================================
if view_mode == "Current Traffic (Actual)":
    st.subheader(f"📊 Current Traffic — Sensor {selected_sensor_name}")
else:
    st.subheader(f"📊 Prediction — Sensor {selected_sensor_name}")

# Only 3 columns now (removed MAE and RMSE)
col1, col2, col3 = st.columns(3)

if view_mode == "Current Traffic (Actual)":
    col1.metric("Current Speed (avg)", f"{pred_avg:.1f} mph")
    col2.metric("Actual Speed (avg)",  f"{true_avg:.1f} mph" if true_avg is not None else "N/A")
    if is_congested:
        col3.metric("Status", "🔴 CONGESTED")
    else:
        col3.metric("Status", "🟢 FREE FLOW")
else:
    col1.metric("Predicted Speed (avg)", f"{pred_avg:.1f} mph",
                delta=f"{pred_avg - true_avg:+.1f} vs actual" if true_avg is not None else None)
    col2.metric("Actual Speed (avg)",    f"{true_avg:.1f} mph" if true_avg is not None else "N/A (future date)")
    if is_congested:
        col3.metric("Status", "🔴 CONGESTED")
    else:
        col3.metric("Status", "🟢 FREE FLOW")

if match_label:
    st.caption(f"ℹ️ {match_label}")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================
# ROW 2: CONGESTION ALERT + NEARBY ROUTE SUGGESTIONS
# =============================================================
st.subheader("🚨 Congestion Alert & Route Guidance")

pred_avgs = pred_all.mean(axis=0)

if is_congested:
    st.markdown(f"""
    <div class="alert-red">
        ⚠️ CONGESTION PREDICTED — Sensor {selected_sensor_name} &nbsp;·&nbsp; {time_display}<br>
        Predicted avg speed: <b>{pred_avg:.1f} mph</b> — below threshold of {CONGESTION_THRESHOLD} mph<br>
        ➡ Consider alternative sensors/routes listed below
    </div>
    """, unsafe_allow_html=True)

    nearby = get_nearby_sensors(sensor_idx, A_numpy, sensor_cols, top_k=8)
    if nearby:
        st.markdown("**🛣️ Nearby Sensors — Alternative Route Options:**")
        nearby_data = []
        for idx, name, weight in nearby:
            p_avg_n = float(pred_avgs[idx])
            t_avg_n = float(true_all[:, idx].mean()) if true_all is not None else None
            nearby_data.append((idx, name, weight, p_avg_n, t_avg_n))
        nearby_data.sort(key=lambda x: x[3], reverse=True)

        best_shown = 0
        for idx, name, weight, p_avg_n, t_avg_n in nearby_data:
            status      = "🟢 FREE FLOW — ✅ USE THIS ROUTE" if p_avg_n >= CONGESTION_THRESHOLD else "🔴 ALSO CONGESTED"
            conn_str    = "Strong" if weight > 0.3 else "Moderate" if weight > 0.1 else "Weak"
            actual_part = f"Actual: {t_avg_n:.1f} mph &nbsp;|&nbsp;" if t_avg_n is not None else ""
            if p_avg_n >= CONGESTION_THRESHOLD:
                best_shown += 1
            st.markdown(f"""
            <div class="route-card">
                <b>Sensor {name}</b> &nbsp;|&nbsp;
                Predicted: <b>{p_avg_n:.1f} mph</b> &nbsp;|&nbsp;
                {actual_part}Connection: {conn_str} &nbsp;|&nbsp; {status}
            </div>
            """, unsafe_allow_html=True)

        if best_shown == 0:
            st.warning("⚠️ All nearby sensors are also congested. No clear alternative route at this time.")
    else:
        st.info("No adjacent sensors found in the graph for this sensor.")
else:
    st.markdown(f"""
    <div class="alert-green">
        ✅ TRAFFIC FLOWING FREELY — Sensor {selected_sensor_name} &nbsp;·&nbsp; {time_display}<br>
        Predicted avg speed: <b>{pred_avg:.1f} mph</b> — above threshold of {CONGESTION_THRESHOLD} mph<br>
        No rerouting needed.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================
# ROW 3: PREDICTION CHART
# CHANGED: Added X-axis label "Time Step" to the chart
# =============================================================
if view_mode == "Current Traffic (Actual)":
    st.subheader(f"📈 Current Traffic Conditions — Sensor {selected_sensor_name}")
    chart_title = f"Traffic Speed (Actual) — Sensor {selected_sensor_name} · {selected_dt.strftime('%d %b %Y %H:%M')}"
else:
    st.subheader(f"📈 Speed Forecast — Sensor {selected_sensor_name}")
    chart_title = f"Traffic Speed Forecast ({time_window_minutes} min) — Sensor {selected_sensor_name} · {selected_dt.strftime('%d %b %Y %H:%M')}"

step_times = [
    (selected_dt + timedelta(minutes=5*(i+1))).strftime("%H:%M")
    for i in range(pred_steps)
]

if view_mode == "Current Traffic (Actual)":
    steps = [f"{step_times[i]}" for i in range(pred_steps)]
else:
    steps = [f"t+{5*(i+1)}m\n{step_times[i]}" for i in range(pred_steps)]

fig, ax = plt.subplots(figsize=(max(10, pred_steps * 1.5), 4))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#f8f9fa')

x = range(pred_steps)

if view_mode == "Current Traffic (Actual)":
    ax.plot(x, pred_real, marker='o', linewidth=2.5,
            color='#1976d2', label='Actual Traffic', markersize=9)
else:
    if true_real is not None:
        ax.plot(x, true_real, marker='o', linewidth=2.5,
                color='#1976d2', label='Actual', markersize=9)
        ax.fill_between(x, true_real, pred_real, alpha=0.15, color='gray')
    ax.plot(x, pred_real, marker='s', linewidth=2.5,
            color='#d32f2f', linestyle='--', label='Predicted', markersize=9)

ax.axhline(y=CONGESTION_THRESHOLD, color='#ff4444',
           linestyle=':', linewidth=1.5, label=f'Threshold ({CONGESTION_THRESHOLD} mph)')

for i, p in enumerate(pred_real):
    ax.annotate(f"{p:.1f}", (i, p), textcoords="offset points",
                xytext=(0, 10), ha='center',
                color='#1976d2' if view_mode == "Current Traffic (Actual)" else '#d32f2f',
                fontsize=9, fontweight='bold')

if view_mode != "Current Traffic (Actual)" and true_real is not None:
    for i, t in enumerate(true_real):
        ax.annotate(f"{t:.1f}", (i, t), textcoords="offset points",
                    xytext=(0, -16), ha='center', color='#1976d2', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(steps, color='#000000', fontsize=9, fontweight='bold')

# ── CHANGED: Added X-axis label "Time Step" ──
ax.set_xlabel("Time Step", color='#000000', fontweight='bold', fontsize=11)
ax.set_ylabel("Speed (mph)", color='#000000', fontweight='bold')

ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#cccccc')
ax.spines['left'].set_color('#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#000000', fontsize=9)
ax.set_title(chart_title, color='#000000', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2, color='#cccccc')
plt.tight_layout()
st.pyplot(fig)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================
# ROW 4: ALL SENSORS OVERVIEW TABLE
# =============================================================
st.subheader("🗺️ All Sensors Overview")

true_avgs_arr = true_all.mean(axis=0) if true_all is not None else np.full(num_nodes, np.nan)

sensor_status = []
for i, name in enumerate(sensor_cols):
    p_avg = float(pred_avgs[i])
    t_avg = float(true_avgs_arr[i]) if true_all is not None else None
    row = {
        "Sensor":          name,
        "Predicted (mph)": round(p_avg, 1),
        "Actual (mph)":    round(t_avg, 1) if t_avg is not None else "N/A",
        "Error (mph)":     round(abs(p_avg - t_avg), 2) if t_avg is not None else "N/A",
        "Status":          "🔴 Congested" if p_avg < CONGESTION_THRESHOLD else "🟢 Free Flow"
    }
    sensor_status.append(row)

df_status = pd.DataFrame(sensor_status)
n_congested = (df_status["Status"] == "🔴 Congested").sum()
n_free      = (df_status["Status"] == "🟢 Free Flow").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total Sensors", num_nodes)
c2.metric("🔴 Congested",  n_congested)
c3.metric("🟢 Free Flow",  n_free)

filter_opt = st.selectbox("Filter table:", ["All", "Congested Only", "Free Flow Only"])
df_show = df_status.copy()
if filter_opt == "Congested Only":
    df_show = df_status[df_status["Status"] == "🔴 Congested"].copy()
elif filter_opt == "Free Flow Only":
    df_show = df_status[df_status["Status"] == "🟢 Free Flow"].copy()
df_show = df_show.reset_index(drop=True)

st.markdown(create_styled_table(df_show), unsafe_allow_html=True)

csv_data = df_status.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Export All Sensor Predictions as CSV",
    data=csv_data,
    file_name=f"pems_bay_{selected_dt.strftime('%Y%m%d_%H%M')}_{time_window_minutes}min.csv",
    mime="text/csv"
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================
# ROW 5: BAR CHART — TOP 40 MOST AT-RISK SENSORS
# =============================================================
st.subheader("📊 Top 40 Sensors — Sorted by Predicted Speed (Congestion Risk)")

sorted_indices = np.argsort(pred_avgs)[:40]
sorted_speeds  = pred_avgs[sorted_indices]
sorted_names   = [sensor_cols[i] for i in sorted_indices]
bar_colors     = ['#ff4b4b' if s < CONGESTION_THRESHOLD else '#00c853' for s in sorted_speeds]

fig2, ax2 = plt.subplots(figsize=(16, 4))
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#f8f9fa')

ax2.bar(sorted_names, sorted_speeds, color=bar_colors, edgecolor='none', width=0.6)
ax2.axhline(y=CONGESTION_THRESHOLD, color='#000000',
            linestyle='--', linewidth=1.2, label=f'Threshold ({CONGESTION_THRESHOLD} mph)')

ax2.set_ylabel("Predicted Speed (mph)", color='#000000', fontweight='bold')
ax2.tick_params(colors='#000000', axis='both')
ax2.set_xticklabels(sorted_names, rotation=60, ha='right', color='#000000', fontsize=7)
ax2.spines['bottom'].set_color('#cccccc')
ax2.spines['left'].set_color('#cccccc')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

red_patch   = mpatches.Patch(color='#ff4b4b', label=f'Congested (< {CONGESTION_THRESHOLD} mph)')
green_patch = mpatches.Patch(color='#00c853', label=f'Free Flow (≥ {CONGESTION_THRESHOLD} mph)')
ax2.legend(handles=[red_patch, green_patch],
           facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#000000', fontsize=9)
ax2.set_title("40 Most At-Risk Sensors (Lowest Predicted Speed)", color='#000000', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, color='#cccccc', axis='y')

plt.tight_layout()
st.pyplot(fig2)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(
    "Built with **MS-GWN-A** · PEMS-BAY Dataset · 325 sensors · 5-min intervals · "
    f"Congestion Threshold: {CONGESTION_THRESHOLD} mph"
)