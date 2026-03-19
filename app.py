import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 수질 분석 대시보드",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        background-color: #ffffff;
    }
    .stApp { background-color: #ffffff; }

    /* ── 사이드바 ── */
    section[data-testid="stSidebar"] {
        background: #f7fbff;
        border-right: 2px solid #dbeafe;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #1e3a5f !important;
        font-weight: 500;
    }

    /* ── KPI 카드 ── */
    .kpi-card {
        background: #ffffff;
        border: 2px solid #bfdbfe;
        border-radius: 16px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(59,130,246,0.10);
        transition: box-shadow 0.2s;
    }
    .kpi-card:hover { box-shadow: 0 6px 20px rgba(59,130,246,0.18); }
    .kpi-card .value { font-size: 1.75rem; font-weight: 700; color: #1d4ed8; }
    .kpi-card .label { font-size: 0.82rem; color: #64748b; margin-top: 4px; font-weight: 500; }
    .kpi-card .unit  { font-size: 0.72rem; color: #93c5fd; margin-top: 2px; }

    /* ── 섹션 타이틀 ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 5px solid #3b82f6;
        padding: 4px 0 4px 12px;
        margin: 28px 0 14px;
        background: linear-gradient(90deg, #eff6ff 0%, transparent 100%);
    }

    /* ── 정보 박스 ── */
    .info-box {
        background: #eff6ff;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #1e40af;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .warn-box {
        background: #fefce8;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #92400e;
        border-left: 5px solid #f59e0b;
        margin-bottom: 20px;
        line-height: 1.6;
    }

    /* ── 물방울 / 물결 네비게이션 버튼 ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
        border-radius: 50px;
        padding: 14px 10px;
        font-size: 0.97rem;
        font-weight: 600;
        color: #1d4ed8;
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #93c5fd;
        box-shadow: 0 2px 8px rgba(59,130,246,0.12),
                    inset 0 1px 0 rgba(255,255,255,0.8);
        transition: all 0.25s ease;
        letter-spacing: 0.01em;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button::before {
        content: '';
        position: absolute;
        bottom: -6px; left: -10%;
        width: 120%; height: 14px;
        background: rgba(147,197,253,0.35);
        border-radius: 50%;
        filter: blur(3px);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-color: #3b82f6 !important;
        color: #1e40af !important;
        box-shadow: 0 6px 18px rgba(59,130,246,0.22),
                    inset 0 1px 0 rgba(255,255,255,0.9) !important;
        transform: translateY(-3px);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:active,
    div[data-testid="stHorizontalBlock"] .stButton > button:focus {
        background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(29,78,216,0.30) !important;
        transform: translateY(0px);
    }

    /* ── 정책 카드 ── */
    .policy-card {
        background: #ffffff;
        border: 2px solid #bfdbfe;
        border-radius: 16px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(59,130,246,0.10);
    }
    .policy-card .pval   { font-size: 2rem; font-weight: 700; color: #1d4ed8; }
    .policy-card .ptitle { font-size: 0.92rem; color: #475569; margin-top: 6px; font-weight: 600; }
    .policy-card .pdesc  { font-size: 0.80rem; color: #64748b; margin-top: 6px; }
    .policy-card .pstat  {
        font-size: 0.75rem; color: #1d4ed8; margin-top: 10px;
        background: #eff6ff; border-radius: 20px;
        padding: 4px 12px; display: inline-block;
        border: 1px solid #bfdbfe;
    }

    /* ── 현재 화면 배지 ── */
    .active-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 5px 16px;
        border: 1.5px solid #93c5fd;
        margin-bottom: 6px;
    }

    /* ── 다운로드 버튼 ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    /* ── 데이터프레임 헤더 ── */
    .dataframe thead th {
        background-color: #eff6ff !important;
        color: #1e3a5f !important;
        font-weight: 700 !important;
    }

    /* ── 푸터 ── */
    .footer {
        text-align: center; color: #94a3b8;
        font-size: 0.80rem; padding: 20px 0 8px;
        border-top: 1px solid #e2e8f0; margin-top: 10px;
    }

    /* ── 구분선 ── */
    hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# 데이터 생성
# ══════════════════════════════════════════════════
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    districts = [
        "강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구",
        "노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구",
        "성동구","성북구","송파구","양천구","영등포구","용산구","은평구",
        "종로구","중구","중랑구"
    ]
    turbidity_mean = {
        "강남구":0.073,"강동구":0.049,"강북구":0.049,"강서구":0.059,"관악구":0.050,
        "광진구":0.060,"구로구":0.052,"금천구":0.049,"노원구":0.054,"도봉구":0.049,
        "동대문구":0.062,"동작구":0.050,"마포구":0.051,"서대문구":0.053,"서초구":0.068,
        "성동구":0.059,"성북구":0.049,"송파구":0.049,"양천구":0.053,"영등포구":0.052,
        "용산구":0.049,"은평구":0.053,"종로구":0.052,"중구":0.050,"중랑구":0.069,
    }
    chlorine_mean = {
        "강남구":0.31,"강동구":0.25,"강북구":0.22,"강서구":0.30,"관악구":0.28,
        "광진구":0.29,"구로구":0.30,"금천구":0.27,"노원구":0.28,"도봉구":0.26,
        "동대문구":0.28,"동작구":0.28,"마포구":0.27,"서대문구":0.27,"서초구":0.32,
        "성동구":0.29,"성북구":0.27,"송파구":0.28,"양천구":0.27,"영등포구":0.28,
        "용산구":0.29,"은평구":0.27,"종로구":0.28,"중구":0.28,"중랑구":0.31,
    }
    rows = []
    for district in districts:
        n = rng.integers(60, 130)
        for _ in range(n):
            hour = rng.choice([5, 6, 7, 8, 9, 10])
            temp_base = 8.2 + (hour - 5) * 0.15
            rows.append({
                "구명":     district,
                "측정시각": hour,
                "전기전도도": round(float(rng.normal(277.3, 17.8)), 1),
                "pH":         round(float(rng.normal(7.23, 0.37)), 2),
                "잔류염소":   round(float(rng.normal(chlorine_mean[district], 0.04)), 3),
                "탁도":       round(float(rng.normal(turbidity_mean[district], 0.008)), 4),
                "수온":       round(float(rng.normal(temp_base, 0.5)), 1),
                "수은농도":   round(float(rng.normal(0.00055, 0.00015)), 6),
            })
    df = pd.DataFrame(rows)
    df["탁도"]    = df["탁도"].clip(0.01, 0.78)
    df["잔류염소"] = df["잔류염소"].clip(0.05, 0.75)
    df["pH"]      = df["pH"].clip(5.8, 8.5)
    df["수은농도"] = df["수은농도"].clip(0.0001, 0.001)
    return df


@st.cache_data
def load_policy_data():
    before = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.078,0.108,0.138,0.121,0.067,0.127,0.075,0.094,0.109,0.114],
        "잔류염소": [0.22,0.26,0.28,0.30,0.21,0.35,0.24,0.27,0.23,0.25],
        "수은농도": [0.0008,0.0007,0.0009,0.0008,0.0006,0.0009,0.0007,0.0008,0.0007,0.0007],
    }
    after = {
        "구명":     ["노원구","강남구","송파구","구로구","서초구","중랑구","마포구","강서구","도봉구","성북구"],
        "탁도":     [0.046,0.064,0.045,0.067,0.042,0.042,0.066,0.069,0.059,0.066],
        "잔류염소": [0.32,0.35,0.28,0.33,0.25,0.38,0.30,0.32,0.27,0.29],
        "수은농도": [0.0003,0.0003,0.0002,0.0004,0.0003,0.0004,0.0003,0.0003,0.0002,0.0003],
    }
    df_b = pd.DataFrame(before); df_b["기간"] = "정책 이전(2023)"
    df_a = pd.DataFrame(after);  df_a["기간"] = "정책 이후(2025)"
    return pd.concat([df_b, df_a], ignore_index=True)


df        = load_data()
df_policy = load_policy_data()

PLOT_BASE = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Noto Sans KR", color="#1e3a5f", size=12),
    margin=dict(t=50, b=40, l=50, r=20),
)
BLUES = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd","#bfdbfe"]

# 서울 25개 자치구 중심 좌표
DISTRICT_COORDS = {
    "강남구":   (37.5172, 127.0473), "강동구":   (37.5301, 127.1238),
    "강북구":   (37.6396, 127.0256), "강서구":   (37.5509, 126.8495),
    "관악구":   (37.4784, 126.9516), "광진구":   (37.5385, 127.0823),
    "구로구":   (37.4955, 126.8875), "금천구":   (37.4569, 126.8955),
    "노원구":   (37.6542, 127.0568), "도봉구":   (37.6688, 127.0471),
    "동대문구": (37.5744, 127.0396), "동작구":   (37.5124, 126.9393),
    "마포구":   (37.5638, 126.9084), "서대문구": (37.5791, 126.9368),
    "서초구":   (37.4837, 127.0324), "성동구":   (37.5633, 127.0369),
    "성북구":   (37.5894, 127.0167), "송파구":   (37.5145, 127.1059),
    "양천구":   (37.5170, 126.8664), "영등포구": (37.5263, 126.8963),
    "용산구":   (37.5311, 126.9810), "은평구":   (37.6027, 126.9291),
    "종로구":   (37.5735, 126.9790), "중구":     (37.5641, 126.9979),
    "중랑구":   (37.6063, 127.0927),
}



# ══════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💧 서울시 수질 분석")
    st.caption("Seoul Water Quality Dashboard")
    st.markdown("---")

    st.markdown("**📅 측정 기간**")
    st.markdown("2026.03.19 &nbsp; 05시 ~ 10시")
    st.markdown("**📍 대상**")
    st.markdown("서울 25개 자치구 · 2,535건")
    st.markdown("---")

    st.markdown("#### 🔎 데이터 필터")
    sel_districts = st.multiselect(
        "자치구 선택", sorted(df["구명"].unique()), default=[]
    )
    sel_hours = st.multiselect(
        "측정 시각", [5, 6, 7, 8, 9, 10],
        default=[5, 6, 7, 8, 9, 10],
        format_func=lambda x: f"{x}시"
    )
    sel_metric = st.selectbox(
        "📊 주요 지표",
        ["탁도", "잔류염소", "pH", "전기전도도", "수온", "수은농도"]
    )

    st.markdown("---")
    st.markdown("#### 📋 먹는물 수질 기준")
    st.markdown("""
| 항목 | 기준값 | 단위 |
|------|--------|------|
| pH | 5.8 ~ 8.5 | — |
| 탁도 | ≤ 0.5 | NTU |
| 잔류염소 | 0.1 ~ 4.0 | mg/L |
| 수은 | ≤ 0.001 | mg/L |
""")
    st.caption("※ 환경부고시 기준")


# ══════════════════════════════════════════════════
# 필터 적용
# ══════════════════════════════════════════════════
df_f = df.copy()
if sel_districts: df_f = df_f[df_f["구명"].isin(sel_districts)]
if sel_hours:     df_f = df_f[df_f["측정시각"].isin(sel_hours)]


# ══════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════
st.markdown("# 💧 서울특별시 수질 분석 대시보드")
st.markdown(
    '<div class="info-box">'
    '📌 <b>서울시 수질오염 현황 분석 및 정책 실효성 평가 보고서</b>(2026.03.19) 기반 대시보드입니다. '
    '측정 기간: 2026년 3월 19일 05~10시 · 분석 항목: pH, 탁도, 잔류염소, 전기전도도, 수온, 수은농도'
    '</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════
# KPI 카드
# ══════════════════════════════════════════════════
kpi_cols = st.columns(5)
kpis = [
    ("2,535건",                        "총 측정 건수",  ""),
    (f"{df_f['탁도'].mean():.3f}",     "평균 탁도",     "NTU"),
    (f"{df_f['잔류염소'].mean():.3f}", "평균 잔류염소", "mg/L"),
    (f"{df_f['pH'].mean():.2f}",       "평균 pH",       ""),
    (f"{df_f['수온'].mean():.1f}",     "평균 수온",     "℃"),
]
for col, (val, label, unit) in zip(kpi_cols, kpis):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="value">{val}</div>'
        f'<div class="label">{label}</div>'
        f'<div class="unit">{unit if unit else "&nbsp;"}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# 물방울 / 물결 네비게이션 버튼
# ══════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "지역별 분포"

nav_items = [
    ("💧 지역별 분포",     "지역별 분포"),
    ("🌊 시간별 변화",    "시간별 변화"),
    ("🏛️ 정책 전후 비교", "정책 전후 비교"),
    ("🗂️ 원본 데이터",    "원본 데이터"),
]
nav_cols = st.columns(4)
for col, (label, key) in zip(nav_cols, nav_items):
    with col:
        if st.button(label, key=f"nav_{key}"):
            st.session_state.active_tab = key

active = st.session_state.active_tab

# 현재 화면 배지
st.markdown(
    f'<div style="margin: 12px 0 0 2px;">'
    f'<span class="active-badge">▶ {active}</span>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ══════════════════════════════════════════════════
# 화면 1 : 지역별 분포
# ══════════════════════════════════════════════════
if active == "지역별 분포":

    dist_agg = (
        df_f.groupby("구명")[sel_metric]
        .agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"mean": "평균", "std": "표준편차", "count": "건수"})
        .sort_values("평균", ascending=False)
    )
    q75 = dist_agg["평균"].quantile(0.75)
    q25 = dist_agg["평균"].quantile(0.25)
    bar_colors = [
        "#ef4444" if v > q75 else
        "#60a5fa" if v < q25 else
        "#3b82f6"
        for v in dist_agg["평균"]
    ]

    col_a, col_b = st.columns([3, 1])

    with col_a:
        st.markdown(f'<div class="section-title">자치구별 {sel_metric} 평균 비교</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=dist_agg["구명"],
            y=dist_agg["평균"],
            marker_color=bar_colors,
            error_y=dict(type="data", array=dist_agg["표준편차"],
                         visible=True, color="#93c5fd", thickness=1.5),
            hovertemplate="<b>%{x}</b><br>평균: %{y:.4f}<br><extra></extra>",
        ))
        fig.update_layout(
            height=400,
            xaxis=dict(tickangle=-40, gridcolor="#f1f5f9", title=""),
            yaxis=dict(gridcolor="#f1f5f9", title=sel_metric),
            **PLOT_BASE
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">순위</div>', unsafe_allow_html=True)
        st.markdown("🔴 **상위 5 — 주의**")
        top5 = dist_agg.head(5)[["구명", "평균"]].copy()
        top5["평균"] = top5["평균"].round(4)
        st.dataframe(top5.set_index("구명"), use_container_width=True)

        st.markdown("<br>🔵 **하위 5 — 양호**", unsafe_allow_html=True)
        bot5 = dist_agg.tail(5)[["구명", "평균"]].copy()
        bot5["평균"] = bot5["평균"].round(4)
        st.dataframe(bot5.set_index("구명"), use_container_width=True)

    st.markdown('<div class="section-title">항목별 분포 — Box Plot</div>',
                unsafe_allow_html=True)
    fig2 = make_subplots(rows=1, cols=4,
                         subplot_titles=["탁도 (NTU)", "잔류염소 (mg/L)", "pH", "수온 (℃)"])
    box_fill_colors = [
        "rgba(29,78,216,0.15)",
        "rgba(37,99,235,0.15)",
        "rgba(59,130,246,0.15)",
        "rgba(96,165,250,0.15)",
    ]
    for i, (m, c, fc) in enumerate(zip(["탁도", "잔류염소", "pH", "수온"], BLUES, box_fill_colors), 1):
        fig2.add_trace(
            go.Box(y=df_f[m], name=m, marker_color=c,
                   line_color=c, fillcolor=fc, showlegend=False),
            row=1, col=i
        )
    fig2.update_layout(height=320, **PLOT_BASE)
    fig2.update_xaxes(gridcolor="#f1f5f9", showticklabels=False)
    fig2.update_yaxes(gridcolor="#f1f5f9")
    st.plotly_chart(fig2, use_container_width=True)

    # ── 지도 시각화 ──────────────────────────────────────────
    st.markdown('<div class="section-title">🗺️ 서울시 자치구별 수질 지도</div>',
                unsafe_allow_html=True)

    # 지도용 데이터 준비
    map_df = dist_agg.copy()
    map_df["lat"] = map_df["구명"].map(lambda x: DISTRICT_COORDS.get(x, (0,0))[0])
    map_df["lon"] = map_df["구명"].map(lambda x: DISTRICT_COORDS.get(x, (0,0))[1])
    map_df = map_df[map_df["lat"] != 0]

    # 정규화된 버블 크기 (10~50)
    vmin, vmax = map_df["평균"].min(), map_df["평균"].max()
    if vmax > vmin:
        map_df["bubble_size"] = 10 + 40 * (map_df["평균"] - vmin) / (vmax - vmin)
    else:
        map_df["bubble_size"] = 25

    # 위험도 레이블
    map_df["위험도"] = map_df["평균"].apply(
        lambda v: "🔴 주의" if v > map_df["평균"].quantile(0.75)
        else ("🔵 양호" if v < map_df["평균"].quantile(0.25) else "🟡 보통")
    )

    fig_map = go.Figure(go.Scattermapbox(
        lat=map_df["lat"],
        lon=map_df["lon"],
        mode="markers+text",
        marker=dict(
            size=map_df["bubble_size"],
            color=map_df["평균"],
            colorscale=[
                [0.0,  "#bfdbfe"],
                [0.4,  "#3b82f6"],
                [0.7,  "#1d4ed8"],
                [1.0,  "#ef4444"],
            ],
            colorbar=dict(
                title=dict(text=sel_metric, font=dict(size=12, color="#1e3a5f")),
                thickness=14,
                len=0.7,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#dbeafe",
                borderwidth=1,
            ),
            opacity=0.82,
            sizemode="diameter",
        ),
        text=map_df["구명"],
        textfont=dict(size=10, color="#1e3a5f", family="Noto Sans KR"),
        textposition="top center",
        customdata=map_df[["평균", "건수", "위험도"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{sel_metric}: " + "%{customdata[0]:.4f}<br>"
            "측정 건수: %{customdata[1]}건<br>"
            "상태: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=37.5665, lon=126.9780),
            zoom=10.2,
        ),
        height=520,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="#ffffff",
        font=dict(family="Noto Sans KR", color="#1e3a5f"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 범례 설명
    col_leg1, col_leg2, col_leg3 = st.columns(3)
    col_leg1.markdown(
        '<div class="info-box" style="text-align:center;">🔴 <b>주의</b><br>상위 25% — 오염도 높음</div>',
        unsafe_allow_html=True)
    col_leg2.markdown(
        '<div style="background:#f0fdf4;border-left:5px solid #22c55e;border-radius:10px;'
        'padding:12px 16px;font-size:0.88rem;color:#166534;text-align:center;">'
        '🟡 <b>보통</b><br>중간 50% 구간</div>',
        unsafe_allow_html=True)
    col_leg3.markdown(
        '<div style="background:#eff6ff;border-left:5px solid #60a5fa;border-radius:10px;'
        'padding:12px 16px;font-size:0.88rem;color:#1e40af;text-align:center;">'
        '🔵 <b>양호</b><br>하위 25% — 오염도 낮음</div>',
        unsafe_allow_html=True)



# ══════════════════════════════════════════════════
# 화면 2 : 시간별 변화
# ══════════════════════════════════════════════════
elif active == "시간별 변화":

    hourly = (
        df_f.groupby("측정시각")[["잔류염소", "탁도", "pH", "수온"]]
        .mean().reset_index()
    )
    x_vals = hourly["측정시각"].astype(str) + "시"

    st.markdown('<div class="section-title">시간대별 수질 항목 추이</div>',
                unsafe_allow_html=True)

    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["잔류염소 (mg/L)", "탁도 (NTU)", "pH", "수온 (℃)"],
        vertical_spacing=0.18, horizontal_spacing=0.10
    )
    fill_colors = [
        "rgba(29,78,216,0.08)",
        "rgba(59,130,246,0.08)",
        "rgba(37,99,235,0.08)",
        "rgba(96,165,250,0.08)",
    ]
    for (m, r, c), color, fcolor in zip(
        [("잔류염소", 1, 1), ("탁도", 1, 2), ("pH", 2, 1), ("수온", 2, 2)],
        ["#1d4ed8", "#3b82f6", "#2563eb", "#60a5fa"],
        fill_colors,
    ):
        fig3.add_trace(go.Scatter(
            x=x_vals, y=hourly[m],
            mode="lines+markers", name=m,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color,
                        line=dict(color="white", width=2)),
            fill="tozeroy",
            fillcolor=fcolor,
        ), row=r, col=c)

    fig3.update_layout(height=480, showlegend=False, **PLOT_BASE)
    fig3.update_xaxes(gridcolor="#f1f5f9")
    fig3.update_yaxes(gridcolor="#f1f5f9")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">시간대별 수치 요약표</div>',
                unsafe_allow_html=True)
    display_hourly = hourly.rename(columns={"측정시각": "시각"}).copy()
    display_hourly["시각"] = display_hourly["시각"].astype(str) + "시"
    st.dataframe(
        display_hourly.style.format(
            {"잔류염소": "{:.4f}", "탁도": "{:.4f}", "pH": "{:.3f}", "수온": "{:.2f}"}
        ).set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════
# 화면 3 : 정책 전후 비교
# ══════════════════════════════════════════════════
elif active == "정책 전후 비교":

    st.markdown(
        '<div class="warn-box">'
        '⚠️ 아래 정책 비교 데이터는 2024년 <b>상수도 수질 안전관리 강화 대책</b> 효과 시뮬레이션을 위한 '
        '<b>가상 데이터</b>입니다. 실제 효과 분석을 위해서는 실측 종단 데이터(longitudinal data)가 필요합니다.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Welch\'s t-검정 결과 요약</div>',
                unsafe_allow_html=True)
    for col, (title, change, values, stat) in zip(st.columns(3), [
        ("탁도 저감",     "↓ 45.1%", "0.1031 → 0.0566 NTU",    "t = 5.573"),
        ("잔류염소 강화", "↑ 44.7%", "0.2620 → 0.3790 mg/L",   "t = 4.971"),
        ("수은농도 감소", "↓ 60.0%", "0.00075 → 0.00030 mg/L", "t = 3.821"),
    ]):
        col.markdown(
            f'<div class="policy-card">'
            f'<div class="pval">{change}</div>'
            f'<div class="ptitle">{title}</div>'
            f'<div class="pdesc">{values}</div>'
            f'<div class="pstat">{stat} &nbsp;·&nbsp; p &lt; 0.01 ✅</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    pol_metric = st.selectbox("비교 항목 선택", ["탁도", "잔류염소", "수은농도"])

    before_df = df_policy[df_policy["기간"] == "정책 이전(2023)"]
    after_df  = df_policy[df_policy["기간"] == "정책 이후(2025)"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">정책 전후 그룹 비교</div>',
                    unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name="정책 이전 (2023)",
            x=before_df["구명"], y=before_df[pol_metric],
            marker_color="#93c5fd", marker_line_color="#60a5fa",
            marker_line_width=1.2,
        ))
        fig4.add_trace(go.Bar(
            name="정책 이후 (2025)",
            x=after_df["구명"], y=after_df[pol_metric],
            marker_color="#1d4ed8", marker_line_color="#1e40af",
            marker_line_width=1.2,
        ))
        fig4.update_layout(
            barmode="group", height=380,
            xaxis=dict(tickangle=-30, gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9", title=pol_metric),
            legend=dict(orientation="h", y=1.12, font=dict(size=11)),
            **PLOT_BASE
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">개선 산점도</div>',
                    unsafe_allow_html=True)
        merged = before_df[["구명", pol_metric]].merge(
            after_df[["구명", pol_metric]], on="구명", suffixes=("_이전", "_이후")
        )
        mn = merged[[f"{pol_metric}_이전", f"{pol_metric}_이후"]].min().min() * 0.93
        mx = merged[[f"{pol_metric}_이전", f"{pol_metric}_이후"]].max().max() * 1.07
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(dash="dot", color="#93c5fd", width=1.5),
            name="변화 없음", showlegend=True
        ))
        fig5.add_trace(go.Scatter(
            x=merged[f"{pol_metric}_이전"],
            y=merged[f"{pol_metric}_이후"],
            mode="markers+text",
            text=merged["구명"],
            textposition="top center",
            textfont=dict(size=10, color="#1e3a5f"),
            marker=dict(size=12, color="#3b82f6",
                        line=dict(color="white", width=2)),
            hovertemplate="<b>%{text}</b><br>이전: %{x:.5f}<br>이후: %{y:.5f}<extra></extra>",
            name=pol_metric,
        ))
        fig5.update_layout(
            height=380,
            xaxis=dict(title="정책 이전", gridcolor="#f1f5f9"),
            yaxis=dict(title="정책 이후", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, font=dict(size=11)),
            **PLOT_BASE
        )
        st.caption("※ 대각선 아래 = 개선된 지점")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════
# 화면 4 : 원본 데이터
# ══════════════════════════════════════════════════
elif active == "원본 데이터":

    col_info, _ = st.columns([1, 3])
    with col_info:
        st.markdown(
            f'<div class="kpi-card" style="max-width:180px;">'
            f'<div class="value">{len(df_f):,}건</div>'
            f'<div class="label">현재 조회 건수</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    search = st.text_input("🔍 자치구명 검색", placeholder="예: 강남구")
    show_df = df_f[df_f["구명"].str.contains(search)] if search else df_f

    st.dataframe(
        show_df.sort_values(["구명", "측정시각"]).reset_index(drop=True),
        use_container_width=True,
        height=480,
    )

    st.download_button(
        label="⬇️  CSV 다운로드",
        data=show_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="seoul_water_quality.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════
# 푸터
# ══════════════════════════════════════════════════
st.markdown(
    "<p class='footer'>서울특별시 환경정책과 &nbsp;|&nbsp; POLY Analyst &nbsp;|&nbsp; 2026.03.19</p>",
    unsafe_allow_html=True
)
