"""frontend/components/styles.py"""
import streamlit as st

def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
.stApp * { box-sizing: border-box; }

/* ── Global text override ─────────────────────────────────────────────────── */
.stApp p, .stApp li, .stApp span, .stApp div,
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #8b949e !important;
}
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #58a6ff !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #30363d !important;
    margin: 12px 0 !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: #8b949e !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
    border: 1px solid transparent !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #21262d !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] > p {
    color: #6e7681 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #21262d !important;
    color: #f85149 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #2d1f1e !important;
    border-color: #f85149 !important;
    color: #ff7b72 !important;
}

/* ── Page header banner ───────────────────────────────────────────────────── */
.amdes-title {
    background: linear-gradient(135deg, #0d419d 0%, #1158c7 50%, #388bfd 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid #1f6feb;
    box-shadow: 0 0 20px rgba(31,111,235,0.15);
}
.amdes-title h1, .amdes-title h2, .amdes-title h3 {
    color: #ffffff !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}
.amdes-title p, .amdes-title span {
    color: #a5c8ff !important;
    margin: 0 !important;
    font-size: 0.9rem !important;
}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.amdes-card {
    background: #161b22 !important;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    border: 1px solid #30363d;
}
.amdes-card h3 {
    color: #e6edf3 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin: 0 0 16px 0 !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #30363d !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #161b22 !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
    border: 1px solid #30363d !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tables ───────────────────────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; }
thead tr th {
    background: #21262d !important;
    color: #8b949e !important;
    padding: 10px 14px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #30363d !important;
}
tbody tr td {
    color: #c9d1d9 !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #21262d !important;
    font-size: 0.9rem !important;
}
tbody tr:hover td { background: #1c2128 !important; }

/* ── Primary buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    border-color: #3fb950 !important;
    box-shadow: 0 0 12px rgba(46,160,67,0.3) !important;
}

/* ── Download buttons ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: 1px solid #388bfd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #388bfd !important;
    box-shadow: 0 0 12px rgba(56,139,253,0.3) !important;
}

/* ── Alerts ───────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    background: #161b22 !important;
    border-radius: 8px !important;
    border-left-width: 3px !important;
}
div[data-testid="stAlert"] p { color: #c9d1d9 !important; }

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #388bfd !important;
}

/* ── Selectbox / dropdowns ────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #58a6ff !important;
    border-color: #58a6ff !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
details {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
details summary p {
    color: #c9d1d9 !important;
    font-weight: 500 !important;
}

/* ── Input fields ─────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1f6feb, #58a6ff) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Caption ──────────────────────────────────────────────────────────────── */
.stCaption p { color: #6e7681 !important; font-size: 0.8rem !important; }

/* ── Code blocks ──────────────────────────────────────────────────────────── */
code { background: #21262d !important; color: #79c0ff !important; border-radius: 4px; padding: 2px 6px; }
</style>
""", unsafe_allow_html=True)