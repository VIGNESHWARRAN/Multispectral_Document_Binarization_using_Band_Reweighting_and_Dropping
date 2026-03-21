"""frontend/pages/dashboard.py"""
import streamlit as st
import requests
from config.settings import BACKEND_URL
from frontend.auth import get_current_user


def _check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


def render():
    user = get_current_user()
    name = user.get("name", "Researcher") if user else "Researcher"

    st.markdown(f"""
<div class="amdes-title">
  <h2>👋 Welcome back, {name.split()[0]}</h2>
  <p>AMDES Dashboard · Advanced Multispectral Document Enhancement System</p>
</div>
""", unsafe_allow_html=True)

    # Status row
    backend_ok, backend_info = _check_backend()
    has_upload = bool(st.session_state.get("uploaded_image"))
    has_result = bool(st.session_state.get("binarized_image"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Backend", "🟢 Online" if backend_ok else "🔴 Offline")
    with c2:
        model = "Local Model" if backend_info.get("local_model_loaded") else "HF API"
        st.metric("Active Model", model if backend_ok else "—")
    with c3:
        st.metric("Session Upload", "✅ Ready" if has_upload else "None")
    with c4:
        st.metric("Binarization", "✅ Done" if has_result else "Pending")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline overview
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Processing Pipeline")
        steps = [
            ("🔐", "Authentication",   "Auth0 OAuth2 / Email login",                "✅"),
            ("📤", "Upload",           "PNG · JPG · TIFF · GeoTIFF · HDF5",         "✅"),
            ("🔬", "Spectral Unmixing","VCA · N-FINDR · NNLS — numpy/scipy",        "✅"),
            ("〰️","Wavelet Enhancement","Haar · Daubechies · Bior — PyWavelets",    "✅"),
            ("🤖", "CNN Binarization", "Local model / HuggingFace / Otsu fallback", "✅"),
            ("📊", "Quality Metrics",  "PSNR · SSIM · CNR · CER · WER — live",     "✅"),
            ("🔤", "OCR Extraction",   "Tesseract 5.0+ · 8 languages",              "✅"),
            ("💾", "Export",           "PNG · JPEG · TIFF · GeoTIFF · PDF · TXT",   "✅"),
        ]
        for icon, name_, desc, status in steps:
            color = {"✅": "#3fb950", "⚡ Partial": "#d29922", "🔜 Planned": "#8b949e"}
            badge_bg = {"✅": "#0f2a0f", "⚡ Partial": "#2a1f00", "🔜 Planned": "#21262d"}
            s_color = color.get(status, "#8b949e")
            b_bg = badge_bg.get(status, "#21262d")
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #21262d">
  <span style="font-size:1.1rem;width:24px;text-align:center">{icon}</span>
  <div style="flex:1">
    <div style="color:#e6edf3;font-size:0.88rem;font-weight:600">{name_}</div>
    <div style="color:#6e7681;font-size:0.78rem;margin-top:1px">{desc}</div>
  </div>
  <span style="background:{b_bg};color:{s_color};font-size:0.72rem;font-weight:600;
  padding:3px 10px;border-radius:20px;border:1px solid {s_color}33;white-space:nowrap">{status}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        # Supported formats
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Supported Formats")
        formats = [
            ("PNG / JPG / TIFF", "Standard document images"),
            ("GeoTIFF", "Georeferenced multispectral"),
            ("HDF5", "Hyperspectral datacubes"),
            ("JPEG2000", "Archival compressed format"),
        ]
        for fmt, desc in formats:
            st.markdown(f"""
<div style="padding:8px 0;border-bottom:1px solid #21262d">
  <div style="color:#58a6ff;font-size:0.85rem;font-weight:600">{fmt}</div>
  <div style="color:#6e7681;font-size:0.78rem">{desc}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Metrics reference
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Quality Metrics")
        metrics = [
            ("PSNR", "Peak Signal-to-Noise Ratio"),
            ("SSIM", "Structural Similarity Index"),
            ("CNR",  "Contrast-to-Noise Ratio"),
            ("CER",  "Character Error Rate"),
            ("WER",  "Word Error Rate"),
            ("SAM",  "Spectral Angle Mapper"),
        ]
        for abbr, full in metrics:
            st.markdown(f"""
<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #21262d;align-items:center">
  <code style="background:#1f6feb22;color:#58a6ff;border-radius:4px;padding:2px 7px;
  font-size:0.78rem;font-weight:700;border:1px solid #1f6feb44;white-space:nowrap">{abbr}</code>
  <span style="color:#8b949e;font-size:0.8rem">{full}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick nav
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Quick Start")
        st.markdown('<p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">Upload a manuscript image to begin the enhancement pipeline.</p>', unsafe_allow_html=True)
        if st.button("📤 Go to Upload", use_container_width=True):
            st.session_state["_nav"] = "📤  Upload & Process"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)