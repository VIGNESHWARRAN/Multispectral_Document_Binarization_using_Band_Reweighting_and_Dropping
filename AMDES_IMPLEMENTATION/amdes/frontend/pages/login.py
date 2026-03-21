"""frontend/pages/login.py"""
import streamlit as st
from frontend.auth import get_login_url


def render():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
<div style="text-align:center;padding:48px 0 32px 0">
  <div style="font-size:3.5rem;margin-bottom:12px">📜</div>
  <div style="font-size:2rem;font-weight:800;color:#e6edf3;letter-spacing:-0.03em">AMDES</div>
  <div style="font-size:0.88rem;color:#8b949e;margin-top:6px;line-height:1.6">
    Advanced Multispectral Document Enhancement System<br>
    <span style="color:#58a6ff">Restoring historical manuscripts through AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Sign-in card
        st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:32px;margin-bottom:20px">
  <div style="font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:8px">Sign in to your account</div>
  <div style="font-size:0.85rem;color:#8b949e;margin-bottom:24px;line-height:1.6">
    Securely access the AMDES processing pipeline.<br>Supports email/password and Google OAuth via Auth0.
  </div>
""", unsafe_allow_html=True)

        login_url = get_login_url()
        st.markdown(f"""
<a href="{login_url}" target="_self" style="text-decoration:none;display:block">
  <div style="background:#238636;color:#ffffff;text-align:center;padding:12px 20px;
              border-radius:8px;font-size:0.95rem;font-weight:600;border:1px solid #2ea043;
              transition:background 0.2s;letter-spacing:0.01em">
    🔐 &nbsp; Continue with Auth0
  </div>
</a>
<div style="color:#6e7681;font-size:0.78rem;text-align:center;margin-top:12px">
  Protected by Auth0 · Session encrypted
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature pills
        st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px">
  <div style="font-size:0.78rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:16px">
    What AMDES does
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div style="background:#21262d;border-radius:8px;padding:12px;border:1px solid #30363d">
      <div style="color:#58a6ff;font-size:1.1rem;margin-bottom:4px">🔬</div>
      <div style="color:#e6edf3;font-size:0.82rem;font-weight:600">Spectral Unmixing</div>
      <div style="color:#8b949e;font-size:0.75rem;margin-top:2px">VCA · N-FINDR · NNLS</div>
    </div>
    <div style="background:#21262d;border-radius:8px;padding:12px;border:1px solid #30363d">
      <div style="color:#3fb950;font-size:1.1rem;margin-bottom:4px">〰️</div>
      <div style="color:#e6edf3;font-size:0.82rem;font-weight:600">Wavelet Enhancement</div>
      <div style="color:#8b949e;font-size:0.75rem;margin-top:2px">Haar · Daubechies · Bior</div>
    </div>
    <div style="background:#21262d;border-radius:8px;padding:12px;border:1px solid #30363d">
      <div style="color:#f78166;font-size:1.1rem;margin-bottom:4px">📊</div>
      <div style="color:#e6edf3;font-size:0.82rem;font-weight:600">Quality Metrics</div>
      <div style="color:#8b949e;font-size:0.75rem;margin-top:2px">PSNR · SSIM · CNR · CER</div>
    </div>
    <div style="background:#21262d;border-radius:8px;padding:12px;border:1px solid #30363d">
      <div style="color:#d2a8ff;font-size:1.1rem;margin-bottom:4px">🔤</div>
      <div style="color:#e6edf3;font-size:0.82rem;font-weight:600">OCR Extraction</div>
      <div style="color:#8b949e;font-size:0.75rem;margin-top:2px">Tesseract 5 · Multi-lang</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)