"""frontend/pages/upload.py"""
import io
import streamlit as st
import requests
from PIL import Image

from config.settings import BACKEND_URL, MAX_UPLOAD_MB
from utils.spectral import run_spectral_unmixing
from utils.wavelet import enhance_wavelet, WAVELET_MAP


def render():
    st.markdown("""
<div class="amdes-title">
  <h2>📤 Upload &amp; Process</h2>
  <p>Upload a manuscript image · Configure the full enhancement pipeline · Run</p>
</div>
""", unsafe_allow_html=True)

    col_upload, col_params = st.columns([3, 2])

    with col_upload:
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Image Upload")
        st.markdown(f'<p style="color:#8b949e;font-size:0.83rem;margin-bottom:12px">Accepted: PNG · JPG · TIFF · Max {MAX_UPLOAD_MB} MB</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop image here", type=["png","jpg","jpeg","tif","tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            raw_bytes = uploaded.getvalue()
            size_mb   = len(raw_bytes) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(f"❌ File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB.")
            else:
                img  = Image.open(uploaded)
                w, h = img.size
                st.image(img, use_container_width=True, caption=uploaded.name)
                st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px">
  <div style="background:#21262d;border-radius:8px;padding:10px;border:1px solid #30363d;text-align:center">
    <div style="color:#6e7681;font-size:0.7rem;text-transform:uppercase">Format</div>
    <div style="color:#e6edf3;font-weight:600;font-size:0.9rem;margin-top:3px">{img.format or "—"}</div>
  </div>
  <div style="background:#21262d;border-radius:8px;padding:10px;border:1px solid #30363d;text-align:center">
    <div style="color:#6e7681;font-size:0.7rem;text-transform:uppercase">Dimensions</div>
    <div style="color:#e6edf3;font-weight:600;font-size:0.9rem;margin-top:3px">{w}×{h}</div>
  </div>
  <div style="background:#21262d;border-radius:8px;padding:10px;border:1px solid #30363d;text-align:center">
    <div style="color:#6e7681;font-size:0.7rem;text-transform:uppercase">Size</div>
    <div style="color:#e6edf3;font-weight:600;font-size:0.9rem;margin-top:3px">{size_mb:.2f} MB</div>
  </div>
  <div style="background:#21262d;border-radius:8px;padding:10px;border:1px solid #30363d;text-align:center">
    <div style="color:#6e7681;font-size:0.7rem;text-transform:uppercase">Mode</div>
    <div style="color:#e6edf3;font-weight:600;font-size:0.9rem;margin-top:3px">{img.mode}</div>
  </div>
</div>
""", unsafe_allow_html=True)
                st.session_state["uploaded_image"]   = raw_bytes
                st.session_state["uploaded_name"]    = uploaded.name
                st.session_state["uploaded_img_obj"] = img
        else:
            st.markdown("""
<div style="padding:40px 20px;text-align:center;color:#6e7681">
  <div style="font-size:2.5rem;margin-bottom:8px">🖼️</div>
  <div style="font-size:0.9rem">Drop your manuscript image above</div>
  <div style="font-size:0.78rem;margin-top:4px">PNG · JPG · TIFF</div>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Spectral unmixing results (shown after run) ──────────────────
        if st.session_state.get("spectral_result"):
            _show_spectral_results(st.session_state["spectral_result"])

        # ── Wavelet results (shown after run) ────────────────────────────
        if st.session_state.get("wavelet_result"):
            _show_wavelet_results(st.session_state["wavelet_result"])

    with col_params:
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Pipeline Configuration")

        # Preset
        st.markdown(_param_label("Document Preset"), unsafe_allow_html=True)
        preset = st.selectbox("Preset", [
            "Medieval Manuscript", "Modern Printed Book",
            "Ancient Papyrus", "Handwritten Document", "Custom",
        ], label_visibility="collapsed")
        _apply_preset(preset)

        # Spectral unmixing
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_section("🔬 Spectral Unmixing"), unsafe_allow_html=True)
        unmix_algo = st.selectbox("Unmixing Algorithm", [
            "VCA — Vertex Component Analysis",
            "N-FINDR",
            "NNLS — Non-Negative Least Squares",
        ], label_visibility="collapsed", key="sel_unmix")
        n_endmembers = st.slider("Endmembers (max 3 for RGB)", 2, 3,
                                 st.session_state.get("preset_endmembers", 3),
                                 label_visibility="visible")

        # Wavelet
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_section("〰️ Wavelet Enhancement"), unsafe_allow_html=True)
        wavelet = st.selectbox("Wavelet Family", list(WAVELET_MAP.keys()),
                               index=list(WAVELET_MAP.keys()).index(
                                   st.session_state.get("preset_wavelet","Haar")),
                               label_visibility="collapsed", key="sel_wav")
        levels = st.slider("Decomposition Levels", 1, 5,
                           st.session_state.get("preset_levels", 3),
                           label_visibility="visible")
        thresh_mode = st.selectbox("Threshold Method",
                                   ["Automatic", "Hard", "Soft", "Adaptive"],
                                   label_visibility="collapsed", key="sel_thresh")

        # Binarization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_section("🤖 CNN Binarization"), unsafe_allow_html=True)
        model_choice = st.selectbox("Model", [
            "Auto (Local → HF API → Otsu)",
            "HuggingFace API",
            "Otsu Thresholding",
        ], label_visibility="collapsed", key="sel_model")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("uploaded_image"):
            st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
            st.markdown("### Run Pipeline")

            # Config summary
            st.markdown(f"""
<div style="margin-bottom:14px">
  {"".join(f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #21262d"><span style="color:#8b949e;font-size:0.8rem">{k}</span><span style="color:#e6edf3;font-size:0.8rem;font-weight:500">{v}</span></div>'
    for k, v in [
      ("Preset", preset),
      ("Unmixing", unmix_algo.split("—")[0].strip()),
      ("Endmembers", str(n_endmembers)),
      ("Wavelet", f"{wavelet} L{levels}"),
      ("Threshold", thresh_mode),
    ])}
</div>
""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔬 Unmix", use_container_width=True, help="Run spectral unmixing"):
                    _run_spectral(unmix_algo, n_endmembers)
            with c2:
                if st.button("〰️ Enhance", use_container_width=True, help="Run wavelet enhancement"):
                    _run_wavelet(wavelet, levels, thresh_mode)
            with c3:
                if st.button("🚀 Binarize", use_container_width=True, help="Run CNN binarization"):
                    _run_binarization(uploaded)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
padding:20px;text-align:center">
  <div style="color:#6e7681;font-size:0.85rem">Upload an image to enable processing</div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _param_label(text):
    return f'<p style="color:#6e7681;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:4px">{text}</p>'

def _section(text):
    return f'<div style="color:#c9d1d9;font-size:0.85rem;font-weight:600;padding:6px 0 4px 0;border-bottom:1px solid #30363d;margin-bottom:8px">{text}</div>'

PRESET_CONFIGS = {
    "Medieval Manuscript":  {"endmembers": 4, "wavelet": "Daubechies db4", "levels": 4},
    "Modern Printed Book":  {"endmembers": 2, "wavelet": "Haar",           "levels": 2},
    "Ancient Papyrus":      {"endmembers": 5, "wavelet": "Daubechies db6", "levels": 4},
    "Handwritten Document": {"endmembers": 3, "wavelet": "Biorthogonal bior3.3", "levels": 3},
    "Custom":               {"endmembers": 3, "wavelet": "Haar",           "levels": 3},
}

def _apply_preset(preset):
    cfg = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["Custom"])
    st.session_state["preset_endmembers"] = cfg["endmembers"]
    st.session_state["preset_wavelet"]    = cfg["wavelet"]
    st.session_state["preset_levels"]     = cfg["levels"]


def _run_spectral(algo, n_endmembers):
    with st.spinner("Running spectral unmixing…"):
        try:
            img = st.session_state["uploaded_img_obj"]
            result = run_spectral_unmixing(img, algorithm=algo, n_endmembers=n_endmembers)
            st.session_state["spectral_result"] = result
            st.success(f"✅ {result['algorithm']} complete — {n_endmembers} endmembers extracted.")
        except Exception as e:
            st.error(f"❌ Spectral unmixing failed: {e}")


def _run_wavelet(wavelet, levels, thresh_mode):
    with st.spinner("Running wavelet enhancement…"):
        try:
            img = st.session_state["uploaded_img_obj"]
            result = enhance_wavelet(img, wavelet=wavelet, levels=levels,
                                     threshold_mode=thresh_mode)
            st.session_state["wavelet_result"] = result
            # Use enhanced image as the working image for binarization
            st.session_state["wavelet_enhanced_obj"] = result["enhanced"]
            st.success(f"✅ Wavelet enhancement done — {result['wavelet_used']} L{result['levels_used']} ({result['threshold_mode']})")
        except Exception as e:
            st.error(f"❌ Wavelet enhancement failed: {e}")


def _run_binarization(uploaded):
    fname = st.session_state.get("uploaded_name", "image.png")
    # Prefer wavelet-enhanced image if available
    working_img = st.session_state.get("wavelet_enhanced_obj",
                  st.session_state.get("uploaded_img_obj"))

    mime = "image/png"
    buf  = io.BytesIO()
    working_img.convert("RGB").save(buf, format="PNG")
    raw = buf.getvalue()

    with st.spinner("Running binarization…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/binarize",
                files={"file": (fname, raw, mime)},
                timeout=90,
            )
            if resp.status_code == 200:
                st.session_state["binarized_image"] = resp.content
                st.success("✅ Binarization complete! Go to **🔬 Results & Export**.")
                st.balloons()
            elif resp.status_code == 503:
                st.warning(f"⏳ {resp.json().get('detail','Model warming up. Retry in 20s.')}")
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"❌ Backend error ({resp.status_code}): {detail}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend. Run: `python -m uvicorn backend.main:app --port 8000`")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Try again.")
        except Exception as e:
            st.error(f"❌ {e}")


def _show_spectral_results(result):
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown(f"### 🔬 Spectral Unmixing — {result['algorithm']}")
    st.markdown(f'<p style="color:#8b949e;font-size:0.82rem">{result["n_endmembers"]} endmembers extracted · Ink layer identified</p>', unsafe_allow_html=True)

    cols = st.columns(result["n_endmembers"])
    for i, (col, abu_img) in enumerate(zip(cols, result["abundance_images"])):
        with col:
            is_ink = (i == result.get("ink_idx", 0))
            label  = "🖋️ Ink" if is_ink else f"Layer {i+1}"
            color  = "#3fb950" if is_ink else "#58a6ff"
            st.markdown(f'<p style="color:{color};font-size:0.78rem;font-weight:600;text-align:center">{label}</p>', unsafe_allow_html=True)
            st.image(abu_img, use_container_width=True)

    # Endmember spectra (RGB values)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:0.8rem;font-weight:600">Endmember Spectra (R·G·B reflectance)</p>', unsafe_allow_html=True)
    import numpy as np
    for i, em in enumerate(result["endmembers"]):
        r, g, b = em[0]*100, em[1]*100, em[2]*100
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:4px 0">
  <div style="width:12px;height:12px;border-radius:50%;background:rgb({int(em[0]*255)},{int(em[1]*255)},{int(em[2]*255)})"></div>
  <span style="color:#8b949e;font-size:0.8rem;width:70px">EM {i+1}</span>
  <div style="flex:1;background:#21262d;border-radius:4px;height:8px;overflow:hidden">
    <div style="width:{r:.0f}%;background:#f85149;height:100%;float:left"></div>
  </div>
  <span style="color:#6e7681;font-size:0.75rem">R:{r:.1f} G:{g:.1f} B:{b:.1f}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _show_wavelet_results(result):
    import numpy as np
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown(f"### 〰️ Wavelet Enhancement — {result['wavelet_used']}")
    st.markdown(f'<p style="color:#8b949e;font-size:0.82rem">Levels: {result["levels_used"]} · Threshold: {result["threshold_mode"]}</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p style="color:#58a6ff;font-size:0.78rem;font-weight:600">Original (gray)</p>', unsafe_allow_html=True)
        st.image(st.session_state["uploaded_img_obj"].convert("L"), use_container_width=True)
    with col_b:
        st.markdown('<p style="color:#3fb950;font-size:0.78rem;font-weight:600">Enhanced</p>', unsafe_allow_html=True)
        st.image(result["enhanced"], use_container_width=True)

    if result.get("subbands"):
        with st.expander("📊 Wavelet Subbands (level 1)"):
            sb = result["subbands"][0]
            c1, c2, c3 = st.columns(3)
            for col, img_sub, label in zip([c1,c2,c3], sb, ["Horizontal","Vertical","Diagonal"]):
                with col:
                    st.markdown(f'<p style="color:#6e7681;font-size:0.75rem;text-align:center">{label}</p>', unsafe_allow_html=True)
                    st.image(img_sub, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)