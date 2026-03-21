"""frontend/pages/results.py"""

import io
import numpy as np
import streamlit as st
from PIL import Image

from utils.image_utils import image_stats
from utils.metrics import compute_all_metrics
from utils.tesseract_utils import ocr_image as run_ocr
from utils.export_utils import (
    export_png, export_jpeg, export_tiff,
    export_geotiff, export_pdf_report, export_text_report,
)


# ── Cache metrics so they don't recompute on every widget interaction ─────────
@st.cache_data(show_spinner=False)
def _cached_metrics(orig_bytes: bytes, bin_bytes: bytes, lang: str, run_ocr_flag: bool):
    orig = Image.open(io.BytesIO(orig_bytes))
    bina = Image.open(io.BytesIO(bin_bytes))
    return compute_all_metrics(orig, bina, run_ocr_flag=run_ocr_flag, ocr_lang=lang)


def render():
    st.markdown("""
<div class="amdes-title">
  <h2>🔬 Results &amp; Export</h2>
  <p>Quality metrics · OCR extraction · Multi-format download</p>
</div>
""", unsafe_allow_html=True)

    if not st.session_state.get("binarized_image"):
        st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
padding:48px 32px;text-align:center;margin-top:20px">
  <div style="font-size:2.5rem;margin-bottom:12px">🔬</div>
  <div style="color:#e6edf3;font-size:1rem;font-weight:600;margin-bottom:6px">No results yet</div>
  <div style="color:#8b949e;font-size:0.88rem">Upload an image and run binarization first.</div>
</div>
""", unsafe_allow_html=True)
        return

    orig_img   = st.session_state["uploaded_img_obj"]
    binary_img = Image.open(io.BytesIO(st.session_state["binarized_image"]))
    fname      = st.session_state.get("uploaded_name", "document")
    stem       = fname.rsplit(".", 1)[0]

    # ── OCR settings ──────────────────────────────────────────────────────
    with st.expander("⚙️  Analysis Settings", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ocr_lang = st.selectbox(
                "OCR Language",
                ["eng", "lat", "grc", "heb", "ara", "fra", "deu", "spa"],
                format_func=lambda x: {
                    "eng": "English", "lat": "Latin", "grc": "Greek (ancient)",
                    "heb": "Hebrew", "ara": "Arabic", "fra": "French",
                    "deu": "German", "spa": "Spanish",
                }.get(x, x),
            )
        with col_s2:
            run_ocr_flag = st.checkbox("Run OCR analysis", value=True)

    # ── Run metrics (cached) ──────────────────────────────────────────────
    metrics_key = f"metrics_{fname}_{ocr_lang}_{run_ocr_flag}"
    if metrics_key not in st.session_state:
        with st.spinner("Computing quality metrics and OCR…"):
            st.session_state[metrics_key] = _cached_metrics(
                st.session_state["uploaded_image"],
                st.session_state["binarized_image"],
                ocr_lang,
                run_ocr_flag,
            )
    metrics = st.session_state[metrics_key]

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️  Comparison", "📊  Quality Metrics", "🔤  OCR Output", "💾  Export"
    ])

    # ════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Side-by-Side Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:8px">ORIGINAL</p>', unsafe_allow_html=True)
            st.image(orig_img, use_container_width=True)
            orig_s = image_stats(orig_img)
            st.markdown(f'<p style="color:#6e7681;font-size:0.78rem;text-align:center">{orig_s["width"]}×{orig_s["height"]} · {orig_img.mode} · Mean {orig_s["mean"]:.1f}</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:8px">BINARIZED</p>', unsafe_allow_html=True)
            st.image(binary_img, use_container_width=True)
            bin_s = image_stats(binary_img)
            st.markdown(f'<p style="color:#6e7681;font-size:0.78rem;text-align:center">{bin_s["width"]}×{bin_s["height"]} · Mean {bin_s["mean"]:.1f} · Ink {bin_s["ink_pct"]:.1f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    with tab2:
        # Top metric cards
        m_items = [
            ("PSNR",  f"{metrics['psnr']:.2f}",  "dB",  "#58a6ff",  "Peak Signal-to-Noise Ratio. Higher = better fidelity."),
            ("SSIM",  f"{metrics['ssim']:.4f}",  "",    "#3fb950",  "Structural Similarity. 1.0 = identical to original."),
            ("CNR",   f"{metrics['cnr']:.2f}",   "",    "#d2a8ff",  "Contrast-to-Noise Ratio. Higher = cleaner ink separation."),
            ("CER",   f"{metrics['cer']:.2f}" if metrics.get('cer') is not None else "—",
                      "%" if metrics.get('cer') is not None else "",
                      "#ffa657", "Character Error Rate vs original OCR. Lower = better."),
            ("WER",   f"{metrics['wer']:.2f}" if metrics.get('wer') is not None else "—",
                      "%" if metrics.get('wer') is not None else "",
                      "#f85149", "Word Error Rate vs original OCR. Lower = better."),
        ]

        cols = st.columns(5)
        for col, (label, val, unit, color, tooltip) in zip(cols, m_items):
            with col:
                st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:16px 10px;text-align:center;border-top:3px solid {color}">
  <div style="color:#6e7681;font-size:0.68rem;text-transform:uppercase;
  letter-spacing:0.1em;font-weight:600;margin-bottom:8px">{label}</div>
  <div style="color:{color};font-size:1.6rem;font-weight:800;
  letter-spacing:-0.02em;line-height:1">{val}<span style="font-size:0.9rem;font-weight:400;color:#6e7681"> {unit}</span></div>
  <div style="color:#484f58;font-size:0.68rem;margin-top:8px;line-height:1.4">{tooltip}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interpretation guide
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Metric Interpretation")
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        cnr  = metrics['cnr']

        def _quality_bar(val, lo, hi, label, color):
            pct = min(100, max(0, int((val - lo) / (hi - lo) * 100)))
            return f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:#c9d1d9;font-size:0.85rem;font-weight:500">{label}</span>
    <span style="color:{color};font-size:0.85rem;font-weight:700">{val:.3f}</span>
  </div>
  <div style="background:#21262d;border-radius:4px;height:8px;overflow:hidden">
    <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;
    transition:width 0.4s ease"></div>
  </div>
</div>"""

        st.markdown(
            _quality_bar(psnr, 0, 50, f"PSNR — {'Excellent' if psnr>35 else 'Good' if psnr>25 else 'Fair'}", "#58a6ff") +
            _quality_bar(ssim, 0, 1, f"SSIM — {'Excellent' if ssim>0.9 else 'Good' if ssim>0.7 else 'Fair'}", "#3fb950") +
            _quality_bar(min(cnr, 20), 0, 20, f"CNR — {'High contrast' if cnr>10 else 'Medium' if cnr>5 else 'Low contrast'}", "#d2a8ff"),
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Histogram
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Pixel Intensity Distribution")
        orig_arr = np.array(orig_img.convert("L"))
        bin_arr  = np.array(binary_img.convert("L"))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p style="color:#58a6ff;font-size:0.8rem;font-weight:600;margin-bottom:4px">Original</p>', unsafe_allow_html=True)
            st.bar_chart(np.histogram(orig_arr.flatten(), bins=64, range=(0,256))[0], color="#58a6ff")
        with col_b:
            st.markdown('<p style="color:#3fb950;font-size:0.8rem;font-weight:600;margin-bottom:4px">Binarized</p>', unsafe_allow_html=True)
            st.bar_chart(np.histogram(bin_arr.flatten(), bins=64, range=(0,256))[0], color="#3fb950")
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    with tab3:
        if not run_ocr_flag:
            st.info("Enable 'Run OCR analysis' in Settings above to extract text.")
        else:
            ocr_orig = metrics.get("ocr_original") or {}
            ocr_bin  = metrics.get("ocr_binarized") or {}

            # OCR stat badges
            def _ocr_stats(ocr_data, label, color):
                if not ocr_data.get("success"):
                    err = ocr_data.get("error", "OCR failed")
                    st.markdown(f'<div style="background:#161b22;border:1px solid #f85149;border-radius:8px;padding:12px;color:#f85149;font-size:0.85rem">❌ {label}: {err}</div>', unsafe_allow_html=True)
                    return
                st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 16px;margin-bottom:12px;border-left:3px solid {color}">
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div><span style="color:#6e7681;font-size:0.75rem;text-transform:uppercase">Words</span>
    <div style="color:{color};font-size:1.1rem;font-weight:700">{ocr_data.get('word_count',0)}</div></div>
    <div><span style="color:#6e7681;font-size:0.75rem;text-transform:uppercase">Characters</span>
    <div style="color:{color};font-size:1.1rem;font-weight:700">{ocr_data.get('char_count',0)}</div></div>
    <div><span style="color:#6e7681;font-size:0.75rem;text-transform:uppercase">Avg Confidence</span>
    <div style="color:{color};font-size:1.1rem;font-weight:700">{ocr_data.get('avg_conf',0):.1f}%</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

            col_o, col_b2 = st.columns(2)
            with col_o:
                st.markdown('<p style="color:#58a6ff;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em">OCR — Original</p>', unsafe_allow_html=True)
                _ocr_stats(ocr_orig, "Original OCR", "#58a6ff")
                if ocr_orig.get("text"):
                    st.text_area("Extracted text (original)", ocr_orig["text"],
                                 height=280, label_visibility="collapsed")
                else:
                    st.markdown('<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;text-align:center;color:#6e7681;font-size:0.85rem">No text detected</div>', unsafe_allow_html=True)

            with col_b2:
                st.markdown('<p style="color:#3fb950;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em">OCR — Binarized</p>', unsafe_allow_html=True)
                _ocr_stats(ocr_bin, "Binarized OCR", "#3fb950")
                if ocr_bin.get("text"):
                    st.text_area("Extracted text (binarized)", ocr_bin["text"],
                                 height=280, label_visibility="collapsed")
                else:
                    st.markdown('<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;text-align:center;color:#6e7681;font-size:0.85rem">No text detected</div>', unsafe_allow_html=True)

            # CER / WER callout
            if metrics.get("cer") is not None:
                cer_color = "#3fb950" if metrics["cer"] < 10 else "#d29922" if metrics["cer"] < 30 else "#f85149"
                wer_color = "#3fb950" if metrics["wer"] < 10 else "#d29922" if metrics["wer"] < 30 else "#f85149"
                st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:16px 20px;margin-top:8px;display:flex;gap:40px">
  <div>
    <div style="color:#6e7681;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">CER (Character Error Rate)</div>
    <div style="color:{cer_color};font-size:1.4rem;font-weight:700;margin-top:4px">{metrics['cer']:.2f}%
    <span style="color:#6e7681;font-size:0.8rem;font-weight:400"> binarized vs original OCR</span></div>
  </div>
  <div>
    <div style="color:#6e7681;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">WER (Word Error Rate)</div>
    <div style="color:{wer_color};font-size:1.4rem;font-weight:700;margin-top:4px">{metrics['wer']:.2f}%
    <span style="color:#6e7681;font-size:0.8rem;font-weight:400"> binarized vs original OCR</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Export Enhanced Image")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(_export_card("PNG", "Lossless · Archival standard", "#58a6ff"), unsafe_allow_html=True)
            st.download_button(
                "⬇️ PNG",
                data=export_png(binary_img),
                file_name=f"{stem}_binarized.png",
                mime="image/png",
                use_container_width=True,
                key="dl_png",
            )

        with col2:
            st.markdown(_export_card("JPEG", "Compressed · Web sharing", "#3fb950"), unsafe_allow_html=True)
            st.download_button(
                "⬇️ JPEG",
                data=export_jpeg(binary_img),
                file_name=f"{stem}_binarized.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key="dl_jpg",
            )

        with col3:
            st.markdown(_export_card("TIFF", "LZW compressed · Archival", "#d2a8ff"), unsafe_allow_html=True)
            st.download_button(
                "⬇️ TIFF",
                data=export_tiff(binary_img),
                file_name=f"{stem}_binarized.tiff",
                mime="image/tiff",
                use_container_width=True,
                key="dl_tiff",
            )

        with col4:
            st.markdown(_export_card("GeoTIFF", "Identity CRS · QGIS ready", "#ffa657"), unsafe_allow_html=True)
            st.download_button(
                "⬇️ GeoTIFF",
                data=export_geotiff(binary_img),
                file_name=f"{stem}_binarized.geotiff",
                mime="image/tiff",
                use_container_width=True,
                key="dl_geotiff",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Export Original Image")
        col5, col6 = st.columns(4)[:2]
        with col5:
            buf = io.BytesIO()
            orig_img.save(buf, format="PNG")
            st.download_button(
                "⬇️ Original PNG",
                data=buf.getvalue(),
                file_name=f"{stem}_original.png",
                mime="image/png",
                use_container_width=True,
                key="dl_orig_png",
            )
        with col6:
            st.download_button(
                "⬇️ Original TIFF",
                data=export_tiff(orig_img),
                file_name=f"{stem}_original.tiff",
                mime="image/tiff",
                use_container_width=True,
                key="dl_orig_tiff",
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Reports section
        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Export Reports")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown(_export_card("PDF Report", "Multi-page: images + metrics + OCR", "#f78166"), unsafe_allow_html=True)
            pdf_bytes = export_pdf_report(orig_img, binary_img, metrics, fname)
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{stem}_AMDES_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf",
            )

        with col_r2:
            st.markdown(_export_card("Text Report", "Plain-text metrics + OCR text output", "#79c0ff"), unsafe_allow_html=True)
            txt_bytes = export_text_report(metrics, fname, ocr_lang if run_ocr_flag else "eng")
            st.download_button(
                "⬇️ Download Text Report",
                data=txt_bytes,
                file_name=f"{stem}_AMDES_report.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_txt",
            )

        # OCR text-only export
        if run_ocr_flag and metrics.get("ocr_binarized", {}).get("text"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### OCR Text Export")
            ocr_text = metrics["ocr_binarized"]["text"]
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.download_button(
                    "⬇️ OCR Text (Binarized)",
                    data=ocr_text.encode("utf-8"),
                    file_name=f"{stem}_ocr_binarized.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_ocr_bin",
                )
            if metrics.get("ocr_original", {}).get("text"):
                with col_t2:
                    st.download_button(
                        "⬇️ OCR Text (Original)",
                        data=metrics["ocr_original"]["text"].encode("utf-8"),
                        file_name=f"{stem}_ocr_original.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_ocr_orig",
                    )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Reset ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Process Another Image"):
        keys_to_clear = [k for k in st.session_state if k.startswith("metrics_")]
        for k in ["binarized_image","uploaded_image","uploaded_img_obj","uploaded_name"] + keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()


def _export_card(title, desc, color):
    return f"""
<div style="background:#21262d;border-radius:8px;padding:12px;border:1px solid #30363d;
margin-bottom:10px;border-top:2px solid {color}">
  <div style="color:#e6edf3;font-weight:600;font-size:0.88rem">{title}</div>
  <div style="color:#6e7681;font-size:0.76rem;margin-top:2px">{desc}</div>
</div>"""