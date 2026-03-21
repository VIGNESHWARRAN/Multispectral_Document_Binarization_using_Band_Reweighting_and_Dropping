"""
AMDES — Streamlit Frontend
Advanced Multispectral Document Enhancement System

Auth: Auth0 (Regular Web Application — server-side token exchange with client secret)

Run:
    pip install streamlit requests pillow
    streamlit run app.py
"""

import streamlit as st
import requests
from PIL import Image
import io, os, numpy as np
from urllib.parse import urlencode

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMDES — Document Enhancement",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Auth0 Config ─────────────────────────────────────────────────────────────
# Fill these in .streamlit/secrets.toml
AUTH0_DOMAIN        = st.secrets.get("AUTH0_DOMAIN",        os.getenv("AUTH0_DOMAIN",        "YOUR_DOMAIN.us.auth0.com"))
AUTH0_CLIENT_ID     = st.secrets.get("AUTH0_CLIENT_ID",     os.getenv("AUTH0_CLIENT_ID",     "YOUR_CLIENT_ID"))
AUTH0_CLIENT_SECRET = st.secrets.get("AUTH0_CLIENT_SECRET", os.getenv("AUTH0_CLIENT_SECRET", "YOUR_CLIENT_SECRET"))
AUTH0_CALLBACK      = st.secrets.get("AUTH0_CALLBACK",      os.getenv("AUTH0_CALLBACK",      "http://localhost:8501"))
BACKEND_URL         = st.secrets.get("BACKEND_URL",         os.getenv("BACKEND_URL",         "http://localhost:8000"))

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F3864 0%, #2E5FA3 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    .amdes-card {
        background: white; border-radius: 12px;
        padding: 28px 32px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 20px;
    }
    .amdes-title {
        background: linear-gradient(90deg, #1F3864, #2E5FA3);
        color: white; padding: 18px 32px; border-radius: 10px; margin-bottom: 24px;
    }
    div[data-testid="metric-container"] {
        background: white; border-radius: 10px; padding: 12px 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    h1, h2, h3 { color: #1F3864; }
    .stButton>button {
        background: #1F3864; color: white; border: none;
        border-radius: 8px; padding: 10px 28px; font-weight: 600;
    }
    .stButton>button:hover { background: #2E5FA3; }
</style>
""", unsafe_allow_html=True)


# ─── Auth0 Helpers ────────────────────────────────────────────────────────────

def get_login_url() -> str:
    """Build Auth0 authorization URL (standard code flow with client secret)."""
    params = {
        "response_type": "code",
        "client_id":     AUTH0_CLIENT_ID,
        "redirect_uri":  AUTH0_CALLBACK,
        "scope":         "openid profile email",
    }
    return f"https://{AUTH0_DOMAIN}/authorize?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Exchange authorization code for tokens using client secret."""
    resp = requests.post(
        f"https://{AUTH0_DOMAIN}/oauth/token",
        json={
            "grant_type":    "authorization_code",
            "client_id":     AUTH0_CLIENT_ID,
            "client_secret": AUTH0_CLIENT_SECRET,
            "code":          code,
            "redirect_uri":  AUTH0_CALLBACK,
        },
        timeout=10,
    )
    return resp.json()


def get_userinfo(access_token: str) -> dict:
    resp = requests.get(
        f"https://{AUTH0_DOMAIN}/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    return resp.json()


def handle_callback():
    """Runs on every page load — picks up Auth0 redirect code from URL."""
    code = st.query_params.get("code")
    if code and not st.session_state.get("user"):
        try:
            tokens = exchange_code(code)
            if "access_token" in tokens:
                user = get_userinfo(tokens["access_token"])
                st.session_state["user"]         = user
                st.session_state["access_token"] = tokens["access_token"]
                st.query_params.clear()          # clean ?code=xxx from URL
            else:
                st.error(f"Auth0 error: {tokens.get('error_description', tokens)}")
        except Exception as e:
            st.error(f"Login failed: {e}")


def logout():
    for k in ["user", "access_token", "uploaded_image", "uploaded_img_obj",
              "uploaded_name", "binarized_image"]:
        st.session_state.pop(k, None)
    logout_url = (f"https://{AUTH0_DOMAIN}/v2/logout?"
                  f"client_id={AUTH0_CLIENT_ID}&returnTo={AUTH0_CALLBACK}")
    st.markdown(f'<meta http-equiv="refresh" content="0; url={logout_url}">', unsafe_allow_html=True)
    st.stop()


def check_backend() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
    except:
        return False


# ─── Session init + callback ──────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state["user"] = None

handle_callback()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📜 AMDES")
    st.markdown("*Advanced Multispectral Document Enhancement System*")
    st.markdown("---")

    if st.session_state["user"]:
        user    = st.session_state["user"]
        name    = user.get("name", user.get("email", "User"))
        email   = user.get("email", "")
        picture = user.get("picture", "")

        if picture:
            st.image(picture, width=72)
        st.markdown(f"**{name}**")
        st.markdown(f"<small>{email}</small>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio("", ["🏠 Dashboard", "📤 Upload Image", "🔬 Results"],
                        label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout"):
            logout()
    else:
        page = "login"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state["user"]:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class="amdes-title" style="text-align:center">
            <h1 style="color:white;margin:0">📜 AMDES</h1>
            <p style="color:#CBD5E0;margin:4px 0 0">
                Advanced Multispectral Document Enhancement System
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
        st.markdown("### Welcome")
        st.markdown(
            "AMDES uses **CNN-based binarization** to restore historical manuscript images. "
            "Sign in to get started."
        )
        st.markdown("")

        login_url = get_login_url()
        st.markdown(f"""
        <a href="{login_url}" target="_self" style="text-decoration:none">
            <div style="background:#1F3864;color:white;text-align:center;padding:14px;
                        border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;
                        letter-spacing:0.3px;">
                🔐 Sign In / Register
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.caption("Secure login powered by Auth0. Supports email/password and Google.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Dashboard":
    user = st.session_state["user"]
    name = user.get("name", "Researcher")

    st.markdown(f"""
    <div class="amdes-title">
        <h2 style="color:white;margin:0">👋 Welcome, {name.split()[0]}!</h2>
        <p style="color:#CBD5E0;margin:4px 0 0">AMDES Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📤 Session Uploads", "1" if st.session_state.get("uploaded_image") else "0")
    with c2:
        st.metric("🔬 Binarization", "✅ Done" if st.session_state.get("binarized_image") else "—")
    with c3:
        st.metric("🧠 Backend", "🟢 Online" if check_backend() else "🔴 Offline")

    st.markdown("---")
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown("### Pipeline")
    st.markdown("""
| Step | Description | Status |
|------|-------------|--------|
| 1️⃣ Upload | Upload manuscript image (PNG/JPG/TIFF) | Active |
| 2️⃣ Binarize | CNN model separates ink from background | HuggingFace API (placeholder) |
| 3️⃣ Results | Side-by-side view + download | Active |
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Image":
    st.markdown("""
    <div class="amdes-title">
        <h2 style="color:white;margin:0">📤 Upload Document Image</h2>
        <p style="color:#CBD5E0;margin:4px 0 0">Upload a manuscript or document image for binarization</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown("**Accepted formats:** PNG, JPG, TIFF &nbsp;|&nbsp; **Max size:** 200 MB")

    uploaded = st.file_uploader(
        "Drag and drop or click to browse",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        if size_mb > 200:
            st.error(f"❌ File too large ({size_mb:.1f} MB). Max: 200 MB.")
        else:
            img = Image.open(uploaded)
            w, h = img.size

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Preview**")
                st.image(img, use_column_width=True, caption=uploaded.name)
            with col2:
                st.markdown("**File Info**")
                st.markdown(f"""
| | |
|--|--|
| **Name** | `{uploaded.name}` |
| **Format** | {img.format or uploaded.type} |
| **Size** | {w} × {h} px |
| **File size** | {size_mb:.2f} MB |
| **Mode** | {img.mode} |
                """)

            st.session_state["uploaded_image"]   = uploaded.getvalue()
            st.session_state["uploaded_name"]    = uploaded.name
            st.session_state["uploaded_img_obj"] = img
            st.success(f"✅ **{uploaded.name}** loaded.")
            st.markdown("")

            if st.button("🔬 Run Binarization", use_container_width=True):
                with st.spinner("Sending to binarization model..."):
                    try:
                        files = {"file": (uploaded.name, st.session_state["uploaded_image"], uploaded.type)}
                        resp  = requests.post(f"{BACKEND_URL}/binarize", files=files, timeout=60)

                        if resp.status_code == 200:
                            st.session_state["binarized_image"] = resp.content
                            st.success("✅ Done! Go to **🔬 Results** to view output.")
                            st.balloons()
                        elif resp.status_code == 503:
                            st.warning(f"⏳ {resp.json().get('detail', 'Model warming up. Wait 20s and retry.')}")
                        else:
                            st.error(f"❌ Backend error: {resp.json().get('detail', resp.text)}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot reach backend. Is `uvicorn main:app --port 8000` running?")
                    except Exception as e:
                        st.error(f"❌ {e}")
    else:
        st.info("👆 Upload an image to begin.")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Results":
    st.markdown("""
    <div class="amdes-title">
        <h2 style="color:white;margin:0">🔬 Binarization Results</h2>
        <p style="color:#CBD5E0;margin:4px 0 0">Original vs. CNN-binarized output</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("binarized_image"):
        st.warning("⚠️ No results yet. Upload an image and run binarization first.")
        st.stop()

    orig_img    = st.session_state["uploaded_img_obj"]
    binary_img  = Image.open(io.BytesIO(st.session_state["binarized_image"]))
    fname       = st.session_state.get("uploaded_name", "document")

    # Side by side
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown("### Side-by-Side Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(orig_img, use_column_width=True)
    with col2:
        st.markdown("**Binarized Output**")
        st.image(binary_img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Image Statistics")
    orig_arr = np.array(orig_img.convert("L"))
    bin_arr  = np.array(binary_img.convert("L"))
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Original Mean", f"{orig_arr.mean():.1f}")
    with c2: st.metric("Binary Mean",   f"{bin_arr.mean():.1f}")
    with c3:
        ink_pct = (bin_arr < 128).sum() / bin_arr.size * 100
        st.metric("Ink Coverage", f"{ink_pct:.1f}%")
    with c4:
        w, h = binary_img.size
        st.metric("Output Resolution", f"{w}×{h}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Download
    st.markdown('<div class="amdes-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Download")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Binarized PNG",
            data=st.session_state["binarized_image"],
            file_name=f"{fname.rsplit('.', 1)[0]}_binarized.png",
            mime="image/png",
            use_container_width=True,
        )
    with col2:
        buf = io.BytesIO()
        orig_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Original PNG",
            data=buf.getvalue(),
            file_name=f"{fname.rsplit('.', 1)[0]}_original.png",
            mime="image/png",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Process Another Image"):
        for k in ["binarized_image", "uploaded_image", "uploaded_img_obj", "uploaded_name"]:
            st.session_state.pop(k, None)
        st.rerun()