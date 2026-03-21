"""frontend/components/sidebar.py"""
import streamlit as st
from frontend.auth import get_current_user, logout, is_logged_in


def render_sidebar() -> str:
    with st.sidebar:
        # Logo & title
        st.markdown("""
<div style="padding:8px 0 16px 0">
  <div style="font-size:1.5rem;font-weight:800;color:#58a6ff;letter-spacing:-0.02em">
    📜 AMDES
  </div>
  <div style="font-size:0.72rem;color:#6e7681;margin-top:2px;line-height:1.4">
    Advanced Multispectral<br>Document Enhancement System
  </div>
</div>
""", unsafe_allow_html=True)

        if not is_logged_in():
            st.markdown("---")
            st.markdown('<p style="color:#8b949e;font-size:0.85rem">Please sign in to continue.</p>', unsafe_allow_html=True)
            return "login"

        user    = get_current_user()
        name    = user.get("name", user.get("email", "User"))
        email   = user.get("email", "")
        picture = user.get("picture", "")

        st.markdown('<hr style="border-color:#30363d;margin:4px 0 12px 0">', unsafe_allow_html=True)

        # User profile
        col1, col2 = st.columns([1, 3])
        with col1:
            if picture:
                st.image(picture, width=44)
            else:
                initials = "".join(p[0].upper() for p in name.split()[:2])
                st.markdown(f"""
<div style="width:40px;height:40px;background:#1f6feb;border-radius:50%;
display:flex;align-items:center;justify-content:center;
font-weight:700;font-size:0.85rem;color:white">{initials}</div>
""", unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="color:#e6edf3;font-weight:600;font-size:0.88rem;line-height:1.2">{name.split()[0] if name else "User"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#6e7681;font-size:0.75rem;overflow:hidden;text-overflow:ellipsis">{email[:22]}{"…" if len(email)>22 else ""}</div>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#30363d;margin:12px 0">', unsafe_allow_html=True)

        # Navigation
        st.markdown('<p style="color:#6e7681;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:6px">Navigation</p>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["🏠  Dashboard", "📤  Upload & Process", "🔬  Results & Export"],
            label_visibility="collapsed",
        )

        st.markdown('<hr style="border-color:#30363d;margin:12px 0">', unsafe_allow_html=True)

        # Session status
        st.markdown('<p style="color:#6e7681;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:8px">Session</p>', unsafe_allow_html=True)
        has_upload = bool(st.session_state.get("uploaded_image"))
        has_result = bool(st.session_state.get("binarized_image"))

        def _badge(label, active):
            color = "#238636" if active else "#30363d"
            text  = "#3fb950" if active else "#484f58"
            dot   = "●" if active else "○"
            return f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0"><span style="color:{text};font-size:0.8rem">{dot}</span><span style="color:#8b949e;font-size:0.82rem">{label}</span></div>'

        st.markdown(_badge("Image uploaded", has_upload), unsafe_allow_html=True)
        st.markdown(_badge("Binarization done", has_result), unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#30363d;margin:12px 0">', unsafe_allow_html=True)
        if st.button("⏻  Sign Out", use_container_width=True):
            logout()

    return page