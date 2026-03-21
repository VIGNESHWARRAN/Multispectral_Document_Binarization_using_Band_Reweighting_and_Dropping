"""frontend/app.py"""
import streamlit as st

st.set_page_config(
    page_title="AMDES",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.auth import init_session, handle_callback, is_logged_in
from frontend.components.styles import inject_styles
from frontend.components.sidebar import render_sidebar
from frontend.pages import login, dashboard, upload, results

inject_styles()
init_session()
handle_callback()

page = render_sidebar()

# Allow dashboard button to navigate
if st.session_state.get("_nav"):
    page = st.session_state.pop("_nav")

if not is_logged_in():
    login.render()
elif page == "🏠  Dashboard":
    dashboard.render()
elif page == "📤  Upload & Process":
    upload.render()
elif page == "🔬  Results & Export":
    results.render()