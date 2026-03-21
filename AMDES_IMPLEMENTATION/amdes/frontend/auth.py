"""
frontend/auth.py
─────────────────
Auth0 helpers for the Streamlit frontend.

Flow: Authorization Code (server-side token exchange with client secret).
  1. User clicks "Sign In" → redirected to Auth0 /authorize
  2. Auth0 redirects back to CALLBACK_URL with ?code=xxx
  3. handle_callback() exchanges code for tokens, fetches user info,
     stores in st.session_state
"""

import streamlit as st
import requests
from urllib.parse import urlencode

from config.settings import (
    AUTH0_DOMAIN,
    AUTH0_CLIENT_ID,
    AUTH0_CLIENT_SECRET,
    AUTH0_CALLBACK,
)


# ─── URL builders ─────────────────────────────────────────────────────────────

def get_login_url() -> str:
    """Build the Auth0 authorization URL."""
    params = {
        "response_type": "code",
        "client_id":     AUTH0_CLIENT_ID,
        "redirect_uri":  AUTH0_CALLBACK,
        "scope":         "openid profile email",
    }
    return f"https://{AUTH0_DOMAIN}/authorize?{urlencode(params)}"


def get_logout_url() -> str:
    """Build the Auth0 logout URL that clears the Auth0 session."""
    params = {
        "client_id": AUTH0_CLIENT_ID,
        "returnTo":  AUTH0_CALLBACK,
    }
    return f"https://{AUTH0_DOMAIN}/v2/logout?{urlencode(params)}"


# ─── Token exchange ────────────────────────────────────────────────────────────

def _exchange_code(code: str) -> dict:
    """POST authorization code → tokens."""
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


def _get_userinfo(access_token: str) -> dict:
    """Fetch user profile from Auth0 /userinfo."""
    resp = requests.get(
        f"https://{AUTH0_DOMAIN}/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    return resp.json()


# ─── Session helpers ───────────────────────────────────────────────────────────

def init_session():
    """Initialise auth-related session state keys (safe to call multiple times)."""
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("access_token", None)


def is_logged_in() -> bool:
    return bool(st.session_state.get("user"))


def get_current_user() -> dict | None:
    return st.session_state.get("user")


def handle_callback():
    """
    Runs on every page load.
    Picks up the ?code= query parameter from the Auth0 redirect and
    exchanges it for tokens. Cleans the URL when done.
    """
    code = st.query_params.get("code")
    if code and not is_logged_in():
        try:
            tokens = _exchange_code(code)
            if "access_token" in tokens:
                user = _get_userinfo(tokens["access_token"])
                st.session_state["user"]         = user
                st.session_state["access_token"] = tokens["access_token"]
                st.query_params.clear()           # remove ?code=xxx from URL
            else:
                st.error(f"Auth0 error: {tokens.get('error_description', tokens)}")
        except Exception as e:
            st.error(f"Login failed: {e}")


def logout():
    """Clear local session and redirect to Auth0 logout endpoint."""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    logout_url = get_logout_url()
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={logout_url}">',
        unsafe_allow_html=True,
    )
    st.stop()
