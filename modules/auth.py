import hashlib
import os

import streamlit as st


def _expected_hash() -> str | None:
    password = None
    try:
        password = st.secrets.get("DASHBOARD_PASSWORD")
    except Exception:
        password = None
    if not password:
        password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return None
    return hashlib.sha256(password.encode()).hexdigest()


def check_password() -> bool:
    """Gates the app behind a password from the DASHBOARD_PASSWORD env var.

    If DASHBOARD_PASSWORD is unset, the app is open (useful for local dev).
    Once entered correctly, the unlock is cached in session_state for the
    rest of the browser session.
    """
    expected = _expected_hash()
    if expected is None:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.markdown('<div class="hero-title">SocksCarving Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">This dashboard is password protected. '
        'Enter the access password to continue.</div>',
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        pw_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock", type="primary")

    if submitted:
        if hashlib.sha256(pw_input.encode()).hexdigest() == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False
