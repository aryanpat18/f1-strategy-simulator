"""
Streamlit Cloud entry point.

Streamlit Cloud defaults to running 'streamlit_app.py' at the repo root.
The actual dashboard lives in app/streamlit_app.py — this wrapper
delegates to it so the app works with default Streamlit Cloud settings.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_app = os.path.join(_here, "app", "streamlit_app.py")

# Ensure project root is importable (for dashboard.api_client)
if _here not in sys.path:
    sys.path.insert(0, _here)

# Point __file__ at the real app so its PROJECT_ROOT calculation works
__file__ = _app

with open(_app) as _f:
    exec(compile(_f.read(), _app, "exec"))
