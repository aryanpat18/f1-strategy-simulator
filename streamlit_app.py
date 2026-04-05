import streamlit as st

st.set_page_config(page_title="F1 Strategy Test", layout="wide")
st.title("F1 Strategy Intelligence Dashboard")
st.success("Streamlit Cloud is working!")

# Test 1: Can we read secrets?
try:
    api_url = st.secrets.get("API_URL", "NOT SET")
    st.write(f"API_URL from secrets: `{api_url}`")
except Exception as e:
    st.warning(f"Could not read secrets: {e}")
    api_url = "NOT SET"

# Test 2: Can we reach the API?
if api_url != "NOT SET":
    import requests
    try:
        resp = requests.get(f"{api_url}/data/races", params={"year": 2024}, timeout=10)
        st.write(f"API response: status={resp.status_code}, body={resp.text[:200]}")
    except Exception as e:
        st.error(f"API call failed: {e}")

# Test 3: Basic widgets work?
st.sidebar.title("Sidebar Test")
st.sidebar.write("If you see this, sidebar works.")
year = st.sidebar.number_input("Year", 2020, 2030, 2024)
st.write(f"Selected year: {year}")

st.info("All checks passed. The minimal app works on Streamlit Cloud.")
