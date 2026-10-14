import streamlit as st

st.set_page_config(page_title="Appliance Energy ML Lab", page_icon="⚡", layout="wide")

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/dataset.py", title="Dataset Overview", icon="📊"),
    st.Page("pages/feature_selection.py", title="Feature Selection", icon="🧩"),
    st.Page("pages/loss_functions.py", title="Loss Functions", icon="📉"),
    st.Page("pages/results.py", title="Results & Conclusions", icon="🏆"),
]

pg = st.navigation(pages)
st.sidebar.caption("CRISP-DM · 7 selectors + baseline × 4 losses")
st.sidebar.caption("Scores shown only when actual experiment CSVs exist.")
pg.run()
