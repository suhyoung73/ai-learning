import streamlit as st

st.set_page_config(page_title="인공지능 학습 원리 시뮬레이션", layout="wide")
st.title("인공지능 학습 원리 시뮬레이션")
st.markdown("### 📂 머신러닝(Machine Learning)")
c1, c2 = st.columns([1, 10])
with c2:
    st.markdown("#### 📂 지도학습")
    col1, col2, col3 = st.columns([1, 4, 4])
    with col2:
        st.page_link("pages/예측.py", label="1-1) 예측", icon="📏")
    with col3:
        st.page_link("pages/분류.py", label="1-2) 분류", icon="📚")

c3, c4 = st.columns([1, 10])
with c4:
    st.markdown("#### 📂 비지도학습")
    col1, col2 = st.columns([1, 8])
    with col2:
        st.page_link("pages/군집.py", label="2-1) 군집", icon="💰")
c5, c6 = st.columns([1, 10])
with c6:
    st.markdown("#### 📂 강화학습")
    col1, col2 = st.columns([1, 8])
    with col2:
        st.markdown("🚧 준비중... 🚧 ")

st.markdown("### 📂 딥러닝(Deep Learning)")
c7, c8 = st.columns([1, 10])
with c8:
    st.markdown("#### 📂 인공신경망")
    col1, col2 = st.columns([1, 8])
    with col2:
        st.markdown("🚧 준비중... 🚧 ")
