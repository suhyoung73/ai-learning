import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import plotly.express as px


st.title("🧑‍💼 비지도학습 - 군집(Clustering)")

df = pd.read_csv("data/Customer.csv", encoding='latin1')
X = df[['Age', 'Income']]

st.markdown("""
이 데이터는 고객의 나이, 소득, 교육 수준, 거주 지역 규모 등을 포함하고 있습니다. 
정해진 정답(레이블) 없이 학습한 **군집 모델**이 비슷한 유형의 고객들을 어떻게 군집화하는지 확인해보세요.
""")
st.page_link("https://www.kaggle.com/datasets/dev0914sharma/customer-clustering", label="출처 : Customer Clustering(kaggle)")

st.subheader("📊 데이터 미리보기")
st.dataframe(df[['Age', 'Income', 'Education', 'Settlement size']].head())
st.markdown("""
- Age: 고객의 나이
- Income: 고객의 연간 소득($)
- Education: 고객의 교육 수준(0~3)
- Settlement size: 고객의 거주 지역 규모(0~2)
""")
features = ['Age', 'Income', 'Education', 'Settlement size']
selected_features = st.multiselect("군집에 사용할 변수(3개)", features, default=features)

if len(selected_features) == 3:
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_features])

    # 군집 수 선택
    n_clusters = st.slider("군집 수", 2, 5, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 군집별 통계 요약
    st.subheader("📊 데이터 군집 결과보기")
    st.markdown("3개의 변수에 대해 K-평균 알고리즘으로 분석한 결과, 각 군집(Cluster)의 평균 값은 다음과 같습니다.")
    cluster_summary = df.groupby('Cluster')[selected_features].mean().round(2)
    st.dataframe(cluster_summary)

    # 3D 시각화
    st.subheader("📊 데이터 시각화 결과보기")
    x_col, y_col, z_col = selected_features[:3]
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=df['Cluster'].astype(str),
                        labels={"color": "Cluster"})
    fig.update_traces(marker=dict(opacity=0.5, size=2))
    fig.update_layout(width=800, height=800)
    st.plotly_chart(fig)

else:
    st.warning("3차원 환경에서 결과를 확인할 수 있도록 변수는 3개만 선택해주세요.")