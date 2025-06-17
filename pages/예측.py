import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import koreanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

st.title("📏 지도학습 - 예측(Prediction)")
st.markdown("""
이 데이터는 영국의 통계학자 프랜시스 골턴이 수집한 가족의 키 데이터로 가족 구성원의 성별과 키를 포함하고 있습니다.
데이터를 학습한 **예측 모델**이 나의 미래 자녀의 키를 어떻게 예측하는지 확인해보세요.
""")
st.page_link("https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data", label="출처 : Parents' Heights vs Adult Children's Heights(kaggle)")

df = pd.read_csv("data/Galton_Families_Heights.csv")
st.subheader("📊 데이터 미리보기")
st.dataframe(df.head())
st.markdown("""
- 키 데이터의 단위는 인치(inch)로, 1 inch는 약 2.54 cm입니다.
""")

# 모델1 : 단변수 모델 (평균 키)
X_mean = df[['father', 'mother']].mean(axis=1).values.reshape(-1, 1) * 2.54
y = df['childHeight'].values * 2.54
model1 = LinearRegression().fit(X_mean, y)

# 모델2 : 다중 선형 회귀 모델 (father/mother 키 각각 사용)
X_multi = df[['father', 'mother']].values * 2.54
father = df['father'].values * 2.54
mother = df['mother'].values * 2.54
child = y
model2 = LinearRegression().fit(X_multi, y)

st.subheader("📊 데이터 시각화 결과보기")
st.markdown("""
- **모델1** : 부모의 평균 키를 기반으로 자녀의 키를 예측한 선형 회귀 모델
""")
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.scatter(X_mean, y, alpha=0.3, label="실제 데이터")
ax1.plot(X_mean, model1.predict(X_mean), color='red', label="예측 모델")
ax1.set_xlabel("부모 평균 키(cm)")
ax1.set_ylabel("자녀 키(cm)")
ax1.legend()
st.pyplot(fig1)

st.markdown("""
- **모델2** : 부모 각각의 키를 기반으로 자녀의 키를 예측한 다중 회귀 모델
""")
st.subheader("📊 부모 각각 키 기반 예측 모델 시각화 (3D)")
fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(father, mother, child, alpha=0.3, label="실제 데이터")

f_range = np.linspace(father.min(), father.max(), 30)
m_range = np.linspace(mother.min(), mother.max(), 30)
F, M = np.meshgrid(f_range, m_range)
X_grid = np.c_[F.ravel(), M.ravel()]
Z = model2.predict(X_grid).reshape(F.shape)

ax2.plot_surface(F, M, Z, alpha=0.3, color='red', label="예측 평면")
ax2.set_xlabel("아버지 키(cm)")
ax2.set_ylabel("어머니 키(cm)")
ax2.set_zlabel("자녀 키(cm)")
st.pyplot(fig2)

st.subheader("📊 데이터 조작해보기")
col1, col2 = st.columns(2)
with col1:
    my_height = st.slider("나의 키(cm)", 130, 200, 160)
with col2:
    partner_height = st.slider("미래 배우자의 키(cm)", 130, 200, 160)

# 평균 키 기반 예측
avg_parent_height = (my_height + partner_height) / 2
pred1 = model1.predict([[avg_parent_height]])[0]

# 다중 회귀 기반 예측
X_input_multi = [[my_height, partner_height]]
pred2 = model2.predict(X_input_multi)[0]

# 결과 출력
st.success(f"📏 부모 평균 키 기반 예측: **{pred1:.1f} cm**")
st.success(f"📐 아빠/엄마 키 각각 사용한 예측: **{pred2:.1f} cm**")