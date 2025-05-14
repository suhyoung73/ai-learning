import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import koreanize_matplotlib

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

X = df[['father', 'mother']].mean(axis=1).values.reshape(-1, 1) * 2.54
y = df['childHeight'].values * 2.54
model = LinearRegression().fit(X, y)

st.subheader("📊 데이터 시각화 결과보기")
fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.3, label="실제 데이터")
ax.plot(X, model.predict(X), color='red', label="예측 모델")
ax.set_xlabel("부모 평균 키(cm)")
ax.set_ylabel("자녀 키(cm)")
ax.legend()
st.pyplot(fig)


st.subheader("📊 데이터 조작해보기")
my_height = st.slider("나의 키(cm)", 130, 200, 160)
partner_height = st.slider("미래 배우자의 키(cm)", 130, 200, 160)
avg_parent_height = (my_height + partner_height) / 2
prediction = model.predict([[avg_parent_height]])[0]
st.success(f"미래 자녀의 키는 **{prediction:.1f} cm** 로 예측됩니다👶")
