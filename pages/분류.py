import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

st.title("📚 지도학습 - 분류(Classification)")
st.markdown("""
이 데이터는 학생들의 공부 시간, 이전 시험 성적, 그리고 최근 시험의 합격/불합격 여부를 포함하고 있습니다.
데이터를 학습한 **분류 모델**이 학생의 합격 여부를 어떻게 예측하는지 확인해보세요.
""")
st.page_link("https://www.kaggle.com/datasets/mrsimple07/student-exam-performance-prediction", label="출처 : Student Exam Performance Prediction(kaggle)")

df = pd.read_csv("data/Student_Exam.csv")
st.subheader("📊 데이터 미리보기")
st.dataframe(df.head())

st.subheader("📊 데이터 시각화 결과보기")
# 공부 시간 - 합격 여부
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="Study Hours", y="Pass/Fail", ax=ax1)
X = df[["Study Hours"]]
y = df["Pass/Fail"]
model_hours = LogisticRegression()
model_hours.fit(X, y)

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model_hours.predict_proba(x_range)[:, 1]
ax1.plot(x_range, y_prob, color='red', label="분류 모델")

threshold = 0.5
diff = np.abs(y_prob - threshold)
boundary_index = diff.argmin()
boundary_x = x_range[boundary_index][0]

ax1.axvline(x=boundary_x, color='green', linestyle='--', label=f'합격 기준: {boundary_x:.2f}시간')
ax1.set_xlabel("공부 시간")
ax1.set_ylabel("합격 여부")
ax1.legend()
st.pyplot(fig1)

# 이전 성적 - 합격 여부
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="Previous Exam Score", y="Pass/Fail", ax=ax2)
X_score = df[["Previous Exam Score"]]
model_score = LogisticRegression()
model_score.fit(X_score, y)

x_range_score = np.linspace(X_score.min(), X_score.max(), 300).reshape(-1, 1)
y_prob_score = model_score.predict_proba(x_range_score)[:, 1]
ax2.plot(x_range_score, y_prob_score, color='blue', label="분류 모델")

threshold_score = 0.5
diff_score = np.abs(y_prob_score - threshold_score)
boundary_index_score = diff_score.argmin()
boundary_score = x_range_score[boundary_index_score][0]

ax2.axvline(x=boundary_score, color='green', linestyle='--', label=f'합격 기준: {boundary_score:.2f}점')
ax1.set_xlabel("이전 시험 성적")
ax1.set_ylabel("합격 여부")
ax2.legend()
st.pyplot(fig2)


st.subheader("📊 데이터 조작해보기")
X = df[["Study Hours", "Previous Exam Score"]]
y = df["Pass/Fail"]
model = LogisticRegression().fit(X, y)

# 입력 슬라이더
hours = st.slider("공부 시간", float(df["Study Hours"].min()), float(df["Study Hours"].max()), float(df["Study Hours"].mean()))
score = st.slider("이전 시험 성적", float(df["Previous Exam Score"].min()), float(df["Previous Exam Score"].max()), float(df["Previous Exam Score"].mean()))

pred = model.predict([[hours, score]])[0]
prob = model.predict_proba([[hours, score]])[0][1]
st.success(f"분석 결과 {prob:.2%} 의 확률로 **{'합격' if pred == 1 else '불합격'}** 입니다{'😊' if pred == 1 else '🥺'}")
