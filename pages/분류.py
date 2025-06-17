import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

st.title("🎯 지도학습 - 분류(Classification)")
st.markdown("""
이 데이터는 학생들의 공부 시간, 이전 시험 성적, 그리고 최근 시험의 합격/불합격 여부를 포함하고 있습니다.
데이터를 학습한 **분류 모델**이 학생의 합격 여부를 어떻게 예측하는지 확인해보세요.
""")
st.page_link("https://www.kaggle.com/datasets/mrsimple07/student-exam-performance-prediction", label="출처 : Student Exam Performance Prediction(kaggle)")

df = pd.read_csv("data/Student_Exam.csv")
st.subheader("📊 데이터 미리보기")
st.dataframe(df.head())
st.markdown("""
- Pass(1)/Fail(0)
""")

st.subheader("📊 데이터 시각화 결과보기")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - **그래프1** : 공부 시간 - 합격 여부
    """)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
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

with col2:
    st.markdown("""
    - **그래프2** : 이전 시험 성적 - 합격 여부
    """)
    # 이전 시험 성적 - 합격 여부
    fig2, ax2 = plt.subplots(figsize=(5, 4))
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
    ax2.set_xlabel("이전 시험 성적")
    ax2.set_ylabel("합격 여부")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("""
- **K-최근접 이웃(KNN) 모델** : 이웃한 데이터의 공부 시간과 이전 시험 성적에 따라 합격 여부 분류 (K=5일 때)
""")
X = df[["Study Hours", "Previous Exam Score"]]
y = df["Pass/Fail"]
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

# 시각화를 위한 meshgrid 생성
h = 0.2  # 격자 간격
x_min, x_max = X["Study Hours"].min() - 1, X["Study Hours"].max() + 1
y_min, y_max = X["Previous Exam Score"].min() - 1, X["Previous Exam Score"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 시각화
fig3, ax3 = plt.subplots(figsize=(5, 4))
ax3.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
sns.scatterplot(data=df, x="Study Hours", y="Previous Exam Score", hue="Pass/Fail", ax=ax3)
ax3.set_xlabel("공부 시간")
ax3.set_ylabel("이전 시험 성적")
st.pyplot(fig3)

st.subheader("📊 데이터 조작해보기")
col1, col2 = st.columns(2)
with col1:
    k_hours = st.slider("공부 시간", float(df["Study Hours"].min()), float(df["Study Hours"].max()), float(df["Study Hours"].mean()))
with col2:
    k_score = st.slider("이전 시험 성적", float(df["Previous Exam Score"].min()), float(df["Previous Exam Score"].max()), float(df["Previous Exam Score"].mean()))

col3, col4 = st.columns(2)
with col3:
    k_value = st.slider("K 값 (최근접 이웃 수)", min_value=1, max_value=20, value=5)
with col4:
    # KNN 모델 학습
    X = df[["Study Hours", "Previous Exam Score"]]
    y = df["Pass/Fail"]
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X, y)

    # meshgrid 생성
    h = 0.2
    x_min, x_max = X["Study Hours"].min() - 1, X["Study Hours"].max() + 1
    y_min, y_max = X["Previous Exam Score"].min() - 1, X["Previous Exam Score"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 예측 대상 점
    user_point = np.array([[k_hours, k_score]])
    knn_pred = knn_model.predict(user_point)[0]
    knn_prob = knn_model.predict_proba(user_point)[0][1]

    # 이웃 추출
    _, neighbor_indices = knn_model.kneighbors(user_point)
    neighbor_points = X.iloc[neighbor_indices[0]]

    # 시각화
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    sns.scatterplot(data=df, x="Study Hours", y="Previous Exam Score", hue="Pass/Fail", ax=ax4, palette="coolwarm", edgecolor='gray', alpha=0.6)

    # 사용자 입력 위치
    ax4.scatter(k_hours, k_score, color='black', s=100, marker='X', label="입력 위치")

    # 이웃 표시
    ax4.scatter(neighbor_points["Study Hours"], neighbor_points["Previous Exam Score"], 
                facecolors='none', edgecolors='black', s=120, linewidths=2, label='이웃 데이터')

    ax4.set_title(f"KNN 분류 시각화 (K={k_value}일 때)")
    ax4.set_xlabel("공부 시간")
    ax4.set_ylabel("이전 시험 성적")
    ax4.legend()
    st.pyplot(fig4)

# 예측 결과
st.success(f"K-최근접 이웃(KNN) 모델 분석 결과 {knn_prob:.2%} 확률로 **{'합격' if knn_pred == 1 else '불합격'}** 입니다{'😊' if knn_pred == 1 else '🥺'}")
