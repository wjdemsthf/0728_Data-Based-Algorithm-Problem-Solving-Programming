import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 페이지 기본 설정 및 상태 초기화
# ==========================================
st.set_page_config(
    page_title="CSV 데이터로 배우는 선형회귀 실험실",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State 초기화 (탭 전환 시 모델 결과 및 데이터 유지)
if "df" not in st.session_state:
    st.session_state["df"] = None
if "simple_model_results" not in st.session_state:
    st.session_state["simple_model_results"] = None
if "multi_model_results" not in st.session_state:
    st.session_state["multi_model_results"] = None


# ==========================================
# 1. 헬퍼 및 도구 함수 정의
# ==========================================
def generate_sample_data(n_samples=150) -> pd.DataFrame:
    """미세먼지(PM2.5) 예측을 위한 샘플 기상 데이터를 생성합니다."""
    np.random.seed(42)
    temperature = np.random.uniform(-5, 32, n_samples)  # 기온 (-5~32도)
    humidity = np.random.uniform(20, 90, n_samples)  # 습도 (20~90%)
    wind_speed = np.random.uniform(0.5, 8.0, n_samples)  # 풍속 (0.5~8m/s)
    rainfall = np.where(
        np.random.rand(n_samples) > 0.7, np.random.exponential(5, n_samples), 0.0
    )

    # 실제 물리적 노이즈가 포함된 PM2.5 생성 관계식
    pm25 = (
        25
        + (35.0 / (wind_speed + 0.5))
        + (0.3 * humidity)
        - (0.4 * temperature)
        - (2.0 * rainfall)
        + np.random.normal(0, 5, n_samples)
    )
    pm25 = np.clip(pm25, 5, 150)  # 음수 방지 처리

    df = pd.DataFrame(
        {
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1),
            "rainfall": np.round(rainfall, 1),
            "pm25": np.round(pm25, 1),
        }
    )
    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    """업로드된 CSV 파일을 UTF-8 또는 CP949 인코딩으로 읽어옵니다."""
    try:
        content = uploaded_file.read()
        try:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="cp949")
    except Exception as e:
        raise Exception(
            f"CSV 파일을 읽는 중 오류가 발생했습니다. 파일 형식을 확인해주세요. ({str(e)})"
        )


def validate_data(df: pd.DataFrame):
    """데이터의 유효성을 검사하고 안내 메시지를 반환합니다."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return (
            False,
            "선형회귀 분석을 수행하려면 최소 2개 이상의 숫자형(수치형) 열이 필요합니다.",
        )
    return True, f"총 {len(df)}개의 행과 {len(df.columns)}개의 열이 정상적으로 확인되었습니다."


def calculate_metrics(y_true, y_pred, n_features):
    """모델 성능 평가 지표를 계산합니다."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n - n_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = r2

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Adj_R2": round(adj_r2, 4),
    }


def explain_coefficient(feature_name, coef, target_name):
    """기울기(회귀계수)를 학생 눈높이에 맞게 해석합니다."""
    direction = "증가" if coef > 0 else "감소"
    abs_coef = abs(round(coef, 3))
    return f"💡 **해석**: `{feature_name}` 변수가 **1단위 증가**할 때, `{target_name}`의 예측값은 평균적으로 약 **{abs_coef}만큼 {direction}**하는 경향을 보입니다. *(주의: 이는 통계적 경향성이며 직접적인 원인과 결과를 의미하지는 않습니다.)*"


# ==========================================
# 2. 사이드바 구성
# ==========================================
with st.sidebar:
    st.header("📌 학습 안내 및 용어집")
    st.info(
        """
    **인공지능 기초: 선형회귀 실험실**
    데이터를 직접 조작하고 단순/다중 선형회귀 모델을 만들어보며 머신러닝의 기본 원리를 탐구합니다.
    """
    )

    with st.expander("📚 핵심 용어 정리", expanded=True):
        st.markdown(
            """
        * **독립변수(X)**: 영향을 주는 변수 (특징/Feature)
        * **종속변수(y)**: 영향을 받는 변수 (타겟/Target)
        * **회귀계수(기울기)**: X가 1 변할 때 y의 변화량
        * **절편**: X가 0일 때 y의 기본 예측값
        * **잔차(Residual)**: 실제값과 예측값의 차이 ($y - \\hat{y}$)
        * **결정계수($R^2$)**: 모델의 설명력 (0~1 사이)
        """
        )

    st.markdown("---")
    st.caption("고등학교 인공지능 기초 수업용 실습 도구")


# ==========================================
# 3. 메인 화면 및 탭 구성
# ==========================================
st.title("📊 CSV 데이터로 배우는 선형회귀 실험실")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1️⃣ 학습 안내",
        "2️⃣ CSV 데이터 업로드",
        "3️⃣ 데이터 탐색",
        "4️⃣ 단순선형회귀",
        "5️⃣ 다중선형회귀",
        "6️⃣ 모델 평가 및 비교",
    ]
)

# ------------------------------------------
# TAB 1: 학습 안내
# ------------------------------------------
with tab1:
    st.subheader("💡 선형회귀(Linear Regression) 핵심 개념 정리")

    st.markdown(
        """
    ### 1. 회귀(Regression)와 선형회귀란?
    * **회귀**: 연속된 숫자(예: 미세먼지 농도, 키, 집값)를 예측하는 대표적인 지도학습 인공지능 기법입니다.
    * **선형회귀**: 입력 데이터(X)와 정답 데이터(y) 사이에 **직선 형태의 선형적 관계**가 있다고 가정하고, 가장 데이터를 잘 설명하는 최적의 직선을 찾는 알고리즘입니다.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info("🔹 **단순선형회귀 (Simple Linear Regression)**\n하나의 독립변수($X$)로 하나의 종속변수($y$)를 예측합니다.")
        st.latex(r"\hat{y} = b_0 + b_1 x")
        st.caption("$b_0$: Y절편, $b_1$: 기울기(회귀계수)")

    with col2:
        st.success("🔸 **다중선형회귀 (Multiple Linear Regression)**\n여러 개의 독립변수($X_1, X_2, ...$)로 하나의 종속변수($y$)를 예측합니다.")
        st.latex(r"\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n")
        st.caption("$b_0$: Y절편, $b_1 \dots b_n$: 각 변수의 회귀계수")

    st.markdown("---")

    st.subheader("🎯 예측값, 잔차, 그리고 인과관계의 오해")
    st.markdown(
        """
    * **실제값($y$) vs 예측값($\hat{y}$)**: 실제 관측된 데이터 값과 모델이 회귀식으로 계산해낸 예측값입니다.
    * **잔차(Residual)**: $\text{실제값} - \text{예측값}$ 입니다. 회귀선과 데이터 점 사이의 수직 거리를 의미하며, 잔차가 0에 가까울수록 정밀한 모델입니다.
    * **⚠️ 상관관계 $\neq$ 인과관계**: 두 변수가 함께 움직이는 경향(상관관계)이 있다고 해서, 하나가 다른 하나의 직접적인 원인(인과관계)이 되는 것은 아닙니다! (예: 아이스크림 판매량과 수영장 사고 건수는 상관관계가 높지만, 원인은 여름철 기온 상승입니다.)
    """
    )

    with st.expander("❓ [탐구 질문 1] 학습 내용 점검하기"):
        st.markdown(
            """
        1. **독립변수(X)가 3개인 경우 단순선형회귀일까요, 다중선형회귀일까요?**
           * 정답: 독립변수가 2개 이상이므로 **다중선형회귀**입니다.
        2. **잔차가 양수(+)라면 실제값이 큰 것일까요, 예측값이 큰 것일까요?**
           * 정답: $\text{잔차} = \text{실제값} - \text{예측값}$ 이므로 잔차가 양수면 **실제값이 예측값보다 더 큽니다.**
        """
        )


# ------------------------------------------
# TAB 2: CSV 데이터 업로드
# ------------------------------------------
with tab2:
    st.subheader("📂 실습용 CSV 데이터 업로드 및 확인")

    col_up, col_sample = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "자신만의 CSV 파일을 업로드하세요 (UTF-8 / CP949 지원)", type=["csv"]
        )

    with col_sample:
        st.markdown("**샘플 데이터로 실습하기**")
        sample_df = generate_sample_data()
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 미세먼지 예제 CSV 다운로드",
            data=csv_bytes,
            file_name="pm25_sample_data.csv",
            mime="text/csv",
        )

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("성공적으로 CSV 파일을 읽어왔습니다!")
        except Exception as e:
            st.error(str(e))
    elif st.session_state["df"] is None:
        # 기본적으로 샘플 데이터 로드
        st.session_state["df"] = sample_df
        st.info("💡 업로드된 파일이 없어 '미세먼지 예제 데이터'를 기본 로드했습니다.")

    df = st.session_state["df"]

    if df is not None:
        is_valid, msg = validate_data(df)
        if not is_valid:
            st.error(msg)
        else:
            st.write(msg)

            st.markdown("### 📋 데이터 미리보기 (상위 5개 행)")
            st.dataframe(df.head(), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            with c1:
                st.markdown(
                    f"**🔢 수치형(숫자) 변수 ({len(num_cols)}개)**\n\n"
                    + ", ".join([f"`{c}`" for c in num_cols])
                )
            with c2:
                st.markdown(
                    f"**🔤 범주형/문자형 변수 ({len(cat_cols)}개)**\n\n"
                    + (", ".join([f"`{c}`" for c in cat_cols]) if cat_cols else "없음")
                )
            with c3:
                null_info = df.isnull().sum()
                has_null = null_info.sum() > 0
                st.markdown(f"**⚠️ 결측치(빈 값) 존재 여부**: {'있음' if has_null else '없음'}")
                if has_null:
                    st.dataframe(null_info[null_info > 0], use_container_width=True)

    with st.expander("❓ [탐구 질문 2] 데이터 확인 시 체크포인트"):
        st.markdown(
            """
        * 문자형(범주형) 데이터는 직접적인 선형회귀 입력값으로 사용할 수 있을까요?
        * 데이터에 결측치(Null)가 포함되어 있다면 인공지능 모델 학습 시 어떤 문제가 생길까요?
        """
        )


# ------------------------------------------
# TAB 3: 데이터 탐색
# ------------------------------------------
with tab3:
    st.subheader("🔍 데이터 탐색 (EDA) 및 변수 간 관계 확인")

    df = st.session_state.get("df", None)
    if df is None:
        st.warning("먼저 데이터 업로드 탭에서 데이터를 준비해주세요.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.markdown("### 1. 기술 통계량 요약")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

        st.markdown("---")
        st.markdown("### 2. 변수 분포 및 산점도 탐색")

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            var_x = st.selectbox(
                "X축 변수 선택", num_cols, index=0, key="eda_x"
            )
        with col_sel2:
            default_y_idx = 1 if len(num_cols) > 1 else 0
            var_y = st.selectbox(
                "Y축 변수 선택", num_cols, index=default_y_idx, key="eda_y"
            )

        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            fig_hist = px.histogram(
                df,
                x=var_x,
                title=f"[{var_x}] 변수의 히스토그램 (분포)",
                marginal="box",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_graph2:
            fig_scatter = px.scatter(
                df,
                x=var_x,
                y=var_y,
                title=f"[{var_x}] vs [{var_y}] 산점도",
                hover_data=df.columns,
                trendline="ols",
                trendline_color_override="red",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. 상관계수(Correlation Coefficient) 분석")

        corr_matrix = df[num_cols].corr().round(3)

        c_corr1, c_corr2 = st.columns([1, 1])
        with c_corr1:
            st.markdown("**수치형 변수 간 상관계수 표**")
            st.dataframe(corr_matrix, use_container_width=True)

        with c_corr2:
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="상관계수 히트맵",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        st.info(
            """
        💡 **상관계수 수치 이해하기**:
        * **+1.0에 가까움**: 강한 **양의 상관관계** (X가 늘어날 때 Y도 증가)
        * **-1.0에 가까움**: 강한 **음의 상관관계** (X가 늘어날 때 Y는 감소)
        * **0.0 근처**: 두 변수 간 선형적 관계가 거의 없음
        """
        )

        with st.expander("❓ [탐구 질문 3] 산점도와 상관계수 관찰하기"):
            st.markdown(
                f"""
            1. 선택하신 `{var_x}`와 `{var_y}`는 양의 관계인가요, 음의 관계인가요?
            2. 산점도의 데이터 점들이 붉은색 트렌드선 주변에 빽빽하게 모여 있나요, 아니면 넓게 흩어져 있나요?
            3. 유독 혼자 멀리 떨어져 있는 **이상치(Outlier)**가 관찰되나요?
            4. 두 변수의 상관계수가 높다면, 두 변수는 반드시 원인과 결과(인과관계) 관계일까요?
            """
            )


# ------------------------------------------
# TAB 4: 단순선형회귀
# ------------------------------------------
with tab4:
    st.subheader("📈 단순선형회귀 (Simple Linear Regression) 모델링")

    df = st.session_state.get("df", None)
    if df is None:
        st.warning("데이터가 준비되지 않았습니다.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            feature_x = st.selectbox(
                "독립변수 (X) 선택", num_cols, index=2, key="sim_x"
            )
        with c_s2:
            # y는 X와 다른 열 선택되도록
            y_options = [c for c in num_cols if c != feature_x]
            target_y = st.selectbox(
                "종속변수 (y) 선택",
                y_options,
                index=len(y_options) - 1,
                key="sim_y",
            )
        with c_s3:
            test_ratio = st.slider(
                "테스트 데이터 비율 (%)",
                10,
                40,
                20,
                step=5,
                key="sim_split",
            )

        # 데이터 클리닝 및 분리
        sub_df = df[[feature_x, target_y]].dropna()

        if len(sub_df) < 10:
            st.error("⚠️ 결측치 제거 후 유효 데이터가 10개 미만이므로 모델을 학습할 수 없습니다.")
        else:
            if len(sub_df) < 30:
                st.warning("⚠️ 데이터 수가 30개 미만으로 적습니다. 평가 결과 해석에 유의하세요.")

            X = sub_df[[feature_x]]
            y = sub_df[target_y]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio / 100.0, random_state=42
            )

            # 모델 학습
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred_test, n_features=1)

            coef = model.coef_[0]
            intercept = model.intercept_

            # 세션에 저장
            st.session_state["simple_model_results"] = {
                "X_col": feature_x,
                "y_col": target_y,
                "coef": coef,
                "intercept": intercept,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred_test,
                "X_test": X_test,
            }

            st.markdown("### 1. 학습 결과 및 회귀식")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("학습 데이터 수", f"{len(X_train)}개")
            m2.metric("테스트 데이터 수", f"{len(X_test)}개")
            m3.metric("기울기 ($b_1$)", f"{coef:.4f}")
            m4.metric("절편 ($b_0$)", f"{intercept:.4f}")

            # 회귀식 표기
            sign = "+" if intercept >= 0 else "-"
            st.success(
                f"📐 **학습된 단순회귀식**:  \n"
                f"$$\\hat{{\\text{{{target_y}}}}} = {coef:.4f} \\times \\text{{{feature_x}}} {sign} {abs(intercept):.4f}$$"
            )

            st.markdown(explain_coefficient(feature_x, coef, target_y))

            st.markdown("---")
            st.markdown("### 2. 회귀선 및 잔차 시각화")

            # 그래프 그리기
            fig_sim = go.Figure()

            # Train Data
            fig_sim.add_trace(
                go.Scatter(
                    x=X_train[feature_x],
                    y=y_train,
                    mode="markers",
                    name="Train Data",
                    marker=dict(color="blue", opacity=0.5),
                )
            )

            # Test Data
            fig_sim.add_trace(
                go.Scatter(
                    x=X_test[feature_x],
                    y=y_test,
                    mode="markers",
                    name="Test Data",
                    marker=dict(color="orange", size=8),
                )
            )

            # 회귀선
            x_range = np.linspace(X[feature_x].min(), X[feature_x].max(), 100)
            y_line = model.predict(pd.DataFrame({feature_x: x_range}))
            fig_sim.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_line,
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", width=2),
                )
            )

            # 테스트 데이터 잔차 선 표시
            for _, row in X_test.iterrows():
                x_val = row[feature_x]
                y_act = y_test.loc[row.name]
                y_pr = model.predict(pd.DataFrame({feature_x: [x_val]}))[0]
                fig_sim.add_trace(
                    go.Scatter(
                        x=[x_val, x_val],
                        y=[y_act, y_pr],
                        mode="lines",
                        line=dict(color="gray", width=1, dash="dot"),
                        showlegend=False,
                    )
                )

            fig_sim.update_layout(
                title=f"{feature_x}에 따른 {target_y} 회귀선 및 잔차(회색 점선)",
                xaxis_title=feature_x,
                yaxis_title=target_y,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            st.markdown("---")
            st.markdown("### 3. 실시간 예측 시뮬레이터")

            min_val = float(X[feature_x].min())
            max_val = float(X[feature_x].max())
            avg_val = float(X[feature_x].mean())

            user_input_x = st.number_input(
                f"새로운 [{feature_x}] 값을 입력하세요:",
                min_value=min_val - 10,
                max_value=max_val + 10,
                value=round(avg_val, 1),
            )

            pred_val = model.predict(pd.DataFrame({feature_x: [user_input_x]}))[
                0
            ]

            st.metric(
                label=f"예측된 [{target_y}] 값", value=f"{pred_val:.2f}"
            )

            if pred_val < 0:
                st.warning(
                    "⚠️ **선형회귀의 한계 안내**: 예측값이 음수로 계산되었습니다. 미세먼지나 수량 등 현실의 물리량은 음수가 될 수 없지만, 선형회귀식은 수학적 직선이므로 음수 결과가 나올 수 있습니다."
                )

            st.caption(
                "📌 *이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.*"
            )

        with st.expander("❓ [탐구 질문 4] 단순선형회귀 모델 탐구"):
            st.markdown(
                """
            1. 회귀선이 모든 데이터 점을 완벽하게 통과하나요? 그렇지 않은 이유는 무엇일까요?
            2. 테스트 데이터 비율을 늘리거나 줄였을 때 기울기와 절편은 어떻게 바뀌나요?
            3. 잔차선(회색 점선)의 길이가 길수록 모델의 예측 정확도는 높은 것일까요, 낮은 것일까요?
            """
            )


# ------------------------------------------
# TAB 5: 다중선형회귀
# ------------------------------------------
with tab5:
    st.subheader("🔢 다중선형회귀 (Multiple Linear Regression) 모델링")

    df = st.session_state.get("df", None)
    if df is None:
        st.warning("데이터가 준비되지 않았습니다.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_m1, c_m2 = st.columns([1, 2])
        with c_m1:
            target_y_multi = st.selectbox(
                "종속변수 (y) 선택", num_cols, index=len(num_cols) - 1, key="mul_y"
            )

        avail_x = [c for c in num_cols if c != target_y_multi]
        with c_m2:
            features_x_multi = st.multiselect(
                "독립변수들 (X) 선택 (2개 이상 선택 권장)",
                avail_x,
                default=avail_x[:2] if len(avail_x) >= 2 else avail_x,
                key="mul_x",
            )

        use_scaler = st.checkbox(
            "입력 변수 표준화(StandardScaler) 적용하기",
            value=False,
            help="변수들의 단위(Scale)가 서로 다를 때 회귀계수의 상대적 크기를 비교하기 위해 표준화를 수행합니다.",
        )

        if len(features_x_multi) < 2:
            st.error("⚠️ 다중선형회귀 분석을 위해 최소 2개 이상의 독립변수(X)를 선택해주세요.")
        else:
            sub_df_m = df[features_x_multi + [target_y_multi]].dropna()

            if len(sub_df_m) < 10:
                st.error("⚠️ 결측치 제거 후 유효 데이터가 부족합니다.")
            else:
                X_m = sub_df_m[features_x_multi]
                y_m = sub_df_m[target_y_multi]

                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_m, y_m, test_size=0.2, random_state=42
                )

                if use_scaler:
                    model_m = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("regressor", LinearRegression()),
                        ]
                    )
                    model_m.fit(X_train_m, y_train_m)
                    coefs = model_m.named_steps["regressor"].coef_
                    intercept_m = model_m.named_steps["regressor"].intercept_
                else:
                    model_m = LinearRegression()
                    model_m.fit(X_train_m, y_train_m)
                    coefs = model_m.coef_
                    intercept_m = model_m.intercept_

                y_pred_m = model_m.predict(X_test_m)
                metrics_m = calculate_metrics(
                    y_test_m, y_pred_m, n_features=len(features_x_multi)
                )

                # 세션 저장
                st.session_state["multi_model_results"] = {
                    "X_cols": features_x_multi,
                    "y_col": target_y_multi,
                    "coefs": coefs,
                    "intercept": intercept_m,
                    "metrics": metrics_m,
                    "y_test": y_test_m,
                    "y_pred": y_pred_m,
                    "is_scaled": use_scaler,
                }

                st.markdown("### 1. 회귀계수 분석")

                coef_df = pd.DataFrame(
                    {"변수": features_x_multi, "회귀계수": coefs}
                )

                c_m_res1, c_m_res2 = st.columns([1, 1])

                with c_m_res1:
                    st.dataframe(coef_df, use_container_width=True)
                    st.write(f"**Y 절편 ($b_0$)**: {intercept_m:.4f}")

                    st.warning(
                        "📌 **주의사항**: "
                        "다중선형회귀의 회귀계수는 **다른 입력 변수들이 일정하다고 고정했을 때** 해당 변수가 1 변할 때의 예측값 변화입니다.\n"
                        "또한 변수마다 단위(기온: ℃, 습도: %)가 다르면 단순 계수 크기만으로 중요도를 직접 비교할 수 없습니다."
                    )

                with c_m_res2:
                    fig_coef = px.bar(
                        coef_df,
                        x="변수",
                        y="회귀계수",
                        color="회귀계수",
                        title="변수별 회귀계수 크기 비교",
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

                st.markdown("---")
                st.markdown("### 2. 다중 모델 사용자 예측 테스트")
                st.write("각 변수의 값을 설정하고 예측 결과를 확인해보세요.")

                user_inputs = {}
                cols_input = st.columns(len(features_x_multi))
                for i, col in enumerate(features_x_multi):
                    with cols_input[i]:
                        mean_v = float(X_m[col].mean())
                        user_inputs[col] = st.number_input(
                            f"{col}", value=round(mean_v, 1), key=f"m_in_{col}"
                        )

                user_df = pd.DataFrame([user_inputs])
                m_pred_single = model_m.predict(user_df)[0]

                st.success(
                    f"🎯 **다중선형회귀 예측 [{target_y_multi}]**: **{m_pred_single:.2f}**"
                )

        with st.expander("❓ [탐구 질문 5] 다중선형회귀 탐구"):
            st.markdown(
                """
            1. 독립변수 개수를 늘릴 때 결정계수($R^2$)는 항상 증가하나요?
            2. 표준화(StandardScaler) 옵션을 켰을 때 회귀계수의 값은 어떻게 변하나요? 왜 그럴까요?
            """
            )


# ------------------------------------------
# TAB 6: 모델 평가 및 비교
# ------------------------------------------
with tab6:
    st.subheader("⚖️ 단순선형회귀 vs 다중선형회귀 성능 비교")

    sim_res = st.session_state.get("simple_model_results", None)
    mul_res = st.session_state.get("multi_model_results", None)

    if sim_res is None or mul_res is None:
        st.info("💡 탭 4(단순선형회귀)와 탭 5(다중선형회귀)에서 두 모델을 모두 학습시킨 후 방문해주세요.")
    else:
        st.markdown("### 1. 성능 평가 지표 비교표")

        comp_data = {
            "비교 항목": [
                "사용된 독립변수(X)",
                "결정계수 (R²)",
                "조정된 결정계수 (Adj R²)",
                "MAE (평균 절대 오차)",
                "MSE (평균 제곱 오차)",
                "RMSE (제곱근 평균 제곱 오차)",
            ],
            "단순선형회귀": [
                f"{sim_res['X_col']}",
                sim_res["metrics"]["R2"],
                sim_res["metrics"]["Adj_R2"],
                sim_res["metrics"]["MAE"],
                sim_res["metrics"]["MSE"],
                sim_res["metrics"]["RMSE"],
            ],
            "다중선형회귀": [
                f"{', '.join(mul_res['X_cols'])}",
                mul_res["metrics"]["R2"],
                mul_res["metrics"]["Adj_R2"],
                mul_res["metrics"]["MAE"],
                mul_res["metrics"]["MSE"],
                mul_res["metrics"]["RMSE"],
            ],
        }

        st.table(pd.DataFrame(comp_data))

        with st.expander("📖 평가 지표 쉬운 설명서 보기"):
            st.markdown(
                """
            * **MAE**: 오차의 절댓값 평균. 직관적으로 예측이 평균적으로 얼마나 틀렸는지 나타냅니다.
            * **MSE**: 오차 제곱의 평균. 큰 오차에 대해 더 강하게 벌점을 줍니다.
            * **RMSE**: MSE에 루트를 씌워 실제 정답($y$)과 같은 단위로 맞춘 지표입니다.
            * **$R^2$ (결정계수)**: 모델이 데이터의 전체 변동성을 몇 %나 설명하는지 나타냅니다. (1에 가까울수록 성능 우수)
            * **조정된 $R^2$**: 무의미한 변수를 무작위로 많이 넣었을 때 $R^2$가 쓸데없이 올라가는 현상을 보정한 지표입니다.
            """
            )

        st.markdown("---")
        st.markdown("### 2. 잔차 및 예측 성능 진단 그래프")

        t_col1, t_col2 = st.columns(2)

        # 다중 회귀 기준 시각화
        y_test_m = mul_res["y_test"]
        y_pred_m = mul_res["y_pred"]
        residuals_m = y_test_m - y_pred_m

        with t_col1:
            # 실제값 vs 예측값 산점도
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(
                go.Scatter(
                    x=y_test_m,
                    y=y_pred_m,
                    mode="markers",
                    name="예측 점",
                    marker=dict(color="purple"),
                )
            )

            # 1:1 기준선
            min_p = min(y_test_m.min(), y_pred_m.min())
            max_p = max(y_test_m.max(), y_pred_m.max())
            fig_act_pred.add_trace(
                go.Scatter(
                    x=[min_p, max_p],
                    y=[min_p, max_p],
                    mode="lines",
                    name="이상적 기준선(y=x)",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_act_pred.update_layout(
                title="실제값 vs 예측값 (다중선형회귀)",
                xaxis_title="실제값 (Actual)",
                yaxis_title="예측값 (Predicted)",
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with t_col2:
            # 잔차 산점도
            fig_res_scat = go.Figure()
            fig_res_scat.add_trace(
                go.Scatter(
                    x=y_pred_m,
                    y=residuals_m,
                    mode="markers",
                    marker=dict(color="green"),
                )
            )
            fig_res_scat.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res_scat.update_layout(
                title="잔차 산점도 (예측값 vs 잔차)",
                xaxis_title="예측값",
                yaxis_title="잔차 (실제값 - 예측값)",
            )
            st.plotly_chart(fig_res_scat, use_container_width=True)

        st.markdown("💡 **그래프 해석 가이드**")
        st.markdown(
            """
        * **실제값 vs 예측값**: 데이터 점들이 붉은 점선($y=x$)에 가까이 모여있을수록 뛰어난 모델입니다.
        * **잔차 산점도**: 0번 선을 기준으로 위아래에 무작위로 고르게 흩어져 있다면 선형 모델이 적절함을 의미합니다. (만약 곡선 패턴이나 부채꼴 모양이 보이면 비선형 모델을 고려해야 합니다.)
        """
        )

        with st.expander("❓ [탐구 질문 6] 최종 평가 및 모델 선택"):
            st.markdown(
                """
            1. 단순 회귀에 비해 다중 회귀의 $R^2$와 RMSE는 어떻게 개선되었나요?
            2. 변수의 개수가 무조건 많다고 해서 항상 최선의 모델이라고 할 수 있을까요? (설명력 vs 모델의 단순함)
            3. 이 모델을 실제 기상청 미세먼지 예보 시스템에 그대로 적용해도 될까요? 한계점은 무엇일까요?
            """
            )
