import streamlit as st
import random
import time
import pandas as pd

st.set_page_config(page_title="정렬 알고리즘 탐구 시뮬레이터", layout="wide")

st.title("📊 정렬 알고리즘 탐구 시뮬레이터")
st.caption("데이터 상태에 따른 버블, 선택, 삽입 정렬의 효율성(비교/교환 횟수)을 비교해 봅시다.")

# ---------------------------------------------------------
# 사이드바: 설정 컨트롤
# ---------------------------------------------------------
st.sidebar.header("⚙️ 실험 설정")
data_size = st.sidebar.slider("데이터 개수 (N)", min_value=5, max_value=30, value=10, step=1)
data_state = st.sidebar.selectbox(
    "데이터 초기 상태 선택",
    ["무작위 (Random)", "완전 역순 (Reverse Sorted)", "이미 정렬됨 (Already Sorted)"]
)
speed = st.sidebar.slider("애니메이션 속도 (초)", min_value=0.01, max_value=0.5, value=0.1, step=0.05)

# 데이터 생성 함수
def generate_data(size, state):
    if state == "무작위 (Random)":
        arr = random.sample(range(1, 100), size)
    elif state == "완전 역순 (Reverse Sorted)":
        arr = list(range(size * 3, 0, -3))
    else:  # 이미 정렬됨
        arr = list(range(3, size * 3 + 1, 3))
    return arr

# ---------------------------------------------------------
# 정렬 알고리즘 제너레이터 함수
# ---------------------------------------------------------
def bubble_sort(arr):
    a = arr.copy()
    n = len(a)
    comparisons = 0
    swaps = 0
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swaps += 1
                yield a.copy(), j, j + 1, comparisons, swaps
            else:
                yield a.copy(), j, j + 1, comparisons, swaps

def selection_sort(arr):
    a = arr.copy()
    n = len(a)
    comparisons = 0
    swaps = 0
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if a[j] < a[min_idx]:
                min_idx = j
            yield a.copy(), i, j, comparisons, swaps
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            swaps += 1
            yield a.copy(), i, min_idx, comparisons, swaps

def insertion_sort(arr):
    a = arr.copy()
    n = len(a)
    comparisons = 0
    swaps = 0
    for i in range(1, n):
        key = a[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                swaps += 1
                j -= 1
                yield a.copy(), j + 1, i, comparisons, swaps
            else:
                break
        a[j + 1] = key
        yield a.copy(), j + 1, i, comparisons, swaps

# ---------------------------------------------------------
# 메인 영역 UI
# ---------------------------------------------------------
if "data" not in st.session_state or st.sidebar.button("🔄 데이터 새로고침"):
    st.session_state.data = generate_data(data_size, data_state)

st.subheader(f"📌 현재 데이터 상태: {data_state}")
st.write(f"**원본 데이터:** `{st.session_state.data}`")

alg_option = st.radio("실행할 알고리즘 선택", ["버블 정렬 (Bubble)", "선택 정렬 (Selection)", "삽입 정렬 (Insertion)"], horizontal=True)

start_btn = st.button("🚀 정렬 시뮬레이션 시작", type="primary")

chart_holder = st.empty()
metrics_holder = st.empty()

if start_btn:
    data = st.session_state.data
    if alg_option.startswith("버블"):
        gen = bubble_sort(data)
    elif alg_option.startswith("선택"):
        gen = selection_sort(data)
    else:
        gen = insertion_sort(data)

    for step_arr, active_idx1, active_idx2, comp, swap in gen:
        df = pd.DataFrame({"값": step_arr, "인덱스": [str(i) for i in range(len(step_arr))]})
        
        # 차트 시각화
        chart_holder.bar_chart(df, x="인덱스", y="값", height=300)
        
        # 실시간 통계 표시
        with metrics_holder.container():
            col1, col2 = st.columns(2)
            col1.metric("🔍 총 비교 횟수", f"{comp} 회")
            col2.metric("🔄 총 교환 횟수", f"{swap} 회")
            
        time.sleep(speed)
        
    st.success("✅ 정렬 완) 완료되었습니다!")

st.divider()

# ---------------------------------------------------------
# 백워드 개념기반 탐구 질문 섹션
# ---------------------------------------------------------
st.header("📝 개념 기반 탐구 질문")

q1 = st.text_input("1. [Fact] '완전 역순' 데이터를 정렬할 때 선택한 알고리즘의 비교 횟수와 교환 횟수는 각각 몇 회인가요?")
q2 = st.text_area("2. [Concept] '이미 정렬된' 데이터에서는 어떤 알고리즘이 가장 효율적인가요? 그 이유는 무엇인가요?")
q3 = st.text_area("3. [Debatable] 코드가 복잡하지만 빠른 알고리즘과, 속도는 다소 느리지만 이해하기 쉬운 알고리즘 중 어떤 것이 더 우수한 알고리즘이라고 생각하나요?")

if st.button("답안 제출하기"):
    st.success("학습 활동 답안이 성공적으로 기록되었습니다!")
