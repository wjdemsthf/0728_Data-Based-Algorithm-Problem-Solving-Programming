import streamlit as st
import random
import time
import pandas as pd
import requests
import altair as alt  # 🎨 막대 색상 지정을 위해 altair 사용

st.set_page_config(page_title="정렬 알고리즘 탐구 대시보드", layout="wide")

st.title("📊 정렬 알고리즘 개념기반 탐구 워크스페이스")
st.caption("단계별 알고리즘 동작 관찰, 파이썬 코드 분석, 및 성능 대시보드를 제공합니다.")

# ---------------------------------------------------------
# 1. 세션 상태 초기화
# ---------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = [25, 12, 45, 8, 30]
if "history" not in st.session_state:
    st.session_state.history = []  # 대시보드 기록용
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "steps" not in st.session_state:
    st.session_state.steps = []
if "calc_time_ms" not in st.session_state:
    st.session_state.calc_time_ms = 0.0

# ---------------------------------------------------------
# 2. 정렬 알고리즘 및 단계(Step) 제너레이터 (하이라이트 인덱스 포함)
# ---------------------------------------------------------
def generate_steps(arr, alg_type):
    a = arr.copy()
    n = len(a)
    steps = []
    comparisons = 0
    swaps = 0
    
    # 시작 상태
    steps.append({"arr": a.copy(), "comp": 0, "swap": 0, "highlight": [], "msg": "정렬 시작 전 상태입니다."})
    
    if alg_type == "버블 정렬":
        for i in range(n):
            for j in range(0, n - i - 1):
                comparisons += 1
                msg = f"인덱스 {j}({a[j]})와 {j+1}({a[j+1]}) 비교 중"
                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    swaps += 1
                    msg += " ➔ [교환 발생!]"
                # 현재 비교/교환 중인 두 인덱스를 highlight 배열에 담음
                steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [j, j + 1], "msg": msg})
                
    elif alg_type == "선택 정렬":
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                comparisons += 1
                if a[j] < a[min_idx]:
                    min_idx = j
                steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [i, j, min_idx], "msg": f"최소값 탐색 중 (현재 최소값: 인덱스 {min_idx})"})
                
            if min_idx != i:
                a[i], a[min_idx] = a[min_idx], a[i]
                swaps += 1
                steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [i, min_idx], "msg": f"인덱스 {i}와 최소값 위치({min_idx}) [교환 발생!]"})
                
    elif alg_type == "삽입 정렬":
        for i in range(1, n):
            key = a[i]
            j = i - 1
            steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [i], "msg": f"Key 값({key}) 선택"})
            while j >= 0:
                comparisons += 1
                if a[j] > key:
                    a[j + 1] = a[j]
                    swaps += 1
                    j -= 1
                    steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [j + 1, i], "msg": f"{a[j+1]}를 오른쪽으로 이동"})
                else:
                    break
            a[j + 1] = key
            steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [j + 1], "msg": f"Key({key}) 위치 삽입 완료"})

    steps.append({"arr": a.copy(), "comp": comparisons, "swap": swaps, "highlight": [], "msg": "🎉 정렬이 완료되었습니다!"})
    return steps

PYTHON_CODES = {
    "버블 정렬": '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',
    
    "선택 정렬": '''def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr''',
    
    "삽입 정렬": '''def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr'''
}

# 🎨 색상 차트 생성 함수 (Altair 활용)
def render_colored_chart(step_data):
    arr = step_data["arr"]
    highlight = step_data.get("highlight", [])
    
    # 상태별 색상 데이터 생성
    colors = []
    for idx in range(len(arr)):
        if idx in highlight:
            colors.append("비교/교환 중 🔴")
        else:
            colors.append("일반 데이터 🔵")
            
    df = pd.DataFrame({
        "인덱스": [f"[{i}]" for i in range(len(arr))],
        "값": arr,
        "상태": colors
    })
    
    # Altair 커스텀 막대 차트
    chart = alt.Chart(df).mark_bar(size=25).encode(
        x=alt.X("인덱스:N", sort=None, axis=alt.Axis(title="데이터 인덱스", labelAngle=0)),
        y=alt.Y("값:Q", axis=alt.Axis(title="값")),
        color=alt.Color(
            "상태:N",
            scale=alt.Scale(
                domain=["일반 데이터 🔵", "비교/교환 중 🔴"],
                range=["#4A90E2", "#FF4B4B"]  # 파란색, 빨간색
            ),
            legend=alt.Legend(title="구분")
        ),
        tooltip=["인덱스", "값", "상태"]
    ).properties(height=300)
    
    return chart

# ---------------------------------------------------------
# 3. 사이드바: 설정 영역
# ---------------------------------------------------------
st.sidebar.header("⚙️ 실험 데이터 및 조건 설정")
data_size = st.sidebar.slider("데이터 개수 (N)", 5, 20, 8)
data_state = st.sidebar.selectbox("데이터 상태", ["무작위 (Random)", "완전 역순 (Reverse)", "이미 정렬됨 (Sorted)"])
selected_alg = st.sidebar.selectbox("알고리즘 선택", ["버블 정렬", "선택 정렬", "삽입 정렬"])
anim_speed = st.sidebar.slider("자동 재생 속도 (초)", 0.05, 0.5, 0.1, 0.05)

if st.sidebar.button("🔄 데이터 생성 및 준비"):
    if data_state == "무작위 (Random)":
        st.session_state.data = random.sample(range(5, 99), data_size)
    elif data_state == "완전 역순 (Reverse)":
        st.session_state.data = list(range(data_size * 5, 0, -5))
    else:
        st.session_state.data = list(range(5, data_size * 5 + 1, 5))
    
    start_time = time.perf_counter()
    st.session_state.steps = generate_steps(st.session_state.data, selected_alg)
    end_time = time.perf_counter()
    
    st.session_state.calc_time_ms = (end_time - start_time) * 1000
    st.session_state.step_idx = 0
    st.rerun()

# ---------------------------------------------------------
# 4. 메인 화면: 2컬럼 레이아웃 (시각화 + 파이썬 코드)
# ---------------------------------------------------------
if not st.session_state.steps:
    start_time = time.perf_counter()
    st.session_state.steps = generate_steps(st.session_state.data, selected_alg)
    end_time = time.perf_counter()
    st.session_state.calc_time_ms = (end_time - start_time) * 1000

col_vis, col_code = st.columns([3, 2])

with col_vis:
    st.subheader(f"📌 {selected_alg} 시뮬레이션")
    
    # 조작 버튼 그룹
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    
    if ctrl_col1.button("◀ 이전 단계"):
        if st.session_state.step_idx > 0:
            st.session_state.step_idx -= 1
            
    if ctrl_col2.button("다음 단계 ▶"):
        if st.session_state.step_idx < len(st.session_state.steps) - 1:
            st.session_state.step_idx += 1
            
    if ctrl_col3.button("🔄 처음으로"):
        st.session_state.step_idx = 0
        
    play_all = ctrl_col4.button("▶️ 끝까지 자동 재생", type="primary")

    # 끝까지 자동 재생 로직 (색상 하이라이트 반영)
    if play_all:
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        info_placeholder = st.empty()
        
        for idx in range(st.session_state.step_idx, len(st.session_state.steps)):
            st.session_state.step_idx = idx
            step_data = st.session_state.steps[idx]
            
            # 커스텀 색상 차트 실시간 렌더링
            chart = render_colored_chart(step_data)
            chart_placeholder.altair_chart(chart, use_container_width=True)
            info_placeholder.info(f"**Step {idx} / {len(st.session_state.steps)-1}:** {step_data['msg']}")
            progress_bar.progress((idx + 1) / len(st.session_state.steps))
            
            time.sleep(anim_speed)
        st.rerun()

    # 현재 단계 정보 및 시각화 표시
    curr_step = st.session_state.steps[st.session_state.step_idx]
    max_step = len(st.session_state.steps) - 1
    
    st.progress((st.session_state.step_idx + 1) / len(st.session_state.steps))
    st.info(f"**Step {st.session_state.step_idx} / {max_step}:** {curr_step['msg']}")

    # 🎨 색상이 반영된 막대 차트 출력
    st.altair_chart(render_colored_chart(curr_step), use_container_width=True)
    
    m1, m2 = st.columns(2)
    m1.metric("현재 비교 횟수", f"{curr_step['comp']} 회")
    m2.metric("현재 교환 횟수", f"{curr_step['swap']} 회")

    # 정렬 완주시 선택적 데이터 기록 버튼
    st.divider()
    if st.session_state.step_idx == max_step:
        st.success("🎉 정렬 과정을 모두 마쳤습니다! 아래 버튼을 눌러 대시보드에 결과를 기록할 수 있습니다.")
        if st.button("📊 현재 정렬 결과를 대시보드에 기록하기"):
            new_record = {
                "알고리즘": selected_alg,
                "데이터 조건": data_state,
                "데이터 크기(N)": data_size,
                "총 비교 횟수": curr_step["comp"],
                "총 교환 횟수": curr_step["swap"],
                "순수 연산 시간": f"{st.session_state.calc_time_ms:.4f} ms"
            }
            st.session_state.history.append(new_record)
            st.toast("대시보드에 결과가 성공적으로 추가되었습니다!", icon="✅")
            time.sleep(0.5)
            st.rerun()
    else:
        st.caption("💡 **안내:** 정렬을 끝까지 마무리하면 대시보드 기록 버튼이 활성화됩니다.")

with col_code:
    st.subheader("💻 알고리즘 파이썬 코드")
    st.code(PYTHON_CODES[selected_alg], language="python")
    st.caption("💡 **Tip:** 파이썬 코드에서 값의 비교(`>`)와 위치 교환(`a, b = b, a`)이 이루어지는 위치를 관찰해 보세요.")

st.divider()

# ---------------------------------------------------------
# 5. 성능 비교 대시보드
# ---------------------------------------------------------
st.header("📈 알고리즘 성능 비교 대시보드")
st.caption("학생이 정렬을 마치고 스스로 기록한 결과들이 저장되는 대시보드입니다.")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    
    if st.button("🗑️ 대시보드 기록 초기화"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("아직 대시보드에 기록된 실험 결과가 없습니다. 정렬을 끝까지 감상한 후 기록해 보세요!")

st.divider()

# ---------------------------------------------------------
# 6. 개념 기반 탐구 답안 제출 (Google Apps Script 연동)
# ---------------------------------------------------------
st.header("📝 탐구 활동지 답안 제출")

GAS_WEBHOOK_URL = "https://script.google.com/macros/s/YOUR_SCRIPT_ID_HERE/exec"

with st.form(key="student_answer_form"):
    student_info = st.text_input("학번 및 이름 (예: 20101 홍길동)")
    q1 = st.text_area("1. [Fact] 선택한 알고리즘에서 데이터가 '완전 역순'일 때 비교 횟수와 교환 횟수의 특징은 무엇인가요?")
    q2 = st.text_area("2. [Concept] '이미 정렬된 데이터'에서 가장 빠른 효율을 보인 알고리즘과 그 이유를 설명하세요.")
    q3 = st.text_area("3. [Debatable] 실행 속도가 다소 느리더라도 코드가 단순한 알고리즘이 더 우수하다고 볼 수 있는 상황은 언제일까요?")
    
    submitted = st.form_submit_button("🚀 답안 구글 시트로 제출하기")
    
    if submitted:
        if not student_info:
            st.error("학번과 이름을 꼭 입력해 주세요!")
        else:
            last_history = st.session_state.history[-1] if st.session_state.history else {}
            
            payload = {
                "student_info": student_info,
                "algorithm": last_history.get("알고리즘", "기록 없음"),
                "data_state": last_history.get("데이터 조건", "-"),
                "data_size": last_history.get("데이터 크기(N)", "-"),
                "comp_count": last_history.get("총 비교 횟수", 0),
                "swap_count": last_history.get("총 교환 횟수", 0),
                "calc_time": last_history.get("순수 연산 시간", "-"),
                "q1": q1,
                "q2": q2,
                "q3": q3
            }
            
            try:
                response = requests.post(GAS_WEBHOOK_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    st.success(f"🎉 {student_info} 학생의 답안이 구글 시트로 성공적으로 제출되었습니다!")
                    st.balloons()
                else:
                    st.error("구글 시트 전송 중 오류가 발생했습니다.")
            except Exception as e:
                st.error(f"전송 실패: {e}")
