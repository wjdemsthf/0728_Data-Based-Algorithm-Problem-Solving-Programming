import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit.components.v1 as components

st.set_page_config(
    page_title="머신러닝 플레이그라운드 - 선형 회귀",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Streamlit UI refinement
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

HTML_PLAYGROUND_CODE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>머신러닝 플레이그라운드 - 선형 회귀</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Noto Sans KR', sans-serif; background-color: #f8fafc; user-select: none; }
        canvas { touch-action: none; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        .pulse-subtle { animation: pulse-subtle 2s infinite; }
        @keyframes pulse-subtle { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    </style>
</head>
<body class="text-slate-800 bg-slate-50 min-h-screen flex flex-col p-2">

    <!-- Main Container -->
    <main class="max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5">

        <!-- Left Column: Interactive Canvas & Mode Controls -->
        <section class="lg:col-span-7 flex flex-col gap-4">
            
            <!-- Canvas Toolbar -->
            <div class="bg-white p-3.5 rounded-2xl shadow-sm border border-slate-200 flex flex-wrap items-center justify-between gap-3">
                <div class="flex items-center space-x-2 bg-slate-100 p-1 rounded-xl">
                    <button id="modeAddBtn" class="px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 bg-indigo-600 text-white shadow-sm">
                        <i class="fa-solid fa-plus-circle"></i> 데이터 점 찍기
                    </button>
                    <button id="modePredictBtn" class="px-3.5 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-indigo-600 transition-all flex items-center gap-2">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Y값 예측하기
                    </button>
                </div>

                <!-- Display Options -->
                <div class="flex items-center space-x-4 text-xs font-medium text-slate-600">
                    <label class="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" id="showErrorLinesToggle" checked class="w-4 h-4 text-indigo-600 rounded">
                        <span>오차(잔차) 표시</span>
                    </label>
                    <label class="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" id="showGridToggle" checked class="w-4 h-4 text-indigo-600 rounded">
                        <span>눈금 격자</span>
                    </label>
                </div>
            </div>

            <!-- Canvas Card -->
            <div class="bg-white rounded-2xl shadow-md border border-slate-200 overflow-hidden flex flex-col relative">
                <!-- Instruction Overlay Top Bar -->
                <div class="bg-slate-800 text-slate-100 text-xs px-4 py-2 flex justify-between items-center">
                    <span id="canvasGuideText" class="flex items-center gap-2 font-medium">
                        <i class="fa-solid fa-hand-pointer text-yellow-400"></i>
                        캔버스 아무 곳이나 클릭(터치)하여 데이터 점을 추가하세요!
                    </span>
                    <span id="dataCountBadge" class="bg-indigo-500/30 text-indigo-200 font-mono text-xs px-2.5 py-0.5 rounded-full border border-indigo-400/30">
                        점 개수: 0개
                    </span>
                </div>

                <!-- Canvas Wrapper -->
                <div class="relative w-full aspect-[4/3] bg-slate-900 cursor-crosshair overflow-hidden">
                    <canvas id="regressionCanvas" class="w-full h-full block"></canvas>
                    
                    <!-- Floating Interactive Tooltip for Predictions -->
                    <div id="predictionTooltip" class="hidden absolute pointer-events-none bg-indigo-900/90 text-white text-xs p-2.5 rounded-xl shadow-lg backdrop-blur-sm border border-indigo-500/40">
                        <div><span class="text-indigo-300 font-semibold">입력 X:</span> <span id="tooltipX" class="font-mono">0</span></div>
                        <div><span class="text-emerald-300 font-semibold">예측 Y:</span> <span id="tooltipY" class="font-mono">0</span></div>
                    </div>
                </div>

                <!-- Canvas Bottom Actions & Status Bar -->
                <div class="p-2.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between gap-2 flex-wrap">
                    <div class="text-xs text-slate-500 flex items-center gap-3">
                        <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block"></span> 데이터 점</span>
                        <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-500 inline-block"></span> 추정 회귀선</span>
                        <span class="flex items-center gap-1"><span class="w-3 h-[1px] bg-rose-400 inline-block"></span> 오차(MSE)</span>
                    </div>

                    <div class="flex items-center gap-2">
                        <button id="clearBtn" class="px-3 py-1 bg-rose-50 hover:bg-rose-100 text-rose-600 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1 border border-rose-200">
                            <i class="fa-solid fa-trash-can"></i> 전체 지우기
                        </button>
                    </div>
                </div>
            </div>

            <!-- Educational Quick Concept -->
            <div class="bg-indigo-50/70 rounded-2xl p-3.5 border border-indigo-100 text-xs text-indigo-950">
                <h3 class="font-bold text-indigo-900 flex items-center gap-1.5 mb-1">
                    <i class="fa-solid fa-lightbulb text-amber-500"></i> 선형 회귀(Linear Regression) 핵심
                </h3>
                <p class="text-slate-700 leading-relaxed">
                    • <strong>회귀선 ($y = wx + b$)</strong>: 점들의 경향을 가장 잘 대표하는 직선입니다.<br>
                    • <strong>오차(MSE)</strong>: 점들과 직선 사이 수직 오차의 제곱 평균입니다. (0에 가까울수록 정밀)<br>
                    • <strong>결정계수 ($R^2$)</strong>: 모델의 설명력 (1.0에 가까울수록 완벽한 추정)
                </p>
            </div>

        </section>

        <!-- Right Column: Controls & Metrics -->
        <section class="lg:col-span-5 flex flex-col gap-4">

            <!-- Presets Section -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                <div class="flex items-center justify-between mb-2.5">
                    <h2 class="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
                        <i class="fa-solid fa-database text-indigo-500"></i> 예시 데이터셋 & 외부 파일
                    </h2>
                    <button id="openCsvModalBtn" class="px-2.5 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-bold rounded-lg text-xs transition-all flex items-center gap-1 border border-indigo-200">
                        <i class="fa-solid fa-file-csv text-indigo-600"></i> CSV 불러오기
                    </button>
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <button data-preset="study" class="preset-btn px-2.5 py-2 bg-slate-50 hover:bg-indigo-50 border border-slate-200 rounded-xl text-left transition-all">
                        <div class="font-bold text-xs text-slate-800">📚 공부 vs 시험점수</div>
                        <div class="text-[10px] text-slate-500">양의 상관관계</div>
                    </button>
                    <button data-preset="caffeine" class="preset-btn px-2.5 py-2 bg-slate-50 hover:bg-indigo-50 border border-slate-200 rounded-xl text-left transition-all">
                        <div class="font-bold text-xs text-slate-800">☕ 카페인 vs 수면</div>
                        <div class="text-[10px] text-slate-500">음의 상관관계</div>
                    </button>
                    <button data-preset="noisy" class="preset-btn px-2.5 py-2 bg-slate-50 hover:bg-indigo-50 border border-slate-200 rounded-xl text-left transition-all">
                        <div class="font-bold text-xs text-slate-800">🎲 노이즈 데이터</div>
                        <div class="text-[10px] text-slate-500">무작위 데이터</div>
                    </button>
                    <button data-preset="curve" class="preset-btn px-2.5 py-2 bg-slate-50 hover:bg-indigo-50 border border-slate-200 rounded-xl text-left transition-all">
                        <div class="font-bold text-xs text-slate-800">🎢 2차 곡선 데이터</div>
                        <div class="text-[10px] text-slate-500">선형 모델의 한계</div>
                    </button>
                </div>
                <div id="csvDatasetBadge" class="hidden mt-2 p-2 bg-indigo-50 border border-indigo-200 rounded-xl text-xs text-indigo-800 flex items-center justify-between">
                    <span class="font-semibold truncate"><i class="fa-solid fa-table"></i> <span id="csvFileNameText">data.csv</span></span>
                    <span id="csvColInfoText" class="text-[10px] text-indigo-600 font-mono">X: Col1, Y: Col2</span>
                </div>
            </div>

            <!-- Model Evaluation Metrics Dashboard -->
            <div class="bg-slate-900 text-white p-4 rounded-2xl shadow-md border border-slate-800">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-xs font-bold text-indigo-300 tracking-wider uppercase flex items-center gap-1.5">
                        <i class="fa-solid fa-gauge-high"></i> 실시간 평가 지표
                    </h2>
                    <span id="trainingStatusTag" class="text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                        학습 완료 (최적 해)
                    </span>
                </div>

                <div class="grid grid-cols-2 gap-2.5">
                    <div class="bg-slate-800/80 p-3 rounded-xl border border-slate-700">
                        <div class="text-[11px] text-slate-400 font-medium mb-0.5">기울기 ($w$, Slope)</div>
                        <div id="metricSlope" class="text-lg font-mono font-bold text-yellow-400">0.000</div>
                    </div>
                    <div class="bg-slate-800/80 p-3 rounded-xl border border-slate-700">
                        <div class="text-[11px] text-slate-400 font-medium mb-0.5">절편 ($b$, Intercept)</div>
                        <div id="metricIntercept" class="text-lg font-mono font-bold text-cyan-400">0.000</div>
                    </div>
                    <div class="bg-slate-800/80 p-3 rounded-xl border border-slate-700">
                        <div class="text-[11px] text-slate-400 font-medium mb-0.5">평균제곱오차 (MSE)</div>
                        <div id="metricMSE" class="text-lg font-mono font-bold text-rose-400">0.00</div>
                    </div>
                    <div class="bg-slate-800/80 p-3 rounded-xl border border-slate-700">
                        <div class="text-[11px] text-slate-400 font-medium mb-0.5">결정계수 ($R^2$)</div>
                        <div id="metricR2" class="text-lg font-mono font-bold text-emerald-400">0.000</div>
                    </div>
                </div>

                <div class="mt-3 pt-2.5 border-t border-slate-800 text-center">
                    <div class="text-[10px] text-slate-400 mb-0.5">현재 회귀 방정식</div>
                    <div id="equationText" class="font-mono text-xs text-indigo-200 font-semibold bg-slate-950 py-1 px-2 rounded-lg border border-slate-800 inline-block w-full">
                        y = 0.00x + 0.00
                    </div>
                </div>
            </div>

            <!-- Gradient Descent Controls -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-slate-200 flex flex-col gap-3">
                <h2 class="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center justify-between">
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-sliders text-indigo-500"></i> 경사하강법 학습 시뮬레이션</span>
                    <span id="epochDisplay" class="font-mono text-indigo-600 text-[11px] bg-indigo-50 px-2 py-0.5 rounded">Epoch: 0 / 100</span>
                </h2>

                <div class="space-y-1">
                    <div class="flex justify-between text-xs font-medium text-slate-600">
                        <span>학습 횟수 (Epochs)</span>
                        <span id="epochValueText" class="font-mono text-indigo-600">100회</span>
                    </div>
                    <input type="range" id="epochSlider" min="1" max="500" value="100" step="1" class="w-full h-1.5 bg-slate-200 rounded appearance-none cursor-pointer accent-indigo-600">
                </div>

                <div class="space-y-1">
                    <div class="flex justify-between text-xs font-medium text-slate-600">
                        <span>학습률 (Learning Rate)</span>
                        <span id="lrValueText" class="font-mono text-indigo-600">0.01</span>
                    </div>
                    <input type="range" id="lrSlider" min="0.001" max="0.05" value="0.01" step="0.001" class="w-full h-1.5 bg-slate-200 rounded appearance-none cursor-pointer accent-indigo-600">
                </div>

                <div class="grid grid-cols-2 gap-2 pt-1">
                    <button id="trainStepBtn" class="py-2 px-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold text-xs shadow transition-all flex items-center justify-center gap-1.5">
                        <i class="fa-solid fa-play"></i> 경사하강법 학습
                    </button>
                    <button id="instantFitBtn" class="py-2 px-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-bold text-xs shadow transition-all flex items-center justify-center gap-1.5">
                        <i class="fa-solid fa-bolt"></i> 최적해 계산
                    </button>
                </div>
            </div>

            <!-- Predict Tool Widget -->
            <div class="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                <h2 class="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <i class="fa-solid fa-calculator text-indigo-500"></i> X값으로 직접 Y값 예측하기
                </h2>
                <div class="flex items-center gap-2">
                    <div class="relative flex-grow">
                        <span class="absolute left-3 top-2 text-xs font-bold text-slate-400">X =</span>
                        <input type="number" id="manualXInput" min="0" max="100" value="50" placeholder="0 ~ 100" class="w-full pl-9 pr-3 py-1.5 bg-slate-50 border border-slate-300 rounded-xl text-xs font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    </div>
                    <button id="manualPredictBtn" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-900 text-white font-bold text-xs rounded-xl transition-colors">
                        예측 실행
                    </button>
                </div>
                <div id="manualPredictResult" class="mt-2 text-xs bg-slate-50 p-2 rounded-xl border border-slate-200 text-slate-700 font-medium">
                    X값을 입력하거나 캔버스를 클릭해보세요!
                </div>
            </div>

        </section>

    </main>

    <!-- CSV Upload Modal -->
    <div id="csvModal" class="hidden fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-white rounded-2xl shadow-xl border border-slate-200 max-w-md w-full overflow-hidden flex flex-col">
            <div class="bg-indigo-600 text-white px-4 py-3 flex items-center justify-between">
                <h3 class="font-bold text-sm flex items-center gap-2"><i class="fa-solid fa-file-csv text-amber-300"></i> CSV 데이터 불러오기</h3>
                <button id="closeCsvModalBtn" class="text-indigo-200 hover:text-white text-base"><i class="fa-solid fa-xmark"></i></button>
            </div>
            <div class="p-4 space-y-3 text-xs">
                <div id="dropZone" class="border-2 border-dashed border-indigo-200 bg-indigo-50/50 hover:bg-indigo-50 rounded-xl p-5 text-center cursor-pointer">
                    <input type="file" id="csvFileInput" accept=".csv,.txt" class="hidden">
                    <i class="fa-solid fa-cloud-arrow-up text-2xl text-indigo-500 mb-1"></i>
                    <p class="font-bold text-slate-700">CSV 파일을 클릭하여 선택하세요</p>
                </div>
                <div id="csvConfigArea" class="hidden space-y-3 pt-2 border-t border-slate-100">
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="block font-bold text-slate-600 mb-1">X축 열</label>
                            <select id="selectXCol" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-1.5 text-xs"></select>
                        </div>
                        <div>
                            <label class="block font-bold text-slate-600 mb-1">Y축 열</label>
                            <select id="selectYCol" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-1.5 text-xs"></select>
                        </div>
                    </div>
                    <label class="flex items-center gap-2 cursor-pointer bg-slate-50 p-2 rounded-lg border">
                        <input type="checkbox" id="autoScaleCsvToggle" checked class="w-4 h-4 text-indigo-600 rounded">
                        <span class="font-bold text-slate-700">캔버스 스케일 맞춤 (Min-Max 정규화)</span>
                    </label>
                </div>
            </div>
            <div class="bg-slate-50 px-4 py-2.5 border-t border-slate-200 flex justify-end gap-2">
                <button id="cancelCsvBtn" class="px-3 py-1.5 bg-slate-200 text-slate-700 rounded-lg font-bold text-xs">취소</button>
                <button id="applyCsvBtn" disabled class="px-3 py-1.5 bg-indigo-600 disabled:opacity-50 text-white rounded-lg font-bold text-xs">적용하기</button>
            </div>
        </div>
    </div>

    <!-- JavaScript Application Code -->
    <script>
        const state = {
            points: [], mode: 'add', w: 0, b: 0, mse: 0, r2: 0,
            epochsTarget: 100, learningRate: 0.01, currentEpoch: 0,
            isTraining: false, animationId: null, predictionPoint: null,
            showErrorLines: true, showGrid: true, csvData: null
        };

        const canvas = document.getElementById('regressionCanvas');
        const ctx = canvas.getContext('2d');
        const padding = { top: 30, right: 30, bottom: 30, left: 40 };

        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            draw();
        }

        function dataToCanvas(x, y) {
            const rect = canvas.getBoundingClientRect();
            const width = rect.width - padding.left - padding.right;
            const height = rect.height - padding.top - padding.bottom;
            const cx = padding.left + (x / 100) * width;
            const cy = padding.top + (1 - y / 100) * height;
            return { x: cx, y: cy };
        }

        function canvasToData(cx, cy) {
            const rect = canvas.getBoundingClientRect();
            const width = rect.width - padding.left - padding.right;
            const height = rect.height - padding.top - padding.bottom;
            let x = Math.max(0, Math.min(100, ((cx - padding.left) / width) * 100));
            let y = Math.max(0, Math.min(100, (1 - (cy - padding.top) / height) * 100));
            return { x, y };
        }

        function calculateAnalyticalRegression() {
            const n = state.points.length;
            if (n < 2) {
                state.w = 0;
                state.b = n === 1 ? state.points[0].y : 0;
                calculateMetrics();
                return;
            }
            let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            for (let p of state.points) {
                sumX += p.x; sumY += p.y; sumXY += p.x * p.y; sumXX += p.x * p.x;
            }
            const meanX = sumX / n, meanY = sumY / n;
            const denom = sumXX - n * meanX * meanX;
            state.w = Math.abs(denom) < 1e-7 ? 0 : (sumXY - n * meanX * meanY) / denom;
            state.b = meanY - state.w * meanX;
            calculateMetrics();
        }

        function calculateMetrics() {
            const n = state.points.length;
            if (n === 0) { state.mse = 0; state.r2 = 0; updateMetricsUI(); return; }
            let sumY = 0;
            for (let p of state.points) sumY += p.y;
            const meanY = sumY / n;
            let ssTot = 0, ssRes = 0;
            for (let p of state.points) {
                const predY = state.w * p.x + state.b;
                const err = p.y - predY;
                ssRes += err * err;
                ssTot += (p.y - meanY) * (p.y - meanY);
            }
            state.mse = ssRes / n;
            state.r2 = ssTot === 0 ? 1 : Math.max(0, 1 - (ssRes / ssTot));
            updateMetricsUI();
        }

        function loadPreset(type) {
            state.points = []; state.predictionPoint = null;
            document.getElementById('csvDatasetBadge').classList.add('hidden');
            if (state.isTraining) cancelAnimationFrame(state.animationId);
            state.isTraining = false;
            const count = 25;
            for (let i = 0; i < count; i++) {
                const x = 10 + (i / count) * 80;
                let y = 50;
                if (type === 'study') y = Math.min(100, Math.max(0, 0.75 * x + 15 + (Math.random() - 0.5) * 16));
                else if (type === 'caffeine') y = Math.min(100, Math.max(0, -0.7 * x + 85 + (Math.random() - 0.5) * 18));
                else if (type === 'noisy') y = Math.random() * 85 + 5;
                else if (type === 'curve') y = Math.min(100, Math.max(10, 15 + Math.pow((x - 50)/30, 2) * 22 + (Math.random() - 0.5) * 10));
                state.points.push({ x, y });
            }
            calculateAnalyticalRegression();
            draw();
        }

        function runGradientDescentStep() {
            if (state.points.length === 0) return;
            const n = state.points.length;
            let dw = 0, db = 0;
            for (let p of state.points) {
                const xNorm = p.x / 100, yNorm = p.y / 100;
                const predYNorm = state.w * xNorm + (state.b / 100);
                const err = predYNorm - yNorm;
                dw += (2 / n) * err * xNorm;
                db += (2 / n) * err;
            }
            state.w = state.w - state.learningRate * dw * 100;
            state.b = state.b - state.learningRate * db * 100;
            state.currentEpoch++;
            calculateMetrics();
            draw();
            if (state.currentEpoch < state.epochsTarget && state.isTraining) {
                document.getElementById('epochDisplay').textContent = `Epoch: ${state.currentEpoch} / ${state.epochsTarget}`;
                state.animationId = requestAnimationFrame(runGradientDescentStep);
            } else {
                state.isTraining = false;
                document.getElementById('trainStepBtn').innerHTML = '<i class="fa-solid fa-play"></i> 경사하강법 학습';
            }
        }

        function draw() {
            const rect = canvas.getBoundingClientRect();
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, rect.width, rect.height);

            if (state.showGrid) {
                ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
                for (let i = 0; i <= 10; i++) {
                    const pos = dataToCanvas(i * 10, i * 10);
                    ctx.beginPath(); ctx.moveTo(pos.x, padding.top); ctx.lineTo(pos.x, rect.height - padding.bottom); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(padding.left, pos.y); ctx.lineTo(rect.width - padding.right, pos.y); ctx.stroke();
                }
            }

            ctx.strokeStyle = '#475569'; ctx.lineWidth = 1.5;
            ctx.strokeRect(padding.left, padding.top, rect.width - padding.left - padding.right, rect.height - padding.top - padding.bottom);

            if (state.showErrorLines && state.points.length > 0) {
                ctx.strokeStyle = 'rgba(251, 113, 133, 0.5)'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
                for (let p of state.points) {
                    const ptCanvas = dataToCanvas(p.x, p.y);
                    const lineCanvas = dataToCanvas(p.x, state.w * p.x + state.b);
                    ctx.beginPath(); ctx.moveTo(ptCanvas.x, ptCanvas.y); ctx.lineTo(lineCanvas.x, lineCanvas.y); ctx.stroke();
                }
                ctx.setLineDash([]);
            }

            if (state.points.length > 0 || state.w !== 0 || state.b !== 0) {
                const p1 = dataToCanvas(0, state.b);
                const p2 = dataToCanvas(100, state.w * 100 + state.b);
                ctx.strokeStyle = '#10b981'; ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
            }

            if (state.predictionPoint) {
                const pt = state.predictionPoint;
                const cPt = dataToCanvas(pt.x, state.w * pt.x + state.b);
                ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(cPt.x, rect.height - padding.bottom); ctx.lineTo(cPt.x, cPt.y); ctx.lineTo(padding.left, cPt.y); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#c084fc'; ctx.beginPath(); ctx.arc(cPt.x, cPt.y, 6, 0, Math.PI * 2); ctx.fill();
            }

            for (let p of state.points) {
                const pos = dataToCanvas(p.x, p.y);
                ctx.fillStyle = '#6366f1'; ctx.beginPath(); ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.stroke();
            }
        }

        function updateMetricsUI() {
            document.getElementById('metricSlope').textContent = state.w.toFixed(3);
            document.getElementById('metricIntercept').textContent = state.b.toFixed(3);
            document.getElementById('metricMSE').textContent = state.mse.toFixed(2);
            document.getElementById('metricR2').textContent = state.r2.toFixed(3);
            const sign = state.b >= 0 ? '+' : '-';
            document.getElementById('equationText').textContent = `y = ${state.w.toFixed(2)}x ${sign} ${Math.abs(state.b).toFixed(2)}`;
            document.getElementById('dataCountBadge').textContent = `점 개수: ${state.points.length}개`;
        }

        function handleCanvasPointer(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            if (!clientX || !clientY) return;
            const dataPt = canvasToData(clientX - rect.left, clientY - rect.top);

            if (state.mode === 'add') {
                state.points.push(dataPt); state.predictionPoint = null;
                if (!state.isTraining) calculateAnalyticalRegression();
                draw();
            } else if (state.mode === 'predict') {
                state.predictionPoint = dataPt;
                const predY = state.w * dataPt.x + state.b;
                document.getElementById('manualXInput').value = dataPt.x.toFixed(1);
                document.getElementById('manualPredictResult').innerHTML = `<span class="text-indigo-600 font-bold">X = ${dataPt.x.toFixed(1)}</span> 이면 <span class="text-emerald-600 font-bold">Y = ${predY.toFixed(1)}</span> 예측!`;
                draw();
            }
        }

        window.addEventListener('load', () => {
            window.addEventListener('resize', resizeCanvas);
            canvas.addEventListener('mousedown', handleCanvasPointer);
            canvas.addEventListener('touchstart', (e) => { e.preventDefault(); handleCanvasPointer(e); });

            document.getElementById('modeAddBtn').addEventListener('click', () => {
                state.mode = 'add';
                document.getElementById('modeAddBtn').className = "px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 bg-indigo-600 text-white shadow-sm";
                document.getElementById('modePredictBtn').className = "px-3.5 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-indigo-600 transition-all flex items-center gap-2";
            });

            document.getElementById('modePredictBtn').addEventListener('click', () => {
                state.mode = 'predict';
                document.getElementById('modePredictBtn').className = "px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 bg-indigo-600 text-white shadow-sm";
                document.getElementById('modeAddBtn').className = "px-3.5 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-indigo-600 transition-all flex items-center gap-2";
            });

            document.querySelectorAll('.preset-btn').forEach(b => b.addEventListener('click', () => loadPreset(b.getAttribute('data-preset'))));
            
            document.getElementById('epochSlider').addEventListener('input', (e) => {
                state.epochsTarget = parseInt(e.target.value);
                document.getElementById('epochValueText').textContent = `${state.epochsTarget}회`;
            });

            document.getElementById('lrSlider').addEventListener('input', (e) => {
                state.learningRate = parseFloat(e.target.value);
                document.getElementById('lrValueText').textContent = state.learningRate.toString();
            });

            document.getElementById('clearBtn').addEventListener('click', () => {
                state.points = []; state.predictionPoint = null; state.w = 0; state.b = 0; state.currentEpoch = 0;
                if (state.isTraining) cancelAnimationFrame(state.animationId);
                state.isTraining = false;
                calculateMetrics(); draw();
            });

            document.getElementById('trainStepBtn').addEventListener('click', () => {
                if (state.points.length === 0) return alert('점 데이터가 필요합니다.');
                if (state.isTraining) { state.isTraining = false; cancelAnimationFrame(state.animationId); return; }
                state.currentEpoch = 0; state.w = (Math.random() - 0.5) * 0.5; state.b = Math.random() * 50;
                state.isTraining = true; runGradientDescentStep();
            });

            document.getElementById('instantFitBtn').addEventListener('click', () => {
                if (state.points.length === 0) return alert('데이터를 추가해 주세요.');
                if (state.isTraining) cancelAnimationFrame(state.animationId);
                state.isTraining = false;
                calculateAnalyticalRegression(); draw();
            });

            document.getElementById('showErrorLinesToggle').addEventListener('change', (e) => { state.showErrorLines = e.target.checked; draw(); });
            document.getElementById('showGridToggle').addEventListener('change', (e) => { state.showGrid = e.target.checked; draw(); });

            resizeCanvas();
            loadPreset('study');
        });
    </script>
</body>
</html>
"""

st.sidebar.title("🧠 ML Playground")
st.sidebar.caption("고등학생을 위한 인공지능 기초 실습실")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 개념 요약노트")
with st.sidebar.expander("1. 선형 회귀란?"):
    st.write("""
    **선형 회귀(Linear Regression)**는 여러 데이터 점들의 전체적인 경향성을 가장 잘 나타내는 하나의 직선 방정식($y = wx + b$)을 찾는 알고리즘입니다.
    """)

with st.sidebar.expander("2. 기울기($w$)와 절편($b$)"):
    st.write("""
    - **기울기($w$, Weight)**: X가 1단위 증가할 때 Y가 변경되는 비율
    - **절편($b$, Bias)**: X가 0일 때의 기본 Y값
    """)

with st.sidebar.expander("3. 오차(MSE)와 결정계수($R^2$)"):
    st.write("""
    - **MSE (평균제곱오차)**: 실젯값과 예측값 차이의 제곱 평균 ($0$에 가까울수록 우수)
    - **$R^2$ (결정계수)**: 데이터 설명력 ($1.0$에 가까울수록 완벽)
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 실습용 CSV 샘플 다운로드")

# Generate Sample CSV Data for Class Homework
sample_df = pd.DataFrame({
    '공부시간_X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '시험성적_Y': [25, 32, 40, 52, 61, 65, 74, 82, 88, 95]
})
csv_bytes = sample_df.to_csv(index=False).encode('utf-8-sig')

st.sidebar.download_button(
    label="📄 샘플 CSV 데이터 (공부시간-성적)",
    data=csv_bytes,
    file_name="study_score_sample.csv",
    mime="text/csv",
    use_container_width=True
)

st.title("🤖 머신러닝 플레이그라운드 (선형 회귀)")
st.caption("직접 클릭하거나 파이썬 분석을 통해 선형 회귀 및 경사하강법의 원리를 탐구해보세요.")

tab1, tab2, tab3 = st.tabs(["🎮 대화형 인터랙티브 캔버스", "📊 파이썬 ML 분석 & 파일 분석", "💻 Google Colab 코드 생성기"])

with tab1:
    st.markdown("##### 💡 캔버스를 클릭하여 데이터 점을 추가하거나, '경사하강법 학습' 버튼을 눌러보세요.")
    components.html(HTML_PLAYGROUND_CODE, height=820, scrolling=True)

with tab2:
    st.subheader("📁 CSV 파일 업로드 및 Scikit-Learn 머신러닝 분석")
    st.write("실제 데이터셋을 업로드하여 선형 회귀 모델을 학습하고 추정 결과를 그래프로 확인하세요.")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv", "txt"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"데이터셋을 성공적으로 불러왔습니다! (총 {len(df)}개 행)")
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.dataframe(df.head(10), use_container_width=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X축 (독립변수 / 특성)", numeric_cols, index=0)
                y_col = st.selectbox("Y축 (종속변수 / 타겟)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            else:
                st.error("숫자형 수치 열이 2개 이상 필요합니다.")
                x_col, y_col = None, None

        with col_right:
            if x_col and y_col and x_col != y_col:
                X = df[[x_col]].values
                y = df[y_col].values
                
                # Fit Scikit-Learn Model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                w = model.coef_[0]
                b = model.intercept_
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("기울기 (w)", f"{w:.4f}")
                m2.metric("절편 (b)", f"{b:.4f}")
                m3.metric("MSE 오차", f"{mse:.2f}")
                m4.metric("결정계수 (R²)", f"{r2:.4f}")
                
                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name='실제 데이터 점', marker=dict(color='#6366f1', size=9)))
                fig.add_trace(go.Scatter(x=df[x_col], y=y_pred, mode='lines', name='최적 회귀선', line=dict(color='#10b981', width=3)))
                
                fig.update_layout(
                    title=f"회귀 방정식: y = {w:.3f}x + {b:.3f}",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⬆️ 상단 파일 업로더에 CSV 파일을 올려보거나, 좌측 사이드바에서 샘플 CSV 파일을 다운로드받아 테스트해보세요!")

with tab3:
    st.subheader("💻 파이썬(Colab/Jupyter) 실습 코드 자동 생성기")
    st.write("플레이그라운드에서 실습한 내용을 파이썬 코드로 구현한 예제입니다. 복사하여 Google Colab에서 실행해보세요!")

    code_snippet = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 예시 데이터 생성 (공부 시간 vs 시험 점수)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([25, 32, 40, 52, 61, 65, 74, 82, 88, 95])

# 2. 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 3. 예측값 계산
y_pred = model.predict(X)

# 4. 기울기(w), 절편(b), 평가 지표 출력
print(f"기울기 (w): {model.coef_[0]:.4f}")
print(f"절편 (b): {model.intercept_:.4f}")
print(f"평균제곱오차 (MSE): {mean_squared_error(y, y_pred):.2f}")
print(f"결정계수 (R²): {r2_score(y, y_pred):.4f}")

# 5. 시각화 (산점도 및 회귀선)
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Study Hours (X)')
plt.ylabel('Exam Score (Y)')
plt.title('Linear Regression Playground Code')
plt.legend()
plt.grid(True)
plt.show()
"""
    st.code(code_snippet, language="python")
