import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ── Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


st.markdown("""
<style>
    /* Prediction number — big and bold */
    .prediction-box {
        text-align: center;
        padding: 1.2rem 0 0.5rem;
    }
    .prediction-label {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-bottom: 0.2rem;
    }
    .prediction-number {
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        color: var(--text-color);
    }
    .confidence-label {
        font-size: 0.85rem;
        opacity: 0.55;
        margin-top: 0.3rem;
    }
    /* Waiting state */
    .waiting {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        font-size: 1rem;
        opacity: 0.45;
    }
    /* Bar chart label visibility fix */
    .stBarChart text {
        fill: var(--text-color) !important;
    }
    /* Canvas border */
    .canvas-container canvas {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()


if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# ── UI Layout
st.title("Digit Recognizer")
st.caption("Draw any digit from 0 to 9 — the model will predict it in real time.")

col_canvas, col_result = st.columns([1.1, 1], gap="large")

with col_canvas:
    st.markdown("**Draw here**")
    canvas_result = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",   # pure black matches MNIST training data exactly
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",  # changes on Clear → forces remount
    )
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.canvas_key += 1   # new key → canvas remounts blank
        st.rerun()

with col_result:
    st.markdown("**Prediction**")

    img_data = canvas_result.image_data

    # ── Blank canvas guard
    has_drawing = (
        img_data is not None
        and np.any(img_data[:, :, :3].min(axis=2) > 200)  # white pixels = drawn stroke
    )

    if not has_drawing:
        st.markdown('<div class="waiting">← Draw a digit to see a prediction</div>', unsafe_allow_html=True)
    else:
        
        img = Image.fromarray(img_data.astype("uint8"))

        img = img.convert("L")

        img = img.resize((28, 28), Image.LANCZOS)

        arr = np.array(img) / 255.0

        #    nn.Flatten() inside the model handles flattening to [1, 784]
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # ── Inference
        with torch.no_grad():
            logits = model(tensor)                     # raw scores: [1, 10]
            probs = torch.softmax(logits, dim=1)[0]    # probabilities: [10]
            pred = probs.argmax().item()               # winning digit (0–9)
            confidence = probs[pred].item() * 100      # as percentage

        # ── Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">I think it's a…</div>
            <div class="prediction-number">{pred}</div>
            <div class="confidence-label">{confidence:.1f}% confident</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bar chart 
        st.markdown("**Confidence per digit**")
        chart_data = {str(i): round(probs[i].item() * 100, 2) for i in range(10)}
        st.bar_chart(chart_data, height=180, use_container_width=True)