import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


model = nn.Sequential(
    nn.Flatten(),           # turns the 28x28 image into 784 inputs
    nn.Linear(784, 256),    
    nn.ReLU(),
    nn.Dropout(0.2), 
    nn.Linear(256, 128),    
    nn.ReLU(),
    nn.Linear(128, 10)
)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

st.title("Digit Recognizer")
st.write("Draw a digit (0–9) in the box below")

canvas = st_canvas(
    stroke_width=18, stroke_color="#FFFFFF",
    background_color="#000000",
    height=280, width=280, drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:
    # converting drawn canvas to a grayscale 28x28 image
    img = Image.fromarray(canvas.image_data.astype('uint8'))
    img = img.convert('L').resize((28,28))
    
    # normalize pixels to 0.0 - 1.0 (matching transforms.ToTensor())
    arr = np.array(img) / 255.0
    
    # convert to tensor and shape it as a single-batch image: [1, 1, 28, 28]
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # make the prediction
    with torch.no_grad():
        out = model(tensor) # turns [1, 1, 28, 28] into [1, 784]
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
        
    st.markdown(f"### Prediction: **{pred}**")
    st.bar_chart({str(i): probs[i].item() for i in range(10)})