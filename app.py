import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os

# Streamlit page config
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

st.title("🧠 Handwritten Digit Recognition (MNIST)")
st.write("Upload an image of a handwritten digit (0–9). The app expects a grayscale image; colored images are converted automatically.")

# Model path
model_path = "model/digit_model.h5"

if not os.path.exists(model_path):
    st.warning("Trained model not found. Please run `save_model.py` to generate `model/digit_model.h5`.")
    st.stop()

# Load model
model = tf.keras.models.load_model(model_path)

# File uploader
uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])

# Preprocessing function to convert any image to MNIST-style 28x28
def preprocess_image_for_mnist(pil_img, target_size=28):
    # Convert to grayscale
    img = pil_img.convert('L')
    arr = np.array(img)

    # Invert if needed so digit is white, background black
    if arr.mean() > 127:
        arr = 255 - arr

    # Mask for digit
    mask = arr > 10
    if not mask.any():
        canvas = np.zeros((target_size, target_size), dtype=np.float32)
        return canvas.reshape(1, target_size, target_size, 1)

    # Crop to bounding box
    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = arr[y0:y1+1, x0:x1+1]

    # Resize while keeping aspect ratio (largest side = 20px)
    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * new_h) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * new_w) / w)))

    cropped_img = Image.fromarray(cropped)
    # Pillow 10+ compatible
    resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized)

    # Place on 28x28 canvas centered
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized_arr

    # Center by mass (like MNIST)
    total = canvas.sum()
    if total > 0:
        rows = np.arange(target_size).reshape(target_size, 1)
        cols = np.arange(target_size).reshape(1, target_size)
        cy = (canvas * rows).sum() / total
        cx = (canvas * cols).sum() / total
        shift_y = int(round((target_size/2 - 1) - cy))
        shift_x = int(round((target_size/2 - 1) - cx))
        canvas = np.roll(np.roll(canvas, shift_y, axis=0), shift_x, axis=1)

    # Normalize 0..1
    canvas = canvas.astype(np.float32) / 255.0
    return canvas.reshape(1, target_size, target_size, 1)

# If a file is uploaded
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    img_array = preprocess_image_for_mnist(input_image, target_size=28)

    # Show processed image
    proc = (img_array[0,:,:,0]*255).astype(np.uint8)
    st.image(proc, caption="Processed MNIST-style image", width=150)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Display results
    st.write(f"### Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {confidence:.3f}")

# streamlit run app.py run command to start the app