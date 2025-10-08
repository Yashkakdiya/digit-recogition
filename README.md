# Handwritten Digit Recognition (MNIST) — Ready-to-Run Project
**Python version:** 3.10 (recommended)

## Contents
- `save_model.py`  — Script to train and save the CNN model (creates `model/digit_model.h5`)
- `app.py`         — Streamlit application that loads the saved model and predicts digits
- `requirements.txt` — Python dependencies
- `model/`         — Directory where the trained model will be saved (NOT included by default in this ZIP)
- `README.md`      — This file

> NOTE: This ZIP intentionally does **not** include a pretrained `digit_model.h5` due to environment and size constraints.
> It's simple to train the model locally or in Google Colab — instructions below.

## How to train the model (locally or on Colab)
1. Create a Python 3.10 virtual environment (optional but recommended):
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python save_model.py
   ```
   This will download MNIST, train a small CNN for 5 epochs, and save `model/digit_model.h5`.

## If you prefer Google Colab (recommended if you don't have a GPU locally)
1. Upload `save_model.py` to Colab or copy-paste its contents.
2. Run the cell after installing `tensorflow` (Colab comes with TF preinstalled in many runtimes):
   ```python
   !pip install -q tensorflow==2.13.0
   !python save_model.py
   ```
3. After training, download the generated `model/digit_model.h5` from the Colab filesystem and place it into the `model/` folder before deploying.

## Deploying on Render (Streamlit)
1. Push this repo to GitHub.
2. Create a new Web Service on Render and connect your GitHub repo.
3. Settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port 10000`
   - Ensure the trained `model/digit_model.h5` exists in the `model/` folder in the repo **or** add a build step that runs training (not recommended on free services).
4. Deploy and open the provided URL.

## Tips & Troubleshooting
- If Streamlit fails to start on Render due to large `tensorflow` install, consider using a smaller model or hosting the model weights on external storage and downloading them at startup.
- If you want me to produce a *pretrained* model file and include it in the ZIP, I can try — but it may fail here due to GPU and environment limits. Alternatively, I can give you a direct Colab notebook that trains and saves the model and produces a downloadable `digit_model.h5`.

## Want me to also create a Colab notebook that trains the model and provides a download link?
If yes, tell me and I'll add the Colab-ready notebook next.
