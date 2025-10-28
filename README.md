# Cyberbullying / Toxic Comment Detection

This mini-project fine-tunes DistilBERT on a YouTube toxicity dataset and serves predictions via a Flask API. A simple static frontend calls the API and color-codes the confidence.

## Project structure
- `youtoxic_english_1000.csv` – dataset with `Text` and `IsToxic` columns
- `train_model.py` – fine-tunes DistilBERT, saves into `model/`
- `app.py` – Flask backend with `/predict` endpoint
- `index.html`, `styles.css`, `script.js` – static frontend
- `requirements.txt` – Python deps

## Setup
1. Python 3.10+
2. Create venv and install deps:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train (optional)
The app will work with the base model if no fine-tuned model is found. To fine-tune:
```bash
python train_model.py
```
This writes the model and tokenizer to `model/`.

## Run backend
```bash
python app.py
```
API: `POST http://127.0.0.1:5000/predict` with JSON `{ "comment": "text" }`
Response: `{ "label": "toxic|non-toxic", "confidence": 0.92 }`

## Run frontend
Open `index.html` in your browser. Ensure the backend is running at `http://127.0.0.1:5000`.

## Color coding
- 🟢 non-toxic (< 0.4)
- 🟠 mild (0.4–0.7)
- 🔴 toxic (> 0.7)
