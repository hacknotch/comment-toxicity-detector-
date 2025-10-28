import os
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_FILES = {"index.html", "styles.css", "script.js"}

# --- Flask app ---
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# --- Load HF model (fine-tuned if available) ---
MODEL_DIR = os.path.join(BASE_DIR, "model")
BASE_MODEL_ID = "distilbert-base-uncased"

_tokenizer = None
_model = None
_max_len = 192

try:
    load_path = MODEL_DIR if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")) else BASE_MODEL_ID
    _tokenizer = AutoTokenizer.from_pretrained(load_path)
    _model = AutoModelForSequenceClassification.from_pretrained(load_path)
    _model.eval()
except Exception as _e:
    # Fallback to rule-based only
    _tokenizer = None
    _model = None

# --- Toxicity Detection Rules ---
TOXIC_WORDS = [
    'hate', 'stupid', 'idiot', 'moron', 'dumb', 'ugly', 'fat', 'loser', 'pathetic',
    'worthless', 'trash', 'garbage', 'kill', 'die', 'death', 'suicide', 'murder',
    'asshole', 'bitch', 'whore', 'slut', 'fuck', 'shit', 'damn', 'hell', 'crap',
    'bastard', 'freak', 'weirdo', 'creep', 'pervert', 'retard', 'gay', 'fag',
    'nigger', 'chink', 'spic', 'kike', 'terrorist', 'criminal', 'thug',
    'disgusting', 'revolting', 'vile', 'horrible', 'awful', 'terrible', 'nasty',
    'gross', 'sick', 'disgusted', 'annoying', 'irritating', 'bothersome',
    'useless', 'pointless', 'meaningless', 'hopeless', 'pitiful',
    'embarrassing', 'shameful', 'disgraceful', 'hateful', 'spiteful', 'cruel',
    'mean', 'vicious', 'toxic', 'harmful', 'dangerous', 'threatening',
    'bully', 'harass', 'abuse', 'insult', 'offensive', 'rude', 'disrespectful',
    'racist', 'sexist', 'homophobic', 'discriminatory', 'violent', 'aggressive',
    'hostile', 'angry', 'furious', 'crazy', 'insane', 'psycho', 'mental',
    'unstable', 'failure', 'flop', 'waste', 'scum', 'liar', 'cheat', 'fraud',
    'fake', 'betray', 'hurt', 'harm', 'destroy'
]

HARSH_PATTERNS = [
    r'\b(you\s+(are\s+)?(an?\s+)?(stupid|idiot|moron|loser|worthless|disgusting|toxic|failure|liar|fraud|fake|awful|horrible|terrible))\b',
    r'\b(go\s+(die|kill\s+yourself|to\s+hell))\b',
    r'\b(i\s+hate\s+(you|this|that|everything))\b',
    r'\b(no\s+one\s+(likes|wants|needs)\s+you)\b',
    r'\b(you\s+are\s+(nothing|useless|pointless|hopeless|pathetic))\b',
    r'\b(you\s+always\s+(fail|lose|suck))\b',
    r'\b(you\s+(look|are)\s+(ugly|fat|gross|sick))\b',
    r'\b(you\s+(are\s+)?(crazy|insane|psycho|mental|unstable))\b',
    r'\b(you\s+(are\s+)?(a\s+)?(slut|whore|bitch|asshole|bastard|pervert))\b',
    r'\b(i\s+(will\s+)?(hurt|harm|kill|destroy)\s+you)\b',
    r'\b(go\s+back\s+to\s+(your\s+)?(country|home))\b',
    r'\b[A-Z]{3,}\b'
]

SECURITY_THREATS = [
    # Cyber threats
    r'\b(hack|hacker|hacked|phishing|scam|fraud|steal|stolen|breach|leak|expose)\b',
    r'\b(password|crack|sql\s+injection|ddos|spyware|malware|keylogger)\b',
    r'\b(identity\s+theft|credit\s+card|bank\s+account|privacy\s+violation)\b',
    # Financial fraud
    r'\b(money\s+laundering|ponzi\s+scheme|investment\s+fraud|get\s+rich\s+quick)\b',
    r'\b(fake\s+investment|fake\s+company|fake\s+website|fake\s+email|phishing\s+email)\b',
    # Social engineering
    r'\b(impersonate|fake\s+identity|catfish|romance\s+scam)\b',
    # Cyberbullying
    r'\b(stalk|harass|dox|doxxing|cyberbullying|online\s+harassment)\b'
]

# --- Cyber Fraud Detection Patterns ---
CYBER_FRAUD_PATTERNS = [
    r'\b(free\s+money|instant\s+loan|easy\s+profit|get\s+rich\s+quick|limited\s+offer|urgent\s+action)\b',
    r'\b(verify\s+account|reset\s+password|update\s+details|confirm\s+identity)\b',
    r'\b(bank\s+alert|credit\s+card\s+blocked|suspicious\s+login|account\s+suspended)\b',
    r'\b(prize\s+winner|lottery|reward\s+claim|congratulations\s+you\s+won)\b',
    r'\b(send\s+otp|share\s+otp|one\s+time\s+password|security\s+code)\b',
    r'\b(contact\s+this\s+number|call\s+this\s+number|click\s+the\s+link|visit\s+this\s+site)\b',
    r'\b(fake\s+website|impersonation|unauthorized\s+access|cyber\s+crime)\b',
    r'\b(fraudulent\s+transaction|fake\s+email|phishing\s+message)\b',
    r'\b(request\s+for\s+payment|transfer\s+funds|wallet\s+topup)\b',
    r'\b(refund\s+offer|cashback\s+offer|win\s+reward|claim\s+bonus)\b'
]

def analyze_toxicity(text):
    text_lower = text.lower()
    toxic_word_count = sum(1 for w in TOXIC_WORDS if w in text_lower)
    pattern_matches = sum(1 for p in HARSH_PATTERNS if re.search(p, text_lower))
    score = min(toxic_word_count * 0.25 + pattern_matches * 0.3, 1.0)
    return {
        'score': round(score, 2),
        'found_words': [w for w in TOXIC_WORDS if w in text_lower],
        'patterns': pattern_matches
    }


def predict_model(text: str) -> float:
    """Return probability of toxic (class 1). Fallback to heuristic if model unavailable."""
    if _tokenizer is None or _model is None:
        return analyze_toxicity(text).get('score', 0.0)
    with torch.no_grad():
        enc = _tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=_max_len)
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # Assume id 1 corresponds to toxic (as trained)
        return float(probs[1].item())

def analyze_security_threats(text):
    text_lower = text.lower()
    matched = [p for p in SECURITY_THREATS if re.search(p, text_lower)]
    
    # Categorize threats
    threat_categories = {
        'cyber': [],
        'financial': [],
        'social_engineering': [],
        'harassment': []
    }
    
    for i, pattern in enumerate(matched):
        if i < 3:  # Cyber threats
            threat_categories['cyber'].append(f"Threat {i+1}")
        elif i < 5:  # Financial fraud
            threat_categories['financial'].append(f"Threat {i+1}")
        elif i < 6:  # Social engineering
            threat_categories['social_engineering'].append(f"Threat {i+1}")
        else:  # Harassment
            threat_categories['harassment'].append(f"Threat {i+1}")
    
    return {
        'threat_count': len(matched),
        'found_threats': matched,
        'threat_categories': threat_categories
    }

def analyze_cyber_frauds(text):
    text_lower = text.lower()
    fraud_matches = []
    for pattern in CYBER_FRAUD_PATTERNS:
        if re.search(pattern, text_lower):
            fraud_matches.append(pattern)
    score = min(len(fraud_matches) * 0.2, 1.0)
    return {
        'fraud_count': len(fraud_matches),
        'fraud_indicators': fraud_matches,
        'fraud_score': round(score, 2)
    }

# ----------------- ROUTES -----------------
@app.get("/")
def root():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/<path:filename>")
def assets(filename: str):
    # Serve only allowed public files
    if filename in PUBLIC_FILES and os.path.exists(os.path.join(BASE_DIR, filename)):
        return send_from_directory(BASE_DIR, filename)
    return ("Not found", 404)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        text = str((payload or {}).get("comment", "")).strip()
        if not text:
            return jsonify({"error": "comment is required"}), 400

        model_score = predict_model(text)
        heur = analyze_toxicity(text).get('score', 0.0)
        toxic_score = max(model_score, heur)
        label = "toxic" if toxic_score >= 0.4 else "non-toxic"
        return jsonify({
            "label": label,
            "confidence": round(toxic_score, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/analyze_visual")
def analyze_visual():
    try:
        payload = request.get_json(force=True)
        text = str((payload or {}).get("comment", "")).strip()
        if not text:
            return jsonify({"error": "comment is required"}), 400

        # Model and heuristic scores
        model_score = predict_model(text)
        analysis = analyze_toxicity(text)
        heur = analysis.get('score', 0.0)
        toxic_score = max(model_score, heur)
        non_toxic_score = 1.0 - toxic_score
        
        result = {
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "scores": {
                "non-toxic": round(non_toxic_score, 4),
                "toxic": round(toxic_score, 4),
                "model_toxic": round(model_score, 4),
                "heuristic_toxic": round(heur, 4)
            },
            "overall_toxicity": round(toxic_score, 4),
            "severity": "high" if toxic_score >= 0.7 else "medium" if toxic_score >= 0.4 else "low",
            "toxic_words": analysis.get('found_words', []),
            "pattern_matches": analysis.get('patterns', 0)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/analyze_security")
def analyze_security():
    try:
        payload = request.get_json(force=True)
        text = str((payload or {}).get("comment", "")).strip()
        if not text:
            return jsonify({"error": "comment is required"}), 400

        security_analysis = analyze_security_threats(text)
        fraud_analysis = analyze_cyber_frauds(text)
        
        # Calculate combined risk level
        total_threats = security_analysis['threat_count'] + fraud_analysis['fraud_count']
        if total_threats >= 5:
            risk_level = "CRITICAL"
        elif total_threats >= 3:
            risk_level = "HIGH"
        elif total_threats >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Create security analysis result
        result = {
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "threat_count": security_analysis['threat_count'],
            "fraud_count": fraud_analysis['fraud_count'],
            "total_threats": total_threats,
            "risk_level": risk_level,
            "threat_categories": security_analysis.get('threat_categories', {}),
            "found_threats": security_analysis['found_threats'],
            "fraud_indicators": fraud_analysis['fraud_indicators'],
            "fraud_score": fraud_analysis.get('fraud_score', 0)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    print(f"Running on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
