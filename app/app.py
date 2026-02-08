import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# 1) Resolve local model path safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_out")

# -------------------------
# 2) Load tokenizer + model 
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# If you have GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 3) Label mapping 
# -------------------------
LABELS = {
    0: "Real News",
    1: "Fake News",
    2: "Satire"
}

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # move tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]  # shape: (num_labels,)
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    return LABELS.get(pred_id, str(pred_id)), confidence, probs.cpu().tolist()


# 4) Simple CLI test loop

if __name__ == "__main__":
    print("✅ Model loaded from:", os.path.abspath(MODEL_PATH))
    print("Type text to classify. Type 'exit' to quit.\n")

    while True:
        text = input("Enter news text: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue

        label, conf, all_probs = predict(text)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {conf:.4f}")
        print(f"All probs: {all_probs}\n")
