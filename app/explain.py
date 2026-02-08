# app/explain.py
# Run: python3 app/explain.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


# --- paths ---
# This file lives in /app, so model is at ../model/model_out
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_out")

# --- labels 
LABELS = ["Real News", "Fake News", "Satire"]


# --- load model/tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
model.to(device)


# --- predict for LIME (returns probabilities for each class)
def predict_proba(texts):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# --- LIME explainer ---
explainer = LimeTextExplainer(class_names=LABELS)


def explain_with_lime(text, num_features=12, num_samples=1000):
    # predict
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))

    print("\nPrediction:", LABELS[pred_idx])
    print("Probabilities:")
    for i, name in enumerate(LABELS):
        print(f"  {name}: {probs[i]:.3f}")

    # explain
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )

    print("\nTop influential words (LIME weights):")
    for word, weight in exp.as_list():
        print(f"  {word}: {weight:.3f}")

    # save html
    out_path = os.path.join(BASE_DIR, "lime_explanation.html")
    exp.save_to_file(out_path)
    print(f"\nSaved explanation to {out_path}")


def main():
    print("\nLIME Explainability Module")
    print("Type text to explain (or 'exit')")

    while True:
     
       text = input("\nText: ").strip()

        if text.lower() == "exit":
            break

        # Stop the crashes
        if len(text.split()) < 5:
            print("⚠️ Please enter a longer text (at least ~5 words).")
            continue

        try:
            explain_with_lime(text)
        except Exception as e:
            print(f"⚠️ LIME failed for this input: {e}")
            print("Try a longer/cleaner text and run again.")


if __name__ == "__main__":
    main()
