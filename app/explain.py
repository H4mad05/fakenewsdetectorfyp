import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


#  paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_out")

#  labels 

LABELS = ["Fake News", "Satire", "Real News"]  


# load model/tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# Fast device for normal single predictions
FAST_DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
model.to(FAST_DEVICE)

# LIME device (force CPU to avoid MPS OOM)
LIME_DEVICE = torch.device("cpu")


def predict_proba(texts, device=FAST_DEVICE, max_length=25):
    """Return probabilities array shape (n, num_classes)."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs


def predict_proba_for_lime(texts, batch_size=16, max_length=128):
    """
    LIME calls this many times with many perturbed samples.
    Force CPU + batching to prevent MPS/CUDA OOM.
    """
    # temporarily move model to CPU for LIME
    original_device = next(model.parameters()).device
    if original_device.type != "cpu":
        model.to(LIME_DEVICE)

    model.eval()
    all_probs = []

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            probs = predict_proba(batch, device=LIME_DEVICE, max_length=max_length)
            all_probs.append(probs)

        return np.vstack(all_probs)

    finally:
        # move back to original device for normal fast predictions
        if original_device.type != "cpu":
            model.to(original_device)


# LIME explainer 
explainer = LimeTextExplainer(class_names=LABELS, split_expression=r"\W+")


def explain_with_lime(text, num_features=15, num_samples=2000):
    # normal prediction (fast device)
    probs = predict_proba([text], device=FAST_DEVICE, max_length=256)[0]
    pred_idx = int(np.argmax(probs))

    print("\nPrediction:", LABELS[pred_idx])
    print("Probabilities:")
    for i, name in enumerate(LABELS):
        print(f"  {name}: {probs[i]:.3f}")

    # explain predicted class only (more stable + faster)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba_for_lime,
        labels=[pred_idx],
        num_features=num_features,
        num_samples=num_samples
    )

    print("\nTop influential words (LIME weights):")
    for word, weight in exp.as_list(label=pred_idx):
        print(f"  {word:15s}: {weight:+.6f}")

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

        if len(text.split()) < 5:
            print("⚠️ Please enter a longer text (at least ~5 words).")
            continue

        try:
            explain_with_lime(text)
        except RuntimeError as e:
            # common: MPS/CUDA OOM or similar
            print(f"⚠️ Runtime error: {e}")
            print("Tip: try smaller num_samples (e.g., 1000) or shorter text.")
        except Exception as e:
            print(f"⚠️ LIME failed for this input: {e}")
            print("Try a longer/cleaner text and run again.")


if __name__ == "__main__":
    main()
