"""
Fake News Detection with TF-IDF + Linear Models
------------------------------------------------
Dataset: Kaggle "Fake and Real News Dataset" (Fake.csv + True.csv)
Put both CSV files inside a folder named `data/` next to this script.

Run:  python fake_news_detection.py
"""

import os
import re
import string

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files (works without a display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
def load_data():
    fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
    true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
    fake["label"] = 1   # 1 = FAKE
    true["label"] = 0   # 0 = REAL
    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


# ---------------------------------------------------------------
# 2. Explore data (EDA)
# ---------------------------------------------------------------
def explore(df):
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isnull().sum())
    print("Class balance (1=fake, 0=real):\n", df["label"].value_counts())

    df["label"].map({0: "Real", 1: "Fake"}).value_counts().plot(
        kind="bar", color=["#4C72B0", "#DD8452"], title="Class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()


# ---------------------------------------------------------------
# 3. Preprocess text
# ---------------------------------------------------------------
# Real articles often start with "WASHINGTON (Reuters) - ".
# If we keep it, the model just learns "Reuters = real" (data leakage).
REUTERS_PREFIX = re.compile(r"^.*?\(reuters\)\s*-\s*", flags=re.IGNORECASE)


def clean_text(text):
    text = str(text)
    text = REUTERS_PREFIX.sub("", text)                 # remove leakage prefix
    text = text.lower()
    text = re.sub(r"\breuters\b", " ", text)            # remove any other "reuters"
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"<.*?>", " ", text)                  # HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)                    # numbers
    text = re.sub(r"\s+", " ", text).strip()            # extra spaces
    return text


def preprocess(df):
    df = df.dropna(subset=["text"]).copy()
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.len() > 0]
    return df


# ---------------------------------------------------------------
# 4. Build models (TF-IDF + linear classifier)
# ---------------------------------------------------------------
def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7,
                                  ngram_range=(1, 2), max_features=50000)),
        ("clf", model),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                                        eta0=1.0, max_iter=1000, random_state=42),
    "Linear SVC": LinearSVC(),
}


# ---------------------------------------------------------------
# 5. Train + evaluate
# ---------------------------------------------------------------
def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=42)

    results = {}
    best_name, best_pipe, best_acc = None, None, -1

    for name, model in MODELS.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

        if acc > best_acc:
            best_name, best_pipe, best_acc = name, pipe, acc

    # Confusion matrix for the best model
    preds = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # Model comparison chart
    pd.Series(results).sort_values().plot(kind="barh", color="#4C72B0",
                                          title="Accuracy by model")
    plt.xlim(min(results.values()) - 0.02, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()

    print(f"\nBest model: {best_name} (accuracy {best_acc:.4f})")
    return best_name, best_pipe


# ---------------------------------------------------------------
# 6. Interpret: which words push towards fake / real?
# ---------------------------------------------------------------
def show_top_words(pipe, n=15):
    words = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    print("\nTop words -> REAL:", ", ".join(words[order[:n]]))
    print("Top words -> FAKE:", ", ".join(words[order[-n:][::-1]]))


# ---------------------------------------------------------------
# 7. Predict on new text
# ---------------------------------------------------------------
def predict_news(pipe, text):
    label = pipe.predict([clean_text(text)])[0]
    return "FAKE" if label == 1 else "REAL"


def main():
    df = load_data()
    explore(df)
    df = preprocess(df)
    best_name, best_pipe = train_and_evaluate(df)
    show_top_words(best_pipe)

    joblib.dump(best_pipe, os.path.join(OUTPUT_DIR, "fake_news_model.joblib"))
    print(f"\nSaved model to {OUTPUT_DIR}/fake_news_model.joblib")

    sample = ("BREAKING: Scientists confirm the moon is made of cheese, "
              "government hiding the truth!")
    print(f"\nSample prediction: '{sample[:50]}...' -> {predict_news(best_pipe, sample)}")


if __name__ == "__main__":
    main()
