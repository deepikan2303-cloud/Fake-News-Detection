# Fake News Detection (TF-IDF + Linear Models)

A text classifier that predicts whether a news article is **fake** or **real**, using TF-IDF features and linear models from scikit-learn.

## Dataset
[Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Download `Fake.csv` and `True.csv` and put them in a `data/` folder:
```
fake-news-detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── fake_news_detection.py
└── requirements.txt
```
The dataset is not included in this repo because of its size.

## How to run
```bash
pip install -r requirements.txt
python fake_news_detection.py
```
Charts and the trained model are saved in `outputs/`.

## Steps
1. **Load** both CSVs, label fake = 1 and real = 0, and shuffle.
2. **EDA**: shape, missing values, class balance.
3. **Preprocess**: combine title and text, lowercase, remove URLs, punctuation and numbers, and strip the `"CITY (Reuters) -"` prefix.
4. **Vectorize** with `TfidfVectorizer` (unigrams + bigrams, English stopwords).
5. **Train** Logistic Regression, Passive Aggressive and Linear SVC.
6. **Evaluate** with accuracy, precision, recall, F1 and a confusion matrix.
7. **Interpret** the top words that push predictions towards fake or real.

## Results
Fill in after running:

| Model | Accuracy |
|---|---|
| Logistic Regression | |
| Passive Aggressive | |
| Linear SVC | |

## Limitations
- Real articles in this dataset come mostly from Reuters, so the model may learn writing style rather than truthfulness. The Reuters prefix is removed to reduce this leakage.
- The model only sees word patterns; it cannot fact-check claims.

## Future work
- Try transformer models such as BERT.
- Test on news from other sources and time periods.
- Add a small web app (Streamlit or Gradio).
