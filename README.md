# Amazon Review Rating Prediction

Predicting star ratings (1–5) from review text on 500K Amazon book reviews,
with hand-written TF-IDF features and classical regressors. Course project
for CSE 158 (Recommender Systems & Web Mining) at UCSD, Fall 2025.

## Results

| Model | Test MSE |
| --- | ---: |
| Mean baseline | 1.017 |
| Ridge + count unigrams | 0.633 |
| Ridge + TF-IDF unigrams | **0.590** |
| Lasso + TF-IDF unigrams | 0.590 |
| Random forest + TF-IDF unigrams | 0.699 |

Ridge on TF-IDF unigrams works best, cutting MSE about 42% below the mean
baseline. Bigrams consistently underperformed unigrams. As a sanity check on
the learned weights: the most negative coefficients were "waste",
"disappointing" and "boring", which matches intuition.

![Model comparison](figures/model_comparison.png)

## Data

Books subset of the [McAuley Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/),
first 500K reviews. The dataset is not committed to the repo — download
`Books.json.gz` and place it at `dataset/Books.json.gz`.

## Method

TF, DF and TF-IDF are implemented from scratch in `src/features.py`
(binary TF, top-1000 unigram and bigram vocabularies) rather than with
sklearn's vectorizers. Models train on the first 400K reviews and are
evaluated on the held-out 100K.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/rating_prediction.ipynb
```

An HTML render of the finished notebook is in `notebooks/workbook.html`.