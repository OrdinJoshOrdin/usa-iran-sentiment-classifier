# USA-Iran Conflict Sentiment Classifier

A Streamlit app that classifies the sentiment of social media comments
about the USA-Iran conflict as positive, negative, or neutral, using a
TF-IDF + Multinomial Naive Bayes model, with a ChatGPT-inspired
chat-thread interface.

The model is trained on `MAYOWA_DATASET.csv`, 18,019 cleaned, real
social media comments about the conflict, and achieves roughly 72%
accuracy on a held-out test split.

## Run locally

```
pip install -r requirements.txt
streamlit run usa_iran_sentiment_app.py
```
