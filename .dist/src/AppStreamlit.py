
import streamlit as St
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from src import data_preprocessing as dp
from Summarization import extractivesummary, AbstractiveSummarize
from Forecasting import prepareTimeseries, forecastNextMonth
import numpy as np
import os

St.title("Customer Feedback Sentiment + Summarization Demo")

uploaded = St.file_uploader("Upload feedback CSV", type=['csv'])
model_path = "../models/sentiment_model.pkl"

if uploaded:
    Df = pd.read_csv(uploaded)
    St.write("Raw data sample:", Df.head())
    St.info("Cleaning and preprocessing...")
    Df.to_csv("temp_uploaded.csv", index=False)
    cleaned = dp.main(infile="temp_uploaded.csv", outfile="temp_cleaned.csv")
    St.write("Cleaned sample:", cleaned.head())

    if os.path.exists(model_path):
        Meta = joblib.load(model_path)
        clf = Meta['clf']; le = Meta['label_encoder']; embed_model_name = Meta['embedding_model_name']
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        Modl = AutoModel.from_pretrained(embed_model_name)
        St.success("Loaded classifier")
        texts = cleaned['preprocessed'].astype(str).tolist()
        def get_embs(texts):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256)
            with torch.no_grad():
                outputs = Modl(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs['attention_mask']
                input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pooled = (sum_embeddings / sum_mask).cpu().numpy()
                return pooled
        X_new = get_embs(texts)
        preds = clf.predict(X_new)
        labels = le.inverse_transform(preds)
        cleaned['predicted_sentiment'] = labels
        St.write("Predicted sentiment counts:")
        St.bar_chart(cleaned['predicted_sentiment'].value_counts())

        negs = cleaned[cleaned['predicted_sentiment']=='negative'].head(5)
        St.header("Sample negative feedback + summaries")
        for idx, row in negs.iterrows():
            St.write("Feedback:", row['clean_text'])
            s = extractivesummary(row['clean_text'])
            St.write("Short summary:", s['short'])
            St.write("Detailed summary:", s['detailed'])
            St.markdown("---")

        St.header("Forecast customer satisfaction (next 30 days)")
        daily = prepareTimeseries(infile="temp_cleaned.csv")
        if len(daily) > 10:
            fc = forecastNextMonth(daily, periods=30, outfile="forecast_tmp.png")
            St.image("forecast_tmp.png")
        else:
            St.write("Not enough data to forecast.")
    else:
        St.error("Model not found; please run training first.")
