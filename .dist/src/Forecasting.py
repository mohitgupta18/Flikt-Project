import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Optional: Prophet for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def LabelToScore(label):
    if str(label).lower().startswith("pos"):
        return 1.0
    elif str(label).lower().startswith("neg"):
        return 0.0
    return 0.5

def prepareTimeseries(infile="../data/cleaned_feedback.csv"):
    df = pd.read_csv(infile, parse_dates=['timestamp'])
    df['score'] = df['sentiment_label'].apply(LabelToScore)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    Daily = df.groupby('date')['score'].mean().reset_index()
    Daily.columns = ['ds', 'y']
    Daily['ds'] = pd.to_datetime(Daily['ds'])
    return Daily

def forecastNextMonth(daily_df, periods=30, outfile="../data/forecast.png"):
    if PROPHET_AVAILABLE:
        M = Prophet()
        M.fit(daily_df)
        Future = M.make_future_dataframe(periods=periods)
        forecast = M.predict(Future)
        Fig = M.plot(forecast)
        Fig.savefig(outfile)
        return forecast, outfile
    else:
        daily_df['t'] = (daily_df['ds'] - daily_df['ds'].min()).dt.days
        X = daily_df[['t']].values
        y = daily_df['y'].values
        model = LinearRegression().fit(X, y)
        future_t = np.arange(daily_df['t'].max() + 1, daily_df['t'].max() + 1 + periods)
        preds = np.clip(model.predict(future_t.reshape(-1, 1)), 0, 1)
        future_dates = [daily_df['ds'].max() + timedelta(days=i) for i in range(1, periods + 1)]
        plt.figure(figsize=(10, 5))
        plt.plot(daily_df['ds'], daily_df['y'], label="Observed")
        plt.plot(future_dates, preds, label="Forecast (Linear)")
        plt.legend()
        plt.title("Customer Satisfaction Forecast")
        plt.savefig(outfile)
        return preds, outfile

def recurring_issues(infile="../data/cleaned_feedback.csv", top_n=15):
    df = pd.read_csv(infile)
    neg = df[df['sentiment_label'] == 'negative']
    if neg.empty:
        return []
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english', min_df=3)
    X = vectorizer.fit_transform(neg['preprocessed'].astype(str))
    Sums = np.asarray(X.sum(axis=0)).flatten()
    Terms = vectorizer.get_feature_names_out()
    idx = np.argsort(Sums)[::-1]
    return [(Terms[i], int(Sums[i])) for i in idx[:top_n]]
