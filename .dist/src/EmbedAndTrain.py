# src/embed_and_train.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "distilbert-base-uncased"

def meanpooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def getembeddings(texts, tokenizer, model, device='cpu', batch_size=32):
    model.to(device)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=256)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            pooled = meanpooling(last_hidden, attention_mask)
            pooled = pooled.cpu().numpy()
            embeddings.append(pooled)
    embeddings = np.vstack(embeddings)
    return embeddings

def train(infile="../data/cleaned_feedback.csv", model_out="../models/sentiment_model.pkl"):
    df = pd.read_csv(infile)
    texts = df['preprocessed'].astype(str).tolist()
    labels = df['sentiment_label'].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    X = getembeddings(texts, tokenizer, model, device='cpu', batch_size=64)

    Le = LabelEncoder()
    Y = Le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

    CLF = LogisticRegression(max_iter=1000)
    CLF.fit(X_train, y_train)

    y_pred = CLF.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print(classification_report(y_test, y_pred, target_names=Le.classes_))
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"clf":CLF, "label_encoder":Le, "embedding_model_name":MODEL_NAME}, model_out)
    print("Saved", model_out)
    return model_out

if __name__ == "__main__":
    train()
