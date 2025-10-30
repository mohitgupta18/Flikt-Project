
import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib


from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


Path("models").mkdir(parents=True, exist_ok=True)

LABEL2ID = {"negative":0, "neutral":1, "positive":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def compute_metrics_sklearn(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


data_path = Path('data/cleaned_feedback.csv')
if not data_path.exists():
    print('Missing data/cleaned_feedback.csv — please run preprocessing or place the cleaned CSV at this path.')
else:
    df = pd.read_csv(data_path)
    print('Loaded rows:', len(df))
    display(df.head(3))


if 'preprocessed' not in df.columns:
    raise ValueError("'preprocessed' column not found. Run preprocessing first.")
df = df.dropna(subset=['preprocessed']).copy()
df['sentiment_label'] = df['sentiment_label'].fillna('neutral')
df = df[df['sentiment_label'].isin(['positive','negative','neutral'])]
df['label_id'] = df['sentiment_label'].map(LABEL2ID)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
print('Train:', len(train_df), 'Test:', len(test_df))


tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train = tfidf.fit_transform(train_df['preprocessed'])
X_test = tfidf.transform(test_df['preprocessed'])

clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', random_state=42)
clf.fit(X_train, train_df['label_id'])

y_pred = clf.predict(X_test)
metrics = compute_metrics_sklearn(test_df['label_id'], y_pred)
print('TF-IDF + LR metrics:', metrics)
print('\nClassification report:\n')
print(classification_report(test_df['label_id'], y_pred, target_names=['negative','neutral','positive']))

joblib.dump({'tfidf': tfidf, 'clf': clf}, 'models/sentiment_model.pkl')
print('Saved TF-IDF baseline to models/sentiment_model.pkl')


hf_train = Dataset.from_pandas(train_df[['preprocessed','label_id']].rename(columns={'preprocessed':'text','label_id':'label'}))
hf_test  = Dataset.from_pandas(test_df[['preprocessed','label_id']].rename(columns={'preprocessed':'text','label_id':'label'}))
print(hf_train, hf_test)


model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

hf_train_t = hf_train.map(tokenize_fn, batched=True)
hf_test_t = hf_test.map(tokenize_fn, batched=True)

hf_train_t = hf_train_t.remove_columns(['text'])
hf_test_t = hf_test_t.remove_columns(['text'])

hf_train_t.set_format(type='torch')
hf_test_t.set_format(type='torch')

print('Tokenized datasets ready.')


model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir='./temp_trainer',
    num_train_epochs=2,             
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_total_limit=1
)

def compute_metrics_hf(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    weighted = report['weighted avg']
    return {'accuracy': (preds==labels).mean(),
            'precision': weighted['precision'],
            'recall': weighted['recall'],
            'f1': weighted['f1']}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_t,
    eval_dataset=hf_test_t,
    compute_metrics=compute_metrics_hf,
    tokenizer=tokenizer
)


print('To run DistilBERT training, uncomment trainer.train() in this cell.\n'
      'After training, the model and tokenizer can be saved to models/distilbert_sentiment/')


from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

model_dir = Path('models/distilbert_sentiment')
if model_dir.exists():
    tokenizer2 = DistilBertTokenizerFast.from_pretrained(model_dir)
    model2 = DistilBertForSequenceClassification.from_pretrained(model_dir)
    clf_pipe = pipeline('text-classification', model=model2, tokenizer=tokenizer2, return_all_scores=False)
    samples = test_df['preprocessed'].astype(str).tolist()[:100]
    preds = [int(clf_pipe(s)[0]['label'].split('_')[-1]) if 'LABEL_' in clf_pipe(s)[0]['label'] else              np.argmax(clf_pipe(s)[0]['score']) for s in samples]
    print('Loaded DistilBERT from models/distilbert_sentiment — run model.predict code suited to your saved model format.')
else:
    print('No DistilBERT saved model found at models/distilbert_sentiment. Train and save first.')
