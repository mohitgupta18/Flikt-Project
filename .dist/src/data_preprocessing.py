
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import os


nltk.download('stopwords')
nltk.download('punkt')

try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except:
    import subprocess
    subprocess.run(["python","-m","spacy","download","en_core_web_sm"])
    NLP = spacy.load("en_core_web_sm", disable=["ner"])

STOPWORDS = set(stopwords.words('english'))

def cleantext(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip()=="":
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[#@]', '', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocesstext(text, lemmatize=True):
    text = cleantext(text)
    if text == "":
        return ""
    doc = NLP(text.lower())
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.is_stop:
            continue
        if tok.text in STOPWORDS:
            continue
        if lemmatize:
            lemma = tok.lemma_.strip()
            if lemma:
                tokens.append(lemma)
        else:
            tokens.append(tok.text)
    return " ".join(tokens)

def main(infile="../data/simulated_feedback.csv", outfile="../data/cleaned_feedback.csv"):
    df = pd.read_csv(infile)
    initial = len(df)
    df = df.drop_duplicates(subset=["feedback_text","customer_id","timestamp"])
    df['feedback_text'] = df['feedback_text'].astype(str)
    tqdm.pandas()
    df['clean_text'] = df['feedback_text'].progress_apply(cleantext)
    df = df[df['clean_text'].str.strip()!='']
    df['preprocessed'] = df['clean_text'].progress_apply(preprocesstext)
    df = df[df['preprocessed'].str.split().str.len()>=2]
    df['sentiment_label'] = df['sentiment_label'].fillna('neutral')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"Initial rows: {initial}, after cleaning: {len(df)}")
    return df

if __name__ == "__main__":
    df = main()
