# src/summarization.py
from transformers import pipeline
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Abstractive
def AbstractiveSummarize(texts, min_length=20, max_length=80):
    Summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    Joined = "\n".join(texts) if isinstance(texts, list) else texts
    Sumry = Summarizer(Joined, max_length=max_length, min_length=min_length, do_sample=False)
    return Sumry[0]['summary_text']

def extractivesummary(text, top_n_short=1, top_n_long=3):
    if not text or len(text.split())<30:
        return text
    sents = sent_tokenize(text)
    if len(sents) <= top_n_long:
        return {"short": text, "detailed": text}
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sents)
    cos = cosine_similarity(X)
    scores = cos.mean(axis=1)
    ranked = np.argsort(scores)[::-1]
    short = " ".join([sents[i] for i in ranked[:top_n_short]])
    detailed = " ".join([sents[i] for i in ranked[:top_n_long]])
    return {"short": short, "detailed": detailed}

if __name__ == "__main__":
    sample = """Your long feedback text here..."""
    print(extractivesummary(sample))
