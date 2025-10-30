import csv
import random
import uuid
from datetime import datetime,timedelta

POS  = [
    "Great product really loved the user experience",
    "Amazing support - quick and helpful replies",
    "Easy to use and intutive . highly recommend",
    "fantastic value for money , exceeded expectations"]

NEG = [
    "very disappointed - app crashes every time i try to save",
    "terrible support , no answer for days",
    "feature incomplete , missing key functionally we expected",
    "performace is slow and laggy on mobile"
]

NEU = [
    "its okay,does what it says . nothing special",
    "average experience. could be improved in some areas",
    "neutral about the product-neither great nor poor",
    "met basic needs but lacks advanced features"
]
sources = ["email","chat","twitter","facebook","googleplay","app_store"]
countries = ["india","USA","UK","Germany","Australia"]

def feedback(sentiment_label):
    base = random.choice({'pos':POS,'neg':NEG,'neu':NEU}[sentiment_label])
    
    noise = ""
    if random.random()<0.3:
        noise = " "+" ".join(random.choices(["!!!","please fix","tnx","pls",":)","worst"],k = random.randint(1,3)))
        
    if random.random()<0.2:
        base += " "+base
        
    if random.random()<0.2:
        base = base.replace(" "," # ")
    return base + noise

def gener(n = 2000, outpath = "data/simulated_feedback.csv"):
    random.seed(42)
    header = ["feedback_id","timestamp","source","country","customer_id","feedback_text","sentiment_label"]
    start  = datetime.now() - timedelta(days = 365)
    with open(outpath,"W",newline = '',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for _ in range(n):
            sid = str(uuid.uuid4())
            ts = (start + timedelta(days=random.randint(0,365), seconds=random.randint(0,86400))).isoformat()
            src = random.choice(sources)
            country = random.choice(countries)
            cid = "CUST" + str(random.randint(1000,9999))
            r = random.random()
            if r < 0.45:
                label = "positive"
            elif r < 0.80:
                label = "neutral"
            else:
                label = "negative"
            text = feedback(label[:3])
            # random missingness
            if random.random() < 0.01:
                text = ""  # missing feedback
            writer.writerow([sid, ts, src, country, cid, text, label])

if __name__ == "__main__":
    gener(2000, outpath="../data/simulated_feedback.csv")
    print("Generated ../data/simulated_feedback.csv")