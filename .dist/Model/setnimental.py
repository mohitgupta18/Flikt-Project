import joblib

# Load model
modelData = joblib.load("sentiment_model.pkl")
vectorizer = modelData["vectorizer"]
CLF = modelData["clf"]
Le = modelData["label_encoder"]

# Example prediction
text = ["The product is awesome but delivery was late."]
X = vectorizer.transform(text)
pred = CLF.predict(X)
sentiment = Le.inverse_transform(pred)[0]
print("Predicted sentiment:", sentiment)
