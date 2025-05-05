import streamlit as st
import joblib
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)

# Streamlit UI
st.title("📰 Fake News Detection App")

news = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    cleaned_news = preprocess(news)
    data = vectorizer.transform([cleaned_news])
    prediction = model.predict(data)

    result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
    st.subheader("Prediction Result:")
    st.success(result)
