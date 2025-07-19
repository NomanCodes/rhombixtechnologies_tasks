from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Stemmer
port_stem = PorterStemmer()

# Text cleaning
def clean_text(content):
    stemmed = re.sub('[^a-zA-Z]', ' ', content)
    stemmed = stemmed.lower().split()
    stemmed = [port_stem.stem(word) for word in stemmed if word not in stopwords.words('english')]
    return ' '.join(stemmed)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    cleaned = clean_text(tweet)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)

    sentiment = 'Positive 😀' if prediction[0] == 1 else 'Negative 😠'
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
