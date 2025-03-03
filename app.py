from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer (assuming they are saved as pickles)
try:
    with open('model/spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or vectorizer files not found. Make sure they exist in the correct directory.")
    exit()

nltk.download('stopwords')
nltk.download('punkt')  # Download punkt for tokenization

app = Flask(__name__)


# Preprocessing function (IMPORTANT: Must match your training preprocessing)
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)  # Use nltk tokenizer
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render your frontend template


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({'error': 'No email text provided'}), 400

        email_text = data['email']
        preprocessed_text = preprocess_text(email_text)

        # Vectorize the preprocessed text
        text_vectorized = tfidf_vectorizer.transform([preprocessed_text])

        # Make prediction (assuming your model expects a vectorized input)
        prediction = model.predict(text_vectorized)[0]

        return jsonify({'is_spam': bool(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production