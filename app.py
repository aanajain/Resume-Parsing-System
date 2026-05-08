import os
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pytesseract
import PyPDF2
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load the Random Forest classifier and TfidfVectorizer
with open("rf_classifier_categorization.pkl", "rb") as classifier_file:
    rf_classifier = pickle.load(classifier_file)

with open("tfidf_vectorizer_categorization.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Configure pytesseract if required
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'  # Update with your Tesseract OCR path

# Helper function to clean the resume text
def clean_resume(text):
    text = re.sub(r'\n+', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Function to extract text from files
def extract_text(file):
    try:
        if file.filename.endswith(".txt"):
            return file.read().decode("utf-8")
        elif file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif file.filename.endswith((".jpeg", ".jpg", ".png")):
            image = Image.open(file)
            return pytesseract.image_to_string(image)
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Prediction function
def predict_category_with_rf(resume_text):
    vectorized_text = tfidf_vectorizer.transform([resume_text])
    prediction = rf_classifier.predict(vectorized_text)[0]
    return prediction

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Resume upload and prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file part"
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file'

    resume_text = extract_text(file)
    if not resume_text.strip():
        return "Unable to extract text from the file. Please upload a valid file."

    resume_text = clean_resume(resume_text)
    predicted_category = predict_category_with_rf(resume_text)

    return render_template('result.html', prediction=predicted_category)

if __name__ == "__main__":
    app.run(debug=True)
