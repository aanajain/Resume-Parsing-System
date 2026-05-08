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

with open("rf_classifier_categorization.pkl", "rb") as classifier_file:
    rf_classifier = pickle.load(classifier_file)

with open("tfidf_vectorizer_categorization.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

def clean_resume(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

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

def extract_name(text):
    name_match = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', text)
    return name_match[0] if name_match else "Name not found"

def extract_experience(text):
    experience_keywords = ["experience", "work history", "employment", "professional experience"]
    experience_pattern = re.compile(
        r"(?:(?:\d{4}(?:-\d{4})?)|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})[\s\S]+?(?=\n\S|\Z)"
    )
    lines = text.split("\n")
    experience_found = False
    experience_lines = []

    for line in lines:
        if any(keyword in line.lower() for keyword in experience_keywords):
            experience_found = True
            continue
        if experience_found:
            if line.strip() == "":
                break
            experience_lines.append(line.strip())

    experience_text = " ".join(experience_lines)
    matches = experience_pattern.findall(experience_text)
    return "\n".join(matches) if matches else "Work experience not found"

def predict_category_with_rf(resume_text):
    vectorized_text = tfidf_vectorizer.transform([resume_text])
    prediction = rf_classifier.predict(vectorized_text)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

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
    extracted_name = extract_name(resume_text)
    extracted_experience = extract_experience(resume_text)

    return render_template(
        'result.html',
        prediction=predicted_category,
        name=extracted_name,
        experience=extracted_experience,
    )

if __name__ == "__main__":
    app.run(debug=True)
