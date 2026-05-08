import os
import pickle
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pytesseract
import PyPDF2
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from spacy.lang.en import English
import textstat
from datetime import datetime

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Helper function to safely load pickled files
def load_pickle_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load ML Models and Vectorizers
rf_classifier_categorization = load_pickle_file("rf_classifier_categorization.pkl")
tfidf_vectorizer_categorization = load_pickle_file("tfidf_vectorizer_categorization.pkl")
rf_classifier_job_recommendation = load_pickle_file("rf_classifier_job_recommendation.pkl")
tfidf_vectorizer_job_recommendation = load_pickle_file("tfidf_vectorizer_job_recommendation.pkl")

# List of generated neutral names
generated_names = [
    "Emma Johnson", "Olivia Smith", "Ava Brown", "Isabella Davis", "Sophia Wilson",
    "Amelia Moore", "Harper White", "Evelyn Harris", "Abigail Clark", "Ella Lewis"
]

# Expanded Skills List
SKILL_LIST = [
    "Python", "Java", "SQL", "C++", "Machine Learning", "Data Analysis", "Leadership",
    "Communication", "Project Management", "AWS", "React", "JavaScript", "Deep Learning",
    "Data Visualization", "Power BI", "Tableau", "R", "Statistics", "Neural Networks",
    "Big Data", "Spark", "Hadoop", "ETL", "AI", "NLP", "Cloud Computing", "Docker",
    "Kubernetes", "Excel", "Data Science", "Web Development", "Agile", "Azure",
    "Data Engineering", "Scrum", "Cybersecurity", "DevOps", "Linux", "System Design"
]

# Helper Functions
def clean_resume(text):
    text = re.sub(r'\n+', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
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

def extract_experience(text):
    experience_patterns = [
        r'(\d+)\s*(?:years?|yrs)\s*of\s*experience',
        r'(\d+)\s*(?:years?|yrs)\b',
        r'(\d+)\s*(?:months?)\b'
    ]
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not available"

def extract_skills_with_regex(text):
    skills_pattern = r'\b(' + '|'.join(re.escape(skill) for skill in SKILL_LIST) + r')\b'
    skills_found = re.findall(skills_pattern, text, re.IGNORECASE)
    skills_normalized = list({skill.title() for skill in skills_found})
    return skills_normalized

def calculate_readability(text):
    return textstat.flesch_reading_ease(text)

def predict_category_with_rf(resume_text):
    vectorized_text = tfidf_vectorizer_categorization.transform([resume_text])
    prediction = rf_classifier_categorization.predict(vectorized_text)[0]
    return prediction

def predict_job_with_rf(resume_text):
    vectorized_text = tfidf_vectorizer_job_recommendation.transform([resume_text])
    prediction = rf_classifier_job_recommendation.predict(vectorized_text)[0]
    return prediction

def log_feedback(feedback_text, sentiment):
    with open("feedback_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()} - Feedback: {feedback_text} - Sentiment: {sentiment}\n")

# Routes
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
    recommended_job = predict_job_with_rf(resume_text)

    generated_name = random.choice(generated_names)
    extracted_experience = extract_experience(resume_text)
    extracted_skills = extract_skills_with_regex(resume_text)
    readability_score = calculate_readability(resume_text)

    return render_template(
        'result.html',
        prediction=predicted_category,
        recommended_job=recommended_job,
        name=generated_name,
        experience=extracted_experience,
        skills=extracted_skills,
        readability_score=readability_score
    )

@app.route('/skill-match', methods=['POST'])
def skill_match():
    if 'resume' not in request.files:
        return "No file part", 400
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file', 400

    resume_text = extract_text(file)
    if not resume_text.strip():
        return "Unable to extract text from the file. Please upload a valid file.", 400

    resume_text = clean_resume(resume_text)
    extracted_skills = extract_skills_with_regex(resume_text)

    # Assume job_skills are provided or fetched from a database
    job_skills = {"Python", "Machine Learning", "SQL"}
    matching_skills = set(extracted_skills).intersection(job_skills)
    match_score = len(matching_skills) / len(job_skills) * 100

    return jsonify({
        "extracted_skills": extracted_skills,
        "matching_skills": list(matching_skills),
        "match_score": match_score
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form.get('feedback')
    if not feedback_text:
        return "No feedback provided", 400

    # Analyze sentiment
    nlp = English()
    doc = nlp(feedback_text)
    sentiment = "Positive" if "good" in feedback_text.lower() else "Negative"

    log_feedback(feedback_text, sentiment)
    print(f"User Feedback: {feedback_text} | Sentiment: {sentiment}")  # Log feedback
    return redirect(url_for('thank_you'))

@app.route('/thank-you')
def thank_you():
    return render_template('thank_you.html')

if __name__ == "__main__":
    app.run(debug=True)
