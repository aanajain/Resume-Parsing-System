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
    "Taylor Smith", "Alex Johnson", "Jordan Brown", "Casey Miller", "Morgan Davis",
    "Cameron Wilson", "Riley Moore", "Logan White", "Blake Harris", "Dylan Clark",
    "Avery Lewis", "Rowan Walker", "Quinn Hall", "Sydney Allen", "Peyton Young",
    "Charlie King", "Robin Lee", "Emery Scott", "Jamie Adams", "Kendall Wright"
]

# Expanded Skills List
SKILL_LIST = [
    "Python", "Java", "SQL", "C++", "Machine Learning", "Data Analysis", "Leadership",
    "Communication", "Project Management", "AWS", "React", "JavaScript", "Deep Learning",
    "Data Visualization", "Power BI", "Tableau", "R", "Statistics", "Neural Networks",
    "Big Data", "Spark", "Hadoop", "ETL", "AI", "NLP", "Cloud Computing", "Docker", 
    "Kubernetes", "Tableau", "Excel", "Data Science", "Web Development", "Agile",
    "Azure", "Data Engineering", "Scrum", "Cybersecurity", "Python Programming",
    "DevOps", "SQL Server", "Cloud Architecture", "Linux", "System Design"
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

@app.route('/feedback', methods=['POST'])
def feedback():
    # Log feedback
    feedback_text = request.form.get('feedback')
    print(f"User Feedback Received: {feedback_text}")
    
    # Redirect user to the Thank You page after submission
    return redirect(url_for('thank_you'))

@app.route('/thank-you')
def thank_you():
    # Render the thank you page
    return render_template('thank_you.html')

if __name__ == "__main__":
    app.run(debug=True)
