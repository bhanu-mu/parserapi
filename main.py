from flask import Flask, request
import re
import pandas as pd
import spacy
from spacy.matcher import Matcher
from pyresparser import ResumeParser
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx2txt
import json
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

import string
import csv

app = Flask(__name__)

@app.route('/parse_resume', methods=['POST'])


def parse_resume():

    if 'file' not in request.files:
        return {'message': 'No file uploaded'}, 400
    
    file = request.files['file']

        # Check if the file is empty
    if file.filename == '':
        return {'message': 'Empty file uploaded'}, 400
    
    filename = secure_filename(file.filename)
    file.save(filename)
     
     # Check the file extension
    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        resume_text = ''
        for page in range(len(pdf_reader.pages) ):
            resume_text += pdf_reader.pages[page].extract_text()
    elif filename.endswith('.docx'):
        resume_text = docx2txt.process(file)
    else:
        # Unsupported file type
        os.remove(filename)  # Remove the temporary file
        return {'message': 'Unsupported file type'}, 400   
    
    proc_data=call_to_clean(resume_text)
    name = extract_name(proc_data)
    email = extract_email(resume_text)
    phone_number = extract_phone_number(resume_text)
    years_of_experience = extract_years_of_experience(resume_text)
    skills_file = 'skills.csv'
    skills = extract_skills(filename, skills_file)
    
    
    parsed_data = {
        'Name': name,
        'Email': email,
        'Phone Number': phone_number,
        'Experience':years_of_experience,
        'Skills':skills
        
    }
    
    os.remove(filename)  # Remove the temporary file
    json_data= json.dumps(parsed_data)
    return json_data



def clean_data(resume_text):
    lemmetizer = WordNetLemmatizer()
    re.sub(r'[\d]', '', resume_text)
    re.sub(r'[^a-zA-Z]', '', resume_text)
    re.sub('\s+', ' ', resume_text)
    text_clean = []
    text_tokens = word_tokenize(resume_text)

    for word in text_tokens:
        if (word not in string.punctuation):
            stem_word = lemmetizer.lemmatize(word)
            text_clean.append(stem_word)

    list_to_str = ' '.join([str(ele) for ele in text_clean])
    return list_to_str.lower()
    
    
    

def call_to_clean(resume_text):
    clean_text = []
    sentences = re.split(r'\n+', resume_text)
    sentence_df = pd.DataFrame(sentences, columns=['Text'])
    sentence_df['Text'] = sentence_df['Text'].apply(clean_data)
    clean_text.append(' '.join(sentence_df['Text']))
    return clean_text[0]




def extract_name(text):
    # Implement your name extraction logic here
    nlp = spacy.load('en_core_web_sm')
    nlp_text = nlp(text)
    matcher = Matcher(nlp.vocab)

        # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern], None)

    matches = matcher(nlp_text)

    for match_id,start, end in matches:
            span = nlp_text[start:end]
            return span.text
    
    

def extract_email(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    if email:
        return email[0]
    else:
        return None

def extract_phone_number(text):
    phone = re.findall(r'\d{10}', text)
    phone = phone[0] if phone else None
    return phone

def extract_years_of_experience(resume_text):
    # Define regex pattern to match years of experience
    pattern = r'\b(\d+)\s*(?:year[s]?\b|YR[s]?\b)'

    # Search for the pattern in the resume text
    matches = re.findall(pattern, resume_text, re.IGNORECASE)

    # Calculate the total years of experience
    total_years = sum(int(match) for match in matches)

    if total_years:
        return total_years
    else:
        return None
    

def extract_skills(resume_file, skills_file):
    # Load the English language model in spaCy
    # Process the resume file with spaCy
    # Extract skills from the resume file using spaCy

    # Read the skills from the CSV file
    skills = pd.read_csv(skills_file,header=None)

    # Extract skills from the resume file using ResumeParser
    data_skills = ResumeParser(resume_file).get_extracted_data()['skills']
    data_skills = ','.join([str(ele) for ele in data_skills])

    return data_skills

    
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3003)
