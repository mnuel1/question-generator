from pprint import pprint
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import PyPDF2
from Questgen import main

# nltk dependencies needed
# import nltk
# nltk.download('brown', quiet=True, force=True)
# nltk.download('stopwords', quiet=True, force=True)
# nltk.download('popular', quiet=True, force=True)

app = Flask(__name__)
CORS(app)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None
    return text

def generate_questions_based_on_type(input_text, question_type, num_questions):
    qg = main.QGen()
    qe = main.BoolQGen()

    payload = {
        "input_text": input_text,
        "max_questions" : num_questions
    }

    if question_type == 'short':
        questions = qg.predict_shortq(payload)
    elif question_type == 'boolean':
        questions = qe.predict_boolq(payload)
    elif question_type == 'mcq':
        questions = qg.predict_mcq(payload)
    else:
        return {"error": "Invalid question type"}, 400
   
    return questions

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files['pdf']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        pdf_path = temp_pdf.name
        pdf_file.save(pdf_path)

    input_text = extract_text_from_pdf(pdf_path).replace('\n', ' ')
    print(input_text)
    if not input_text:
        return jsonify({"error": "No text extracted from PDF"}), 400

    try:
        data = request.form if request.form else request.json
        question_type = data.get('question_type', 'short')
        num_questions = int(data.get('num_questions', 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input parameters"}), 400
    
    questions = generate_questions_based_on_type(input_text, question_type, num_questions)
    if not questions:
        return jsonify({"error": "No questions were generated"}), 404
    
    os.remove(pdf_path)

    return jsonify(questions), 200

if __name__ == "__main__":    
    app.run(debug=True, host='0.0.0.0', port=5000)