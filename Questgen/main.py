import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer, pipeline
import random
import spacy
import zipfile
import os
import json
from sense2vec import Sense2Vec
import requests
from collections import OrderedDict
import string
import pke
import nltk
import numpy 
from nltk import FreqDist
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from Questgen.encoding.encoding import beam_search_decoding
from Questgen.mcq.mcq import tokenize_sentences
from Questgen.mcq.mcq import get_keywords
from Questgen.mcq.mcq import get_sentences_for_keyword
from Questgen.mcq.mcq import generate_questions_mcq
from Questgen.mcq.mcq import generate_normal_questions
import time

class QGen:
    
    def __init__(self):

        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.nlp = spacy.load('en_core_web_sm')

        self.s2v = Sense2Vec().from_disk('s2v_old')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def predict_mcq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

   
        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping,self.device,self.tokenizer,self.model,self.s2v,self.normalized_levenshtein)

            except:
                return final_output
            end = time.time()

            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            final_output["time_taken"] = end-start
            
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return final_output
    
    def predict_shortq(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            print('ZERO')
            return final_output
        else:
            
            generated_questions = generate_normal_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.model)
            print(generated_questions)

            
        final_output["statement"] = modified_text
        final_output["questions"] = generated_questions["questions"]
        
        if torch.device=='cuda':
            torch.cuda.empty_cache()

        return final_output
            
  
    def paraphrase(self,payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        num = inp['max_questions']
        
        self.sentence= text
        self.text= "paraphrase: " + self.sentence + " </s>"

        encoding = self.tokenizer.encode_plus(self.text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length= 50,
            num_beams=50,
            num_return_sequences=num,
            no_repeat_ngram_size=2,
            early_stopping=True
            )

#         print ("\nOriginal Question ::")
#         print (text)
#         print ("\n")
#         print ("Paraphrased Questions :: ")
        final_outputs =[]
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != self.sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        
        output= {}
        output['Question']= text
        output['Count']= num
        output['Paraphrased Questions']= final_outputs
        
        for i, final_output in enumerate(final_outputs):
            print("{}: {}".format(i, final_output))

        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        return output


class BoolQGen:
       
    def __init__(self):        
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize a QA pipeline using a suitable model (e.g., BERT)
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        self.set_seed(42)
        
    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def random_choice(self):
        return random.choice([True, False])
    
    def predict_boolq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": 3
        }
        num_items = payload.get("max_questions")

        text = inp['input_text']
        num = inp['max_questions']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)
        
        questions_and_answers = []

        for _ in range(num):
            answer = self.random_choice()
            form = "truefalse: %s passage: %s </s>" % (modified_text, answer)
            encoding = self.tokenizer.encode_plus(form, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            attention_masks = encoding["attention_mask"].to(self.device)

            questions = beam_search_decoding(input_ids, attention_masks, self.model, self.tokenizer, num_items)
            

        output_array = {}
        output_array["questions"] = []
        for index, question in enumerate(questions):            
            individual_quest= {}
            qa_result = self.qa_pipeline(question=question, context=text)
                    
            if isinstance(qa_result, dict) and 'answer' in qa_result:
                answer_text = qa_result['answer']
            else:
                answer_text = None

            bool_answer = self.convert_answer_to_bool(answer_text)

            questions_and_answers.append((question, bool_answer))

            individual_quest['question']= question
            individual_quest["question_type"] = "boolean"
            individual_quest['right_answer']= bool_answer
            individual_quest["id"] = index+1

            output_array["questions"].append(individual_quest)

        if torch.device == 'cuda':
            torch.cuda.empty_cache()
        
        return output_array

    def convert_answer_to_bool(self, answer_text):
        """Convert the extracted answer text to 'true' or 'false' based on sentiment analysis."""
        
        if answer_text is None or answer_text.strip() == "":
            return "false"

        answer_text = answer_text.lower()

        sentiment_result = self.sentiment_pipeline(answer_text)

        if sentiment_result[0]['label'] == 'POSITIVE':
            return "true"
        elif sentiment_result[0]['label'] == 'NEGATIVE':
            return "false"

        return "false"
