from pprint import pprint
# import nltk
# nltk.download('stopwords')
from Questgen import main

payload = {
            "input_text": "The daisy, belonging to the family Asteraceae, is a diverse group of flowering plants known for their characteristic composite flowers, which typically feature a central disc surrounded by petal-like ray florets. Commonly found in temperate regions, daisies thrive in a variety of habitats, from grasslands to meadows, and are often associated with symbolism of innocence and purity. The most recognized species, the common daisy (Bellis perennis), is notable for its white petals and yellow center and is often seen as a lawn weed in many areas. Daisies reproduce through seeds, and their flowers attract various pollinators, including bees and butterflies, making them vital to ecosystem health. Additionally, some species possess medicinal properties, and their aesthetic appeal makes them popular in gardens and floral arrangements. Their resilience and adaptability contribute to their widespread occurrence and significance in both natural and cultivated landscapes.",
            "max_questions" : 5
        }
# qe= main.BoolQGen()
# output = qe.predict_boolq(payload)
# pprint (output)


qg = main.QGen()

# output = qg.predict_mcq(payload)
# pprint (output)

output = qg.predict_shortq(payload)
pprint (output)

# answer = main.AnswerPredictor()
# payload4 = {
#     "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
#               India and a former captain of the Indian national team. He is widely regarded 
#               as one of the greatest batsmen in the history of cricket. He is the highest
#                run scorer of all time in International cricket.''',
#     "input_question" : "I am a hotdog? "
# }
# output = answer.predict_answer(payload4)
# print (output)