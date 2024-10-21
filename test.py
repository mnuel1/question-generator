from pprint import pprint
# import nltk
# nltk.download('stopwords')
from Questgen import main

# nltk.download('brown', quiet=True, force=True)
# nltk.download('stopwords', quiet=True, force=True)
# nltk.download('popular', quiet=True, force=True)

payload = {
            "input_text": "The daisy, a member of the large and diverse Asteraceae family, is widely admired for its iconic structure, consisting of composite flowers that include a central disc of tiny florets surrounded by petal-like ray florets. This simple yet elegant flower is commonly associated with temperate climates, where it thrives in a variety of environments, ranging from grassy meadows and forest edges to cultivated gardens. Although many think of daisies as purely decorative, they are also symbolic in various cultures, often representing innocence, purity, and new beginnings.The daisy family encompasses numerous species, but the most well-known is the common daisy or Bellis perennis. This species features white petals that encircle a bright yellow disc, a color scheme often linked with feelings of optimism and simplicity. While admired for its beauty, the common daisy is also known as a resilient lawn weed, flourishing in many landscapes despite attempts to control its growth. In addition to the common daisy, other species like the Oxeye daisy and the colorful Gerbera daisies add to the group’s diversity, with Gerbera varieties being especially popular in the floral industry due to their vibrant colors.Daisies reproduce through seed dispersal, allowing them to colonize new areas efficiently. Their flowers are a vital food source for various pollinators, especially bees, butterflies, and other beneficial insects, playing a critical role in maintaining the health of ecosystems. The nectar and pollen they provide help sustain these insects, which, in turn, support biodiversity by pollinating other plant species. Beyond their ecological role, some species of daisy have long been used in traditional medicine. For instance, Bellis perennis has been employed in folk remedies to treat wounds and skin irritations due to its mild anti-inflammatory properties. The aesthetic appeal of daisies is undeniable, and they are a popular choice for garden beds, bouquets, and decorative floral arrangements. Their bright colors and simple form make them a favorite among gardeners and flower enthusiasts. Despite being hardy and adaptable to various conditions, daisies also demonstrate resilience in less-than-ideal growing environments, contributing to their widespread presence across different regions. In many areas, they even display drought resistance, allowing them to survive in harsher climates with minimal care. Overall, daisies are much more than just a pretty flower. Their versatility, ecological importance, cultural symbolism, and medicinal uses highlight their significance in both natural ecosystems and human society. With their ability to adapt and thrive in a wide range of settings, daisies continue to hold a prominent place in the plant world.",
            "max_questions" : 5
        }
#1 
# qe= main.BoolQGen()
# output = qe.predict_boolq(payload)
# pprint (output)

qg = main.QGen()

2
output = qg.predict_mcq(payload)
pprint (output)

#3
# output = qg.predict_shortq(payload)
# pprint (output)

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