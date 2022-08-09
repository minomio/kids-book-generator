# Importing the required libraries
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize

import random 

# Function to extract the nouns 
word = ""
def ColorNounExtractor(text):
    color_list = ['pink', 'holographic','blue', 'yellow', 'rainbow', 'neon', 'red', 'green', 'purple', 'orange', 'turquoise', 'gold', 'silver']
    color = random.choice(color_list)
   # print('COLOR IS: ' + color)
   # print('NOUNS EXTRACTED :')
    
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        i = 0
        for (word, tag) in tagged:
            if tag == 'NN' and i == 0: # If the word is a noun
                textGenPromptOut = color + ' ' + word
                print (textGenPromptOut)
                i += 1
    return textGenPromptOut
        
ColorNounExtractor('The fat elephant was tired and fell asleep fast in town.')
