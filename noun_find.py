# Importing the required libraries
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize

# Function to extract the nouns 

def NounExtractor(text):
    
    print('NOUNS EXTRACTED :')
    
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'NN': # If the word is a proper noun
                print(word)
                
NounExtractor('The cat and the hat went for a walk in town.')