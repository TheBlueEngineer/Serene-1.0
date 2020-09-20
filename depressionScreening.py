import re
import langdetect
from expandContractions import expandContractions 
from contractions import contractions_dict
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [ self.wnl.lemmatize(t) for t in word_tokenize(doc) ]
# RETURN EXPLAINED:
# 0 = Non-depressive
# 1 = Depressive
# 2 = language error
# 3 = Empty textual data

def depressionScreening(text):
    # 1. We want to do a links checking, maybe the post is an entire link, we want it removed before langdetect
    text = re.sub(r"http\S+", " ", text)
    print(text)
    # 2. We need to expand the contractions now, otherwise we can't delete the punctuations
    text, x, y = expandContractions( text, contractions_dict) 
    print(text)
    # 3. Remove all the e-mail
    text = re.sub(r"\S*@\S*", " ", text)
    print(text)
    # 4. Remove punctuation             
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    print(text)
    
    if(text != "" and not text.isspace()):                                                                     
        text = re.sub(r"\s\s+", " ", text)
        # 5. Transform the data to lower case
        text = text.lower()
        print(text)
        
        text = {text}
        # Load the Vectorizer
        vectorizerName = "vector_nolimit.joblib"
        vectorizerPath = "./"
        vectorizer = load("%s%s" % ( vectorizerPath, vectorizerName))
    
        # Load the model
        modelName = "model_nolimit.joblib"
        modelPath = "./"
        model = load("%s%s" % ( modelPath, modelName))

        # Vectorize the text
        predictData = vectorizer.transform(text)
        print(predictData)
        # Predict the data
        answer = model.predict(predictData)

        return answer[0]       
    else:
        return 3       

if __name__ == '__main__':
    import sys
    depressionScreening(sys.argv[1])               