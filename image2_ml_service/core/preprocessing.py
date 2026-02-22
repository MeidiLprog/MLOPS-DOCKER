import re
import pandas as pd 
import spacy
import json
from ..config import SPACY_MODELS
import subprocess as sb
nlp = None
data : pd.DataFrame = None
from langdetect import detect

class textPreprocess:
    def __init__(self,lang : str = 'en'):
        self.lang = lang
        self.nlp = None
        self.data = None
        self.result = [] #result of processed
        self.data = self.initialise()

    def initialise(self):
        with open('data/mini_review.json','r') as f:
            data = json.load(f)
        data = json.loads(data)
        data = pd.DataFrame(data)
        if isinstance(data,pd.DataFrame):
            en_sp = SPACY_MODELS["en"]
            try:
                self.nlp = spacy.load(en_sp)
            except:
                sb.run(["python","-m","spacy","download","en_core_web_sm"],check=True)
                self.nlp = spacy.load(en_sp)
        else:
            raise TypeError("data isn't a Data Frame\n")
        
        sample : pd.DataFrame = data.sample(frac=0.2,random_state=42,replace=False)
        return sample


    def serieProcess(self,serie : str):

        text = serie.lower()
        text = re.sub(r'\u00A0', " ", text)  # Odd spaces 
        text = re.sub(r'http\S+', " ", text) # URLS
        text = re.sub(r"[­‐-‒–—]", "-", text)         
        text = re.sub(r"[“”«»]", '"', text)           
        text = re.sub(r"[‘’]", "'", text)           
        text = re.sub(r"[…]", "...", text)          
        text = re.sub(r"[^\w\s\-']", " ", text)
        text = re.sub("\s+"," ",text)
        
        #process with spacy
        doc = self.nlp(text)     

        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 1]
        return ' '.join(tokens) if tokens else None #just in case
    def langDetection(self,text : str):
        try:
            if pd.isna(text) or len(text) < 10:
                return None
            return detect(text)
        except:
            return None