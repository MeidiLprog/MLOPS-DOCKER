import re
import pandas as pd 
import spacy
import json
from ..config import SPACY_MODELS

nlp = None
data : pd.DataFrame = None

def initialise():
    with open('data/mini_review.json','r') as f:
        data = json.load(f)
    data = json.loads(data)
    data = pd.DataFrame(data)
    if isinstance(data,pd.DataFrame):
        en_sp = SPACY_MODELS["en"]
        nlp = spacy.load(en_sp)
    else:
        raise TypeError("data isn't a Data Frame\n")
    
    sample : pd.DataFrame = data.sample(frac=0.2,random_state=42,replace=False)
    return sample



