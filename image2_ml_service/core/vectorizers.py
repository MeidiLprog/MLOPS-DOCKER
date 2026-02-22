from sklearn.feature_extraction import TfidfVectorizer, CountVectorizer
import numpy as np
import os
from typing import Tuple,List,Optional


class Vectorize:
    def __init__(self,method : str = "tfidf", max_features=500):

        self.model = None
        self.max_features = max_features
        self.method = method
        self.features_names = None #quite useful to retrieve the words


    def fit_transform(self, text : List[str]) -> np.ndarray:
        
        if self.method == "tfidf":
            self.model = TfidfVectorizer(max_features=self.max_features)
        elif self.method == "count":
            self.model = CountVectorizer(max_features=self.max_features)
        else:
            raise ValueError("You may only pick a model amongst tfidf or countvectorizer \n")
        X = self.model.fit_transform(text)
        self.features_names = self.model.get_feature_names_out()
        return X.toarray()

    def transform(self,text : List[str]) -> np.ndarray:

        if not self.model == None:
            X = self.model.transform(text)
            return X.ndarray()
        else:
            raise ValueError("Model cannot be none \n")

    

