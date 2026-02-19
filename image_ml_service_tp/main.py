import os
import random as r
from flask import Flask, jsonify
import requests as req
from datetime import datetime
import numpy as np
import time as t
#SKLEARN imports

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score


app = Flask(__name__)

"""
Simple random number generator, one list, and assigning a random value to a <td> element in html
"""

def timer(func):
   def wrapper(*args,**kwargs):
    start = t.time()
    fn = func(*args,**kwargs)
    end = t.time() - start
    f"Time consumed {end:.3f}s\n"
    return fn
   return wrapper
#root directory
@app.route('/')
def randitup() -> str:
   nb  = [(r.randint(1,1000) % 10) for _ in range(10)]
   
   html = "<html><body><table>"
   for i in nb:
      html += f"<tr><td>{i}</td></tr>"
   html += "</body></html>"
   html += '<br><a href="/train">Train model</a>'
   html += '<br><a href="/results">Results</a>'
   html += '<br><a href="/call-ml">Test an ML service</a>'

   html += "</body></html>"
   return html

#global variables needed for the resultats
accuracy : float = None
precision : float = None
recall : float = None

trained = None

@app.route('/train')
def trainModel():
   global accuracy,precision,recall

   X,y = make_classification(n_samples=2000,n_features=20,n_informative=10,flip_y=0.02,random_state=42)
   
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
   cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
   model = {
      "Logreg" : LogisticRegression(max_iter=1000),
      "RF" : RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42)
   }  
   scores : dict = {}
   for name,val in model.items():
      scores_cv = cross_val_score(val,X_train,y_train,cv=cv)
      #adding mean to dictionnary to automatically compare and gridsearch it later on
      mean_sc = scores_cv.mean()
      scores[name] = mean_sc

      print(f"{name} : {scores.mean():.4f}\n")

   

   model.fit(X_train,y_train)
   
   y_pred = model.predict(X_test)
   y_prob = model.predict_proba(X_test)[:,1]
   
   results = {
      "CV_AUC_MEAN" : float(scores.mean()),
      "CV_AUC_VAR" : float(scores.std()),
      "test_accucary" : float(accuracy := accuracy_score(y_test,y_pred)),
      "test_precision" : float(precision := precision_score(y_test,y_pred)),
      "test_recall" : float(recall := recall_score(y_test,y_pred)), 
      "test_auc" : float(roc_auc_score(y_test,y_prob)),
   }
   
   return f"""



   """
@app.route('/results')
def results():
   if accuracy is None:
      return "No model has been trained \n"
   return f"""
   <html><body>
      <h1>Results</h1>
      <p>Accuracy: {accuracy:.3f}</p>
      <p>Precision: {precision:.3f}</p>
      <p>Recall: {recall:.3f}</p>
      <p>Timestamp: {trained:.3f}</p>
   </body></html>
   """



if __name__ == "__main__":
   app.run(host='0.0.0.0',debug=True,port=5000)
