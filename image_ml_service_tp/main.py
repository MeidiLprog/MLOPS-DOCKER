#FIRST SCRIPT + dockerfile IMG1 associated 
#Simple ML classification comparaison RF/LGREG

import os
import random as r
from flask import Flask, jsonify
import requests as req
from datetime import datetime
import numpy as np
import time as t
#SKLEARN imports

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,GridSearchCV
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
accuracy: float | None = None
precision: float | None = None
recall: float | None = None
auc: float | None = None
trained: datetime | None = None
results: dict | None = None


@app.route('/train')
def trainModel():
   global accuracy, precision, recall, auc, trained, results

   X,y = make_classification(n_samples=2000,n_features=20,n_informative=10,flip_y=0.02,random_state=42)
   
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
   cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
   model = {
      "Logreg" : LogisticRegression(max_iter=1000),
      "RF" : RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42,max_depth=20)
   }  
   scores : dict = {}
   for name,val in model.items():
      scores_cv = cross_val_score(val,X_train,y_train,cv=cv)
      #adding mean to dictionnary to automatically compare and gridsearch it later on
      mean_sc = scores_cv.mean()
      #dictionnary to store the result
      scores[name] = mean_sc

      print(f"{name} : {mean_sc:.4f}")

    
    
   best_mod = max(scores,key=scores.get)
   print(f"The best model was{best_mod}\n")
   #using the name associated to the value to retrieve the model Object
   model_to_use = model[best_mod] 
   
   #once retrieved it is time to carry out a grid search

   param_grids = {
    "Logreg": {
        "penalty" : ["l1","l2"],
        "C": [0.1, 1, 10],
        "solver" : ["liblinear"],
    },
    "RF": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    }
}
   grid = GridSearchCV(estimator=model_to_use,param_grid=param_grids[best_mod],cv=cv,n_jobs=-1)
   
   grid.fit(X_train,y_train)
   
   final_model = grid.best_estimator_

   y_pred = final_model.predict(X_test)
   y_prob = final_model.predict_proba(X_test)[:, 1]

   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred)
   recall = recall_score(y_test, y_pred)
   auc = roc_auc_score(y_test, y_prob)

   trained = datetime.now()





   results = {
      "best_model": best_mod,
      "best_params": grid.best_params_,
      "cv_mean": float(grid.best_score_),
      "cv_std": float(grid.cv_results_['std_test_score'][grid.best_index_]),
      "accuracy": float(accuracy),
      "precision": float(precision),
      "recall": float(recall),
      "auc" : float(auc),
      "trained": trained.strftime("%Y-%m-%d %H:%M:%S")
    }

   return f"""
    <html><body>
    <h1>Training complete</h1>

    <h2>Best Model: {results['best_model']}</h2>
    <p>Best Parameterss: {results['best_params']}</p>

    <h3>Cross-Validation(mn gridsearch)</h3>
    <p>Mean CV score: {results['cv_mean']:.3f}</p>
    <p>Std CV score: {results['cv_std']:.3f}</p>

    <h3>Test Metrics</h3>
    <p>Accuracy: {results['accuracy']:.3f}</p>
    <p>Precision: {results['precision']:.3f}</p>
    <p>Recall: {results['recall']:.3f}</p>
    <p>AUC: {results['auc']:.3f}</p>

    <br><a href="/results">View stored results</a>
    </body></html>
    """


@app.route('/results')
def results():
   global results
   if accuracy is None:
      return "No model has been trained \n"
   
   html = "<html><body>"
   html += "<h1>Model results: </h1><table border='1'>"
   html += "<tr><th>Metric</th><th>Values</th></tr>"
   for k,v in results.items():
      
      if isinstance(v,float):
         v = f"{v:.3f}"
      html += f"<tr><td>{k}</td><td>{v}</td></tr>"

   html += "</table>"
   html += '<br><a href="/">Back</a>'
   html += "</body></html>"

   return html

if __name__ == "__main__":
   app.run(host='0.0.0.0',debug=True,port=5000)
