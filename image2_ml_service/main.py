import os
import random as r
from flask import Flask, jsonify
import requests
from datetime import datetime

#SKLEARN imports

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score


app = Flask(__name__)

"""
Simple random number generator, one list, and assigning a random value to a <td> element in html
"""

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





@app.route('results')
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
