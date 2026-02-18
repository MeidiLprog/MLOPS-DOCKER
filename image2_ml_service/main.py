import os
import random as r
from flask import Flask

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
   html += "</table></body></html>"
   return html

if __name__ == "__main__":
   app.run(host='0.0.0.0',debug=True,port=5000)
