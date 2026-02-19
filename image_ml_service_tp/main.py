import os
import random
from flask import Flask

app = Flask(__name__)

@app.route('/')
def entry():
    nb = [(random.randint(1,100)%15) for _ in range(10)]

    html = "<html><body><table>"
    for i in nb:
      html += f"<tr><td>{i}</td></tr>"
    html += "</table></body></html>"
    return html

if __name__ == "__main__":
   app.run(host='0.0.0.0',debug=True,port=5000)







