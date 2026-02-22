from flask import Flask,request,jsonify
import core.preprocessing
import core.clustering
import core.vectorizers
import config

app = Flask(__name__)

preprocess = core.preprocessing.textPreprocess()#called upon my object to access methods to handle NLP stuff

@app.route('/')
def home():
    return jsonify({'service': 'ML service', 'status':"ok"})

@app.route('/cluster',methods=["POST"])
def cluster():
    data = request.get_json()
    text = data.get('texts',[])
    






if __name__ == "__main__":
    app.run(host=config.IP,port=config.PORT)
