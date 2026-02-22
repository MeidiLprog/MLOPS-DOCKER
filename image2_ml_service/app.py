from flask import Flask, jsonify
from core.preprocessing import textPreprocess
from core.vectorizers import Vectorize
from core.clustering import kmeansClustering, GMM, Dbscan_alg, sil
import config

app = Flask(__name__)

preprocessor = textPreprocess()

df_processed = preprocessor.processData('review')  

@app.route('/')
def home():
    return jsonify({'service': 'ML service', 'status': "ok"})

@app.route('/cluster', methods=['GET'])
def cluster():
    
    texts = preprocessor.data['cleaned_data'].tolist()
    
    
    vect = Vectorize(method='tfidf', max_features=500)
    X = vect.fit_transform(texts)
    
    
    labels, model = kmeansClustering(X, n_cluster=3)
    score = sil(X, labels)
    
    return jsonify({
        'method': 'kmeans',
        'labels': labels.tolist()[:10],  # 10 premiers
        'silhouette': score,
        'n_clusters': len(set(labels)),
        'total_samples': len(texts)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host=config.IP, port=config.PORT)