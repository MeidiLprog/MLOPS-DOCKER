from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def kmeansClustering(X, n_cluster=3):
    model = KMeans(n_clusters=n_cluster,random_state=42)
    labels = model.fit_predict(X)
    return labels,model

def GMM(X,n_components=3):
    model = GaussianMixture(n_components=n_components,random_state=42)
    labels = model.fit_predict(X)
    return labels,model

def Dbscan_alg(X,eps=0.5,min_samples=5):
    model = DBSCAN(eps=eps,min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels,model
def sil(X,labels):
    if len(set(labels)) > 1:
        return silhouette_score(X,labels)
    return 0.0