from utils.utils import plot_TSNE
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans


def plot_distance(X, Y, num, plot_prefix):
    dis = []
    idx = []
    for i in range(num-1):
        dis.append(np.linalg.norm(X[i]-X[i+1]))
        if Y[i] == Y[i+1]:
            idx.append("same")
        else:
            idx.append("change")
    time = range(len(dis))
    data = pd.DataFrame({
        "distance": dis,
        "idx": idx,
        "time": time
    })
    sns.set(rc={'figure.figsize':(20, 5)})
    sns_plot = sns.scatterplot(data=data, x="time", y="distance", hue="idx")
    fig = sns_plot.get_figure()
    fig.savefig(plot_prefix + "_distance.png")


def K_means(X, Y, num, plot_prefix):
    kmeans = KMeans(n_clusters=num)
    kmeans.fit(X)

    y_kmeans = kmeans.predict(X)
    
    sns_plot = sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=y_kmeans)
    fig = sns_plot.get_figure()
    fig.savefig(plot_prefix + "_predict.png")
    
    sns_plot = sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=Y)
    fig = sns_plot.get_figure()
    fig.savefig(plot_prefix + "_truth.png")
    
if __name__ == "__main__":
    TAG = "6"
    demo_file = ["./Expert/HandCollect/" + TAG + "/expert_demo_1.json",  "./Expert/HandCollect/" + TAG + "/expert_demo_2.json", "./Expert/HandCollect/" + TAG + "/expert_demo_3.json", "./Expert/HandCollect/" + TAG + "/expert_demo_4.json", "./Expert/HandCollect/" + TAG + "/expert_demo_5.json"]
    
    X = []
    Y = []
            
    for file in demo_file:
        with open(file, 'r') as fin:
            json_file = json.load(fin)
            ac = json_file["actions"]
            ob = json_file["observations"]
            idx_truth = json_file["idx"]
           
        fin.close()
        for action in ac:
            X.append(np.array(action))
            Y.append(idx_truth)
            
    X = np.stack(X)
    Y = np.array(Y)
        
    # plot_TSNE(X, Y, "./" + TAG, num_color=3)
    K_means(X, Y, 3, "./" + TAG)