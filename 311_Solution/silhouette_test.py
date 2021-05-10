import operator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ## DATA GATHERING
# papers = pd.read_csv('papers.csv')
# test_paper = papers.loc[0]
# train_paper = papers.loc[1:]
# train_paper = train_paper.drop(columns=['id', 'event_type', 'pdf_name'], axis=1)

# ## CLEAN TEXT
# text_df = clean_text(train_paper, 'paper_text')


# ## TFIDF VECTORIZER
# vectorizer = TfidfVectorizer(max_df = 0.5, smooth_idf=True)
# X = vectorizer.fit_transform(text_df['clean_text'])

def silhouette_test(X, max_clusters = 20, min_clusters = 2):
    print("Using Max Cluster: {}\nUsing Min Cluster: {}".format(max_clusters, min_clusters))
    max_score = -1
    cluster_w_max_score = -1
    silhouette_dict = {}
    for n_clusters in range(max_clusters - min_clusters + 1):
        n_clusters = n_clusters + min_clusters
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For {} Clusters: The Avg Silhouette Score = {}".format(n_clusters, silhouette_avg))
        silhouette_dict[n_clusters] = silhouette_avg
    max_silhouette = max(silhouette_dict.items(), key=operator.itemgetter(1))
    print("Cluster with Max Score: {} -- Score: {}".format(max_silhouette[0], max_silhouette[1]))

