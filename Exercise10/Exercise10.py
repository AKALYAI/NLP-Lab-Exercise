from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Input posts
posts = []

n = int(input("Enter number of posts: "))

for i in range(n):
    post = input(f"Enter post {i + 1}: ")
    posts.append(post)

# Input number of clusters
k = int(input("Enter number of clusters: "))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

# K-Means Clustering
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_

# Display cluster results
print("\n========== CLUSTER RESULTS ==========\n")

for i in range(len(posts)):
    print("Post:", posts[i])
    print("Cluster:", labels[i])
    print()

# Display important keywords
terms = vectorizer.get_feature_names_out()

print("========== IMPORTANT KEYWORDS ==========\n")

for i in range(k):
    center = model.cluster_centers_[i]
    top = center.argsort()[-5:]

    print("Cluster", i)
    for j in top[::-1]:   # Print highest-weight keywords first
        print(terms[j])
    print()

# Marketing insight
print("========== MARKETING INSIGHT ==========")
print("• Similar customer opinions are grouped together.")
print("• Clusters help identify product trends and issues.")