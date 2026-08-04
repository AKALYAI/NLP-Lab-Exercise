from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Lists to store documents and labels
docs = []
labels = []

# Input
n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))
    labels.append(input(f"Enter category {i + 1}: "))

# -----------------------------
# Rule-Based Classification
# -----------------------------
rule_pred = []

for doc in docs:
    doc = doc.lower()

    if "contract" in doc:
        rule_pred.append("contract")
    elif "judgment" in doc:
        rule_pred.append("judgment")
    else:
        rule_pred.append("agreement")

# Calculate Rule-Based Accuracy
rule_acc = accuracy_score(labels, rule_pred)

# -----------------------------
# Maximum Entropy (Logistic Regression)
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

ml_pred = model.predict(X)

# Calculate ML Accuracy
ml_acc = accuracy_score(labels, ml_pred)

# -----------------------------
# Results
# -----------------------------
print("\nRule-Based Predictions:")
for doc, pred in zip(docs, rule_pred):
    print(f"Document: {doc}")
    print(f"Predicted Category: {pred}\n")

print("Rule-Based Accuracy:", rule_acc)

print("\nMaximum Entropy Predictions:")
for doc, pred in zip(docs, ml_pred):
    print(f"Document: {doc}")
    print(f"Predicted Category: {pred}\n")

print("Maximum Entropy Accuracy:", ml_acc)