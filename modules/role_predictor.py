import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# load dataset
data = pd.read_csv("dataset/jobs.csv")

# combine skills column
X = data["skills"]
y = data["role"]

# convert text to numbers
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_vector, y)


def predict_role(resume_skills):

    # convert list to string
    resume_text = " ".join(resume_skills)

    resume_vector = vectorizer.transform([resume_text])

    role = model.predict(resume_vector)[0]

    return role