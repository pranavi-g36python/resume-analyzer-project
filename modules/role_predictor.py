import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "dataset", "jobs.csv"))

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