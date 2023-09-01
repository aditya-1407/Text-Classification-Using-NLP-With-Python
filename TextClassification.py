import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('firstfilework/SMS_Dataset.csv', encoding='ISO-8859-1')
# print(df.head())

nltk.download('punkt')
nltk.download('stopwords')

def processText(text):
    words = word_tokenize(text=text.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

X = df['Message_body']
y = df['Label']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = [
    ('Naive Bayes', MultinomialNB()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Gradient Boost', GradientBoostingClassifier()),
    ('Logistic Regression', LogisticRegression())
]

for name, classifier in classifiers:
    print(f'Training {name} Classifer')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('classifier', classifier)
    ])
    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTest)
    print(f'{name} Classifier Accuracy :', accuracy_score(yTest, yPred))
    print(classification_report(yTest, yPred))
    joblib.dump(pipeline, f'{name}_model.pkl')
    print(f'{name} Classifier Saved as {name}_model.pkl')