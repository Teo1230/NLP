import os
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import roner
import re
import joblib
from itertools import product
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 2000000  # Or any value greater than your maximum text length

ner = roner.NER()


def preprocess_text_ro(text):
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\b(\w+?)\b', r'\1 ', text)
    doc = nlp_ro(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text


def load_text_data():
    path = os.getcwd()
    path_centuries = os.path.join(path, 'century')
    txt = {}
    for century in os.listdir(path_centuries):
        path_century = os.path.join(path_centuries, century)
        txt[century] = []
        for txt_file in os.listdir(path_century):
            file_path = os.path.join(path_century, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                txt[century].append(preprocess_text_ro(text))
    return txt


def vectorize_text_data(txt, max_features=1000):
    corpus = []
    labels = []
    named_entities = []
    for century, texts in txt.items():
        for text in texts:
            named_entities_text = ner(text)
            named_entities.append(' '.join([entity['text'] for entity in named_entities_text]))
            corpus.append(text)
            labels.append(century)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    X_named_entities = vectorizer.transform(named_entities)
    X_combined = np.hstack((X.toarray(), X_named_entities.toarray()))
    y = np.array(labels)
    return X_combined, y, vectorizer


def train_classifier(X, y,model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    if model == 'LinearSVC':
        clf = LinearSVC(C=0.5, loss='squared_hinge', penalty='l2', multi_class='ovr', random_state=15)
    elif model == 'KNeighbors':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif model == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(max_depth=3)
    elif model == 'MultinomialNB':
        clf = MultinomialNB()
    elif model == 'GaussianNB':
        clf = GaussianNB()
    else:
        raise ValueError("Invalid model name!")
    

    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Weighted accuracy:", weighted_accuracy(y_test, y_pred))


def weighted_accuracy(y_true, y_pred):
    distances = np.abs(y_true.astype(int) - y_pred.astype(int))
    partial_quotients = 1 / 2 ** distances
    return np.sum(partial_quotients) / len(y_true)


def predict_century(new_text, vectorizer, clf):
    new_text = preprocess_text_ro(new_text)
    named_entities_new = ner(new_text)
    named_entities_text = ' '.join([entity['text'] for entity in named_entities_new])
    new_text_vectorized = vectorizer.transform([new_text])
    named_entities_vectorized = vectorizer.transform([named_entities_text])
    combined_features = np.hstack((new_text_vectorized.toarray(), named_entities_vectorized.toarray()))
    predicted_century = clf.predict(combined_features)
    return predicted_century[0]


if __name__ == "__main__":
    txt = load_text_data()

    X, y, vectorizer = vectorize_text_data(txt)

    clf, X_test, y_test = train_classifier(X, y,'LinearSVC')

    model_filename = 'text_classification_model_knn_7_sample.pkl'
    vectorizer_filename = 'tfidf_vectorizer_model_knn_7_sample.pkl'

    joblib.dump(clf, model_filename)

    joblib.dump(vectorizer, vectorizer_filename)

    print("Model and vectorizer saved successfully.")

    evaluate_model(clf, X_test, y_test)

    example = """ ce se petrece... Şi ne pomenim într-una din zile că părintele vine la şcoală şi ne aduce un scaun nou şi
lung, şi
după ce-a întrebat de dascăl, care cum ne purtăm, a stat puţin pe gânduri, apoi a pus nume scaunului -
Calul Balan
şi l-a lăsat în şcoală.
În altă zi ne trezim că iar vine părintele la şcoală, cu moş Fotea, cojocarul satului, care ne aduce, dar
de şcoală
nouă, un drăguţ de biciuşor de curele, împletit frumos, şi părintele îi pune nume - Sfântul Nicolai,
după cum este
şi hramul bisericii din Humuleşti... Apoi pofteşte pe moş Fotea că, dacă i-or mai pica ceva curele
bune, să mai facă
aşa, din când în când, câte unul, şi ceva mai grosuţ, dacă se poate... Bădiţa Vasile a zâmbit atunci,
iară noi, şcolarii,
am rămas cu ochii holbaţi unii la alţii. Şi a pus părintele pravilă şi a zis că în toată sâmbăta să se
procitească băieţii
şi fetele, adică să asculte dascălul pe fiecare de tot ce-a învăţat peste săptămână; şi câte greşeli va
face să i le
însemne cu cărbune pe ceva, iar la urma urmelor, de fiecare greşeală să-i ardă şcolarului câte un
sfânt-Nicolai.
Atunci copila părintelui, cum era sprinţară şi plină de incuri, a bufnit în râs. Păcatul ei, sărmana! - Ia,
"""
    loaded_model = joblib.load(model_filename)

    loaded_vectorizer = joblib.load(vectorizer_filename)


    predicted_century = predict_century(example, loaded_vectorizer, loaded_model)
    print("Predicted century:", predicted_century)


