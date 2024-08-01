import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Titre de l'application
st.title('Analyse des Maladies Cardiaques')

# Charger les données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=column_names)
data = data.replace('?', np.nan).dropna().astype(float)

# Convertir la cible en binaire
data['target'] = np.where(data['target'] > 0, 1, 0)

# Affichage des données
st.subheader('Données')
st.write(data.head())

# Affichage des informations du DataFrame
st.subheader('Informations sur le DataFrame')
info_buffer = []
def capture_info():
    from io import StringIO
    buffer = StringIO()
    data.info(buf=buffer)
    return buffer.getvalue()

st.text(capture_info())

# Description statistique des données
st.subheader('Description des données')
st.write(data.describe())

# Graphiques
st.subheader('Graphique : Présence de maladie cardiaque par sexe')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='sex', hue='target', data=data, ax=ax, palette={0: 'blue', 1: 'red'})
ax.set_xlabel('Sexe (1=Homme, 0=Femme)')
ax.set_ylabel('Nombre d\'individus')
ax.set_title('Présence de maladie cardiaque par sexe')
st.pyplot(fig)

st.subheader('Graphique : Type de douleur thoracique et présence de maladie cardiaque')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='cp', hue='target', data=data, ax=ax, palette={0: 'blue', 1: 'red'})
ax.set_xlabel('Type de douleur thoracique')
ax.set_ylabel('Nombre d\'individus')
ax.set_title('Type de douleur thoracique et présence de maladie cardiaque')
st.pyplot(fig)

st.subheader('Graphique : Glycémie à jeun et présence de maladie cardiaque')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='fbs', hue='target', data=data, ax=ax, palette={0: 'blue', 1: 'red'})
ax.set_xlabel('Glycémie à jeun > 120 mg/dl (1=Vrai, 0=Faux)')
ax.set_ylabel('Nombre d\'individus')
ax.set_title('Glycémie à jeun et présence de maladie cardiaque')
st.pyplot(fig)

st.subheader('Graphique : Angine induite par l\'exercice et présence de maladie cardiaque')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='exang', hue='target', data=data, ax=ax, palette={0: 'blue', 1: 'red'})
ax.set_xlabel('Angine induite par l\'exercice (1=Oui, 0=Non)')
ax.set_ylabel('Nombre d\'individus')
ax.set_title('Angine induite par l\'exercice et présence de maladie cardiaque')
st.pyplot(fig)

# Préparation des données
array = data.values
X = array[:, :-1]
y = array[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement et évaluation des modèles
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = {}

for model_name, model in models.items():
    st.subheader(f'{model_name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Afficher le rapport de classification
    labels = [0, 1]
    report = classification_report(y_test, y_pred, labels=labels, target_names=["Pas de Maladie", "Maladie"])
    st.text(report)
    
    # Calculer l'AUC-ROC
    try:
        if len(np.unique(y_test)) == 2 and np.all((y_pred_prob >= 0) & (y_pred_prob <= 1)):
            auc_score = roc_auc_score(y_test, y_pred_prob)
            st.write(f"AUC-ROC: {auc_score:.2f}")
            results[model_name] = auc_score
        else:
            st.write("Erreur : Les données pour AUC-ROC ne sont pas correctement configurées.")
    except ValueError as e:
        st.write(f"Erreur lors du calcul de l'AUC-ROC: {e}")

# Comparaison des résultats
st.subheader('Comparaison des modèles')
results_df = pd.DataFrame(list(results.items()), columns=["Modèle", "AUC-ROC"])
st.write(results_df)
