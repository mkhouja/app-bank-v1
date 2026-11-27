# app.py · Application Streamlit de la séance 5 (Bank Marketing)
# Lancer en local : streamlit run app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Qui appeler ?", page_icon="📞")


# 1. Charger le Pipeline (préparation + modèle) une seule fois
@st.cache_resource
def charger_modele():
    return joblib.load("model.joblib")


modele = charger_modele()

# 2. Titre et explication
st.title("📞 Ce client va-t-il souscrire ?")
st.write("Remplissez le profil du client, puis cliquez sur **Prédire**. "
         "Le modèle (régression logistique, jeu Bank Marketing) donne la probabilité qu'il souscrive un dépôt à terme.")

# 3. Le formulaire : une saisie par colonne utilisée par le modèle
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Âge", min_value=18, max_value=100, value=40)
    job = st.selectbox("Métier", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                                  "retired", "self-employed", "services", "student", "technician",
                                  "unemployed", "unknown"])
    marital = st.selectbox("Situation familiale", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Études", ["university.degree", "high.school", "professional.course",
                                        "basic.9y", "basic.6y", "basic.4y", "unknown"])
with col2:
    contact = st.selectbox("Type de téléphone", ["cellular", "telephone"])
    campaign = st.slider("Appels pendant cette campagne", min_value=1, max_value=20, value=2)
    previous = st.slider("Contacts lors des campagnes précédentes", min_value=0, max_value=7, value=0)
    poutcome = st.selectbox("Résultat de la campagne précédente", ["nonexistent", "failure", "success"])

# 4. Prédire au clic : un DataFrame d'une ligne, avec les mêmes noms de colonnes qu'à l'entraînement
if st.button("Prédire"):
    client = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "contact": contact, "campaign": campaign, "previous": previous, "poutcome": poutcome,
    }])
    proba = modele.predict_proba(client)[0, 1]
    st.metric("Probabilité de souscrire", f"{proba:.0%}")
    if proba >= 0.5:
        st.success("Client prometteur : à appeler en priorité.")
    else:
        st.info("Client peu probable : pas prioritaire.")

# 5. Les limites, toujours visibles
st.caption("Limites : modèle appris sur 3 750 clients d'une banque portugaise (2008-2010) ; "
           "il retrouve environ 1 client intéressé sur 5 (rappel 0,20). Outil d'aide, pas de décision automatique.")
