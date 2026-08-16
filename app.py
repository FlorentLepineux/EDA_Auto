import streamlit as st
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Explorateur automatique de données",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITRE DE L'APPLICATION
# ---------------------------------------------------------

st.title("📊 Explorateur automatique de données")

st.write(
    """
    Cette application permet de réaliser automatiquement une
    analyse exploratoire de données (EDA) à partir d'un fichier CSV.
    """
)


# ---------------------------------------------------------
# IMPORT DU FICHIER
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Sélectionnez un fichier CSV",
    type=["csv"]
)


# ---------------------------------------------------------
# TRAITEMENT DU FICHIER
# ---------------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Fichier chargé avec succès !")

    st.subheader("Aperçu des données")

    st.dataframe(df.head())

# ---------------------------------------------------------
# INFORMATIONS GÉNÉRALES
# ---------------------------------------------------------

st.subheader("📋 Informations générales")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        label="Nombre de lignes",
        value=df.shape[0]
    )


with col2:
    st.metric(
        label="Nombre de colonnes",
        value=df.shape[1]
    )


with col3:
    st.metric(
        label="Valeurs manquantes",
        value=df.isna().sum().sum()
    )


with col4:
    st.metric(
        label="Doublons",
        value=df.duplicated().sum()
    )