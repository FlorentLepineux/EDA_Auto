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