import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================

st.set_page_config(
    page_title="Explorateur automatique de données",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITRE
# ============================================================

st.title("📊 Explorateur automatique de données")

st.write(
    """
    Cette application permet de réaliser automatiquement
    une analyse exploratoire de données (EDA)
    à partir d'un fichier CSV.
    """
)


# ============================================================
# FONCTION DE CHARGEMENT DU CSV
# ============================================================

def charger_csv(fichier):
    """
    Essaie de charger automatiquement un fichier CSV.

    La fonction teste plusieurs encodages courants.

    Le séparateur est détecté automatiquement grâce à :
        sep=None
        engine="python"

    Cela permet de lire aussi bien :
    - les fichiers séparés par ;
    - les fichiers séparés par ,
    - les fichiers séparés par tabulation
    - etc.
    """

    encodages = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1"
    ]

    for encodage in encodages:

        try:

            # Important :
            # Streamlit nous fournit un fichier déjà ouvert.
            # Après chaque tentative de lecture, on revient
            # donc au début du fichier.
            fichier.seek(0)

            df = pd.read_csv(
                fichier,
                sep=None,
                engine="python",
                encoding=encodage
            )

            return df, encodage

        except UnicodeDecodeError:
            continue

        except Exception:
            continue

    return None, None


# ============================================================
# BARRE LATÉRALE
# ============================================================

st.sidebar.title("📊 EDA Python")

st.sidebar.write(
    """
    Chargez un fichier CSV pour commencer
    automatiquement son exploration.
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Charger un fichier CSV",
    type=["csv"]
)


# ============================================================
# AUCUN FICHIER
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Chargez un fichier CSV depuis la barre latérale pour commencer."
    )

    st.stop()


# ============================================================
# CHARGEMENT DU FICHIER
# ============================================================

df, encodage_utilise = charger_csv(uploaded_file)


if df is None:

    st.error(
        """
        Impossible de lire ce fichier CSV.

        Vérifiez :
        - son encodage ;
        - son séparateur ;
        - sa structure.
        """
    )

    st.stop()


st.success("✅ Fichier chargé avec succès")


# ============================================================
# INFORMATIONS SUR LE FICHIER
# ============================================================

st.caption(
    f"""
    Fichier : {uploaded_file.name}  
    Encodage détecté : {encodage_utilise}
    """
)


# ============================================================
# CRÉATION DES ONGLETS
# ============================================================

tab_general, tab_qualite = st.tabs([
    "📋 Vue générale",
    "🧹 Qualité des données"
])


# ============================================================
# ONGLET 1 : VUE GÉNÉRALE
# ============================================================

with tab_general:

    st.header("📋 Vue générale")


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    nb_lignes = df.shape[0]
    nb_colonnes = df.shape[1]

    nb_numeriques = len(
        df.select_dtypes(include=np.number).columns
    )

    nb_categorielles = (
        nb_colonnes - nb_numeriques
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Nombre de lignes",
            f"{nb_lignes:,}".replace(",", " ")
        )


    with col2:

        st.metric(
            "Nombre de colonnes",
            nb_colonnes
        )


    with col3:

        st.metric(
            "Variables numériques",
            nb_numeriques
        )


    with col4:

        st.metric(
            "Autres variables",
            nb_categorielles
        )


    # --------------------------------------------------------
    # APERÇU DES DONNÉES
    # --------------------------------------------------------

    st.subheader("🔎 Aperçu des données")

    nb_lignes_affichees = st.slider(
        "Nombre de lignes à afficher",
        min_value=5,
        max_value=min(100, len(df)),
        value=min(20, len(df))
    )

    st.dataframe(
        df.head(nb_lignes_affichees),
        use_container_width=True
    )


    # --------------------------------------------------------
    # TYPES DE VARIABLES
    # --------------------------------------------------------

    st.subheader("🧱 Structure du jeu de données")

    infos_colonnes = pd.DataFrame({

        "Colonne": df.columns,

        "Type Python":
            df.dtypes.astype(str).values,

        "Valeurs non nulles":
            df.notna().sum().values,

        "Valeurs uniques":
            df.nunique(dropna=True).values
    })


    st.dataframe(
        infos_colonnes,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # VARIABLES NUMÉRIQUES / CATÉGORIELLES
    # --------------------------------------------------------

    colonnes_numeriques = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )


    colonnes_categorielles = (
        df
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("🔢 Variables numériques")

        if colonnes_numeriques:

            st.write(
                f"{len(colonnes_numeriques)} variable(s)"
            )

            st.write(colonnes_numeriques)

        else:

            st.info(
                "Aucune variable numérique détectée."
            )


    with col2:

        st.subheader("🔤 Autres variables")

        if colonnes_categorielles:

            st.write(
                f"{len(colonnes_categorielles)} variable(s)"
            )

            st.write(colonnes_categorielles)

        else:

            st.info(
                "Aucune variable catégorielle détectée."
            )


# ============================================================
# ONGLET 2 : QUALITÉ DES DONNÉES
# ============================================================

with tab_qualite:

    st.header("🧹 Qualité des données")

    st.write(
        """
        Cette partie recherche automatiquement plusieurs problèmes
        fréquents dans un jeu de données :

        - valeurs manquantes ;
        - doublons ;
        - colonnes constantes ;
        - colonnes avec beaucoup de valeurs différentes ;
        - variables très peu renseignées.
        """
    )


    # ========================================================
    # KPI QUALITÉ
    # ========================================================

    nb_valeurs_total = df.size

    nb_valeurs_manquantes = (
        df.isna().sum().sum()
    )

    taux_manquant_global = (
        nb_valeurs_manquantes
        / nb_valeurs_total
        * 100
        if nb_valeurs_total > 0
        else 0
    )

    nb_doublons = df.duplicated().sum()

    colonnes_constantes = [
        col
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Valeurs manquantes",
            f"{nb_valeurs_manquantes:,}".replace(",", " ")
        )


    with col2:

        st.metric(
            "Taux de valeurs manquantes",
            f"{taux_manquant_global:.2f} %"
        )


    with col3:

        st.metric(
            "Lignes dupliquées",
            nb_doublons
        )


    with col4:

        st.metric(
            "Colonnes constantes",
            len(colonnes_constantes)
        )


    # ========================================================
    # ANALYSE DES VALEURS MANQUANTES
    # ========================================================

    st.subheader("❓ Valeurs manquantes")


    analyse_manquants = pd.DataFrame({

        "Colonne":
            df.columns,

        "Valeurs manquantes":
            df.isna().sum().values,

        "Pourcentage manquant":
            (
                df.isna().mean().values
                * 100
            ),

        "Valeurs uniques":
            df.nunique(dropna=True).values

    })


    analyse_manquants = (
        analyse_manquants
        .sort_values(
            "Pourcentage manquant",
            ascending=False
        )
        .reset_index(drop=True)
    )


    analyse_manquants["Pourcentage manquant"] = (
        analyse_manquants[
            "Pourcentage manquant"
        ].round(2)
    )


    st.dataframe(
        analyse_manquants,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # COLONNES LES PLUS INCOMPLÈTES
    # ========================================================

    colonnes_avec_manquants = (
        analyse_manquants[
            analyse_manquants[
                "Valeurs manquantes"
            ] > 0
        ]
    )


    if not colonnes_avec_manquants.empty:

        st.subheader(
            "📉 Colonnes les plus incomplètes"
        )

        graphique_manquants = (
            colonnes_avec_manquants
            .head(20)
            .set_index("Colonne")[
                "Pourcentage manquant"
            ]
        )

        st.bar_chart(
            graphique_manquants
        )

    else:

        st.success(
            "✅ Aucune valeur manquante détectée."
        )


    # ========================================================
    # COLONNES TRÈS INCOMPLÈTES
    # ========================================================

    st.subheader(
        "⚠️ Variables très peu renseignées"
    )


    seuil_manquant = st.slider(
        "Seuil considéré comme problématique (%)",
        min_value=10,
        max_value=100,
        value=70,
        step=5
    )


    colonnes_problematiques = (
        analyse_manquants[
            analyse_manquants[
                "Pourcentage manquant"
            ] >= seuil_manquant
        ]
    )


    if len(colonnes_problematiques) > 0:

        st.warning(
            f"""
            {len(colonnes_problematiques)} colonne(s)
            possèdent au moins {seuil_manquant} %
            de valeurs manquantes.
            """
        )

        st.dataframe(
            colonnes_problematiques,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            f"""
            Aucune colonne ne dépasse
            {seuil_manquant} % de valeurs manquantes.
            """
        )


    # ========================================================
    # DOUBLONS
    # ========================================================

    st.subheader("👯 Doublons")


    if nb_doublons > 0:

        st.warning(
            f"""
            {nb_doublons} ligne(s) complètement
            identique(s) ont été détectée(s).
            """
        )

        doublons = df[
            df.duplicated(keep=False)
        ]


        st.dataframe(
            doublons.head(100),
            use_container_width=True
        )

    else:

        st.success(
            "✅ Aucun doublon complet détecté."
        )


    # ========================================================
    # COLONNES CONSTANTES
    # ========================================================

    st.subheader("📌 Colonnes constantes")


    if colonnes_constantes:

        st.warning(
            """
            Une colonne constante contient la même valeur
            pour toutes les observations.

            Elle apporte généralement très peu d'information
            pour une analyse statistique ou un modèle
            de Machine Learning.
            """
        )

        st.write(colonnes_constantes)

    else:

        st.success(
            "✅ Aucune colonne constante détectée."
        )


    # ========================================================
    # CARDINALITÉ
    # ========================================================

    st.subheader("🧮 Cardinalité des variables")


    cardinalite = pd.DataFrame({

        "Colonne":
            df.columns,

        "Nombre de valeurs uniques":
            df.nunique(dropna=True).values,

        "Pourcentage de valeurs uniques":
            (
                df.nunique(dropna=True).values
                / len(df)
                * 100
            )
            if len(df) > 0
            else 0
    })


    cardinalite[
        "Pourcentage de valeurs uniques"
    ] = (
        cardinalite[
            "Pourcentage de valeurs uniques"
        ]
        .round(2)
    )


    cardinalite = cardinalite.sort_values(
        "Nombre de valeurs uniques",
        ascending=False
    )


    st.dataframe(
        cardinalite,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # COLONNES POTENTIELLEMENT IDENTIFIANTES
    # ========================================================

    colonnes_identifiantes = cardinalite[
        cardinalite[
            "Pourcentage de valeurs uniques"
        ] >= 95
    ]


    if len(colonnes_identifiantes) > 0:

        st.info(
            f"""
            💡 {len(colonnes_identifiantes)} colonne(s)
            possèdent au moins 95 % de valeurs uniques.

            Elles peuvent correspondre à des identifiants,
            références, numéros de dossier, identifiants
            techniques, etc.
            """
        )

        st.dataframe(
            colonnes_identifiantes,
            use_container_width=True,
            hide_index=True
        )