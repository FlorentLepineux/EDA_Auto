import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest


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
# FONCTIONS POUR L'ANALYSE AUTOMATIQUE MULTITYPE
# ============================================================

def cramers_v(serie_x, serie_y):
    """
    Mesure l'intensité de l'association entre deux variables
    catégorielles avec le V de Cramér.

    Le résultat varie entre 0 et 1 :
    - proche de 0 : association faible ;
    - proche de 1 : association forte.
    """

    donnees = pd.DataFrame({
        "x": serie_x,
        "y": serie_y
    }).dropna()

    if len(donnees) < 2:
        return np.nan

    tableau = pd.crosstab(
        donnees["x"],
        donnees["y"]
    )

    if tableau.shape[0] < 2 or tableau.shape[1] < 2:
        return np.nan

    try:
        chi2 = chi2_contingency(
            tableau,
            correction=False
        )[0]
    except ValueError:
        return np.nan

    n = tableau.to_numpy().sum()

    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    r, k = tableau.shape

    # Correction du biais, utile notamment sur les petits échantillons.
    phi2_corrige = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    r_corrige = r - ((r - 1) ** 2) / (n - 1)
    k_corrige = k - ((k - 1) ** 2) / (n - 1)

    denominateur = min(
        k_corrige - 1,
        r_corrige - 1
    )

    if denominateur <= 0:
        return np.nan

    return np.sqrt(
        phi2_corrige / denominateur
    )


def eta_squared(serie_cat, serie_num):
    """
    Calcule le rapport de corrélation η² entre une variable
    catégorielle et une variable numérique.

    Le résultat varie entre 0 et 1 :
    - proche de 0 : les groupes expliquent peu la variation ;
    - proche de 1 : les groupes expliquent une grande partie
      de la variation de la variable numérique.
    """

    donnees = pd.DataFrame({
        "categorie": serie_cat,
        "valeur": pd.to_numeric(
            serie_num,
            errors="coerce"
        )
    }).dropna()

    if len(donnees) < 2:
        return np.nan

    if donnees["categorie"].nunique() < 2:
        return np.nan

    moyenne_generale = donnees["valeur"].mean()

    somme_carres_totale = (
        (donnees["valeur"] - moyenne_generale) ** 2
    ).sum()

    if somme_carres_totale == 0:
        return np.nan

    somme_carres_intergroupes = 0

    for _, groupe in donnees.groupby(
        "categorie",
        observed=True
    ):

        somme_carres_intergroupes += (
            len(groupe)
            * (groupe["valeur"].mean() - moyenne_generale) ** 2
        )

    return (
        somme_carres_intergroupes
        / somme_carres_totale
    )


def niveau_association(valeur):
    """
    Fournit une interprétation simple d'un coefficient
    compris entre 0 et 1.
    """

    if pd.isna(valeur):
        return "Non calculable"

    valeur = abs(valeur)

    if valeur < 0.10:
        return "Très faible"
    elif valeur < 0.30:
        return "Faible"
    elif valeur < 0.50:
        return "Modérée"
    elif valeur < 0.70:
        return "Forte"
    else:
        return "Très forte"


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
# DATAFRAME UTILISÉ POUR LES ANALYSES
# ============================================================
#
# "df" conserve les données brutes telles qu'elles ont été importées.
#
# "df_analyse" est une copie utilisée par les différents onglets.
# L'utilisateur peut modifier les types de colonnes dans l'onglet
# "Types des colonnes" avant les analyses.
#
# Exemple :
# un identifiant numérique (123456) peut être transformé en texte
# afin qu'il ne soit pas utilisé comme une vraie variable quantitative.

df_analyse = df.copy()



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

tab_general, tab_types, tab_qualite, tab_auto, tab_univariee, tab_bivariee, tab_correlations, tab_avancee, tab_nettoyage = st.tabs([
    "📋 Vue générale",
    "🧱 Types des colonnes",
    "🧹 Qualité des données",
    "✨ Analyse automatique",
    "📊 Analyse univariée",
    "🔗 Analyse bivariée",
    "🔥 Corrélations",
    "🧠 Analyse avancée",
    "🛠️ Nettoyage & Export"
])


# ============================================================
# ONGLET : TYPES DES COLONNES
# ============================================================

with tab_types:

    st.header("🧱 Types des colonnes")

    st.write(
        """
        Python détecte automatiquement le type des colonnes au moment
        de l'import du fichier CSV.

        Cette détection peut cependant être correcte techniquement
        mais incorrecte du point de vue métier.

        **Exemple :**

        un identifiant comme `123456` peut être interprété comme un
        nombre alors qu'il ne représente pas une quantité mesurable.

        Dans ce cas, il est préférable de le convertir en
        **variable catégorielle / texte** afin qu'il ne soit pas utilisé
        dans les moyennes, corrélations, PCA ou algorithmes de clustering.
        """
    )

    st.info(
        """
        💡 **Conseil**

        Utilisez **Catégorielle / texte** pour les identifiants,
        codes, numéros de dossier, codes postaux ou autres valeurs
        numériques qui servent uniquement à identifier une observation.
        """
    )

    # --------------------------------------------------------
    # TABLEAU DE CONFIGURATION DES TYPES
    # --------------------------------------------------------

    configuration_types = pd.DataFrame({
        "Colonne": df.columns,
        "Type détecté": df.dtypes.astype(str).values,
        "Type utilisé": ["Automatique"] * len(df.columns)
    })

    configuration_types = st.data_editor(
        configuration_types,
        use_container_width=True,
        hide_index=True,
        disabled=[
            "Colonne",
            "Type détecté"
        ],
        column_config={

            "Colonne": st.column_config.TextColumn(
                "Colonne"
            ),

            "Type détecté": st.column_config.TextColumn(
                "Type détecté"
            ),

            "Type utilisé": st.column_config.SelectboxColumn(
                "Type utilisé",
                options=[
                    "Automatique",
                    "Numérique",
                    "Catégorielle / texte",
                    "Booléenne",
                    "Date"
                ],
                required=True
            )
        },
        key="table_types_colonnes"
    )

    # --------------------------------------------------------
    # RÉCUPÉRATION DES TYPES CHOISIS
    # --------------------------------------------------------

    types_choisis = dict(
        zip(
            configuration_types["Colonne"],
            configuration_types["Type utilisé"]
        )
    )

    # --------------------------------------------------------
    # APPLICATION DES CONVERSIONS
    # --------------------------------------------------------

    df_analyse = df.copy()

    erreurs_conversion = []
    conversions_realisees = []

    for colonne, type_choisi in types_choisis.items():

        try:

            # ------------------------------------------------
            # NUMÉRIQUE
            # ------------------------------------------------

            if type_choisi == "Numérique":

                nb_na_avant = df_analyse[colonne].isna().sum()

                df_analyse[colonne] = pd.to_numeric(
                    df_analyse[colonne],
                    errors="coerce"
                )

                nb_na_apres = df_analyse[colonne].isna().sum()

                conversions_realisees.append(
                    f"{colonne} → Numérique"
                )

                if nb_na_apres > nb_na_avant:

                    erreurs_conversion.append(
                        f"{colonne} : "
                        f"{nb_na_apres - nb_na_avant} valeur(s) "
                        "n'ont pas pu être converties en nombre "
                        "et ont été transformées en valeurs manquantes."
                    )

            # ------------------------------------------------
            # CATÉGORIELLE / TEXTE
            # ------------------------------------------------

            elif type_choisi == "Catégorielle / texte":

                df_analyse[colonne] = (
                    df_analyse[colonne]
                    .astype("string")
                )

                conversions_realisees.append(
                    f"{colonne} → Catégorielle / texte"
                )

            # ------------------------------------------------
            # BOOLÉENNE
            # ------------------------------------------------

            elif type_choisi == "Booléenne":

                mapping_bool = {
                    "true": True,
                    "false": False,
                    "vrai": True,
                    "faux": False,
                    "oui": True,
                    "non": False,
                    "1": True,
                    "0": False
                }

                serie_temp = (
                    df_analyse[colonne]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                )

                valeurs_non_vides = serie_temp.dropna()

                valeurs_inconnues = (
                    valeurs_non_vides[
                        ~valeurs_non_vides.isin(mapping_bool.keys())
                    ]
                    .unique()
                    .tolist()
                )

                if valeurs_inconnues:

                    erreurs_conversion.append(
                        f"{colonne} : certaines valeurs ne correspondent "
                        f"pas à un booléen reconnu ({valeurs_inconnues[:5]})."
                    )

                df_analyse[colonne] = (
                    serie_temp
                    .map(mapping_bool)
                    .astype("boolean")
                )

                conversions_realisees.append(
                    f"{colonne} → Booléenne"
                )

            # ------------------------------------------------
            # DATE
            # ------------------------------------------------

            elif type_choisi == "Date":

                nb_na_avant = df_analyse[colonne].isna().sum()

                df_analyse[colonne] = pd.to_datetime(
                    df_analyse[colonne],
                    errors="coerce"
                )

                nb_na_apres = df_analyse[colonne].isna().sum()

                conversions_realisees.append(
                    f"{colonne} → Date"
                )

                if nb_na_apres > nb_na_avant:

                    erreurs_conversion.append(
                        f"{colonne} : "
                        f"{nb_na_apres - nb_na_avant} valeur(s) "
                        "n'ont pas pu être converties en date "
                        "et ont été transformées en valeurs manquantes."
                    )

            # "Automatique" :
            # aucun changement, le type détecté par Pandas est conservé.

        except Exception as e:

            erreurs_conversion.append(
                f"{colonne} : {e}"
            )

    # --------------------------------------------------------
    # RÉSUMÉ DES CONVERSIONS
    # --------------------------------------------------------

    if conversions_realisees:

        st.success(
            f"✅ {len(conversions_realisees)} conversion(s) manuelle(s) appliquée(s)."
        )

    else:

        st.info(
            """
            Aucune conversion manuelle sélectionnée.
            Les types détectés automatiquement sont conservés.
            """
        )

    if erreurs_conversion:

        st.warning(
            "Certaines conversions nécessitent votre attention :"
        )

        for erreur in erreurs_conversion:
            st.write(f"- {erreur}")

    # --------------------------------------------------------
    # TABLEAU DES TYPES FINAUX
    # --------------------------------------------------------

    st.subheader("📋 Types utilisés pour les analyses")

    types_resultats = pd.DataFrame({
        "Colonne": df_analyse.columns,
        "Type initial": df.dtypes.astype(str).values,
        "Type utilisé": df_analyse.dtypes.astype(str).values,
        "Valeurs manquantes": df_analyse.isna().sum().values,
        "Valeurs uniques": df_analyse.nunique(dropna=True).values
    })

    st.dataframe(
        types_resultats,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        """
        Les onglets d'analyse utilisent automatiquement ces types.
        """
    )


# ============================================================
# ONGLET 1 : VUE GÉNÉRALE
# ============================================================

with tab_general:

    st.header("📋 Vue générale")


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    nb_lignes = df_analyse.shape[0]
    nb_colonnes = df_analyse.shape[1]

    nb_numeriques = len(
        df_analyse.select_dtypes(include=np.number).columns
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
        max_value=min(100, len(df_analyse)),
        value=min(20, len(df_analyse))
    )

    st.dataframe(
        df_analyse.head(nb_lignes_affichees),
        use_container_width=True
    )


    # --------------------------------------------------------
    # TYPES DE VARIABLES
    # --------------------------------------------------------

    st.subheader("🧱 Structure du jeu de données")

    infos_colonnes = pd.DataFrame({

        "Colonne": df_analyse.columns,

        "Type Python":
            df_analyse.dtypes.astype(str).values,

        "Valeurs non nulles":
            df_analyse.notna().sum().values,

        "Valeurs uniques":
            df_analyse.nunique(dropna=True).values
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
        df_analyse
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )


    colonnes_categorielles = (
        df_analyse
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

    nb_valeurs_total = df_analyse.size

    nb_valeurs_manquantes = (
        df_analyse.isna().sum().sum()
    )

    taux_manquant_global = (
        nb_valeurs_manquantes
        / nb_valeurs_total
        * 100
        if nb_valeurs_total > 0
        else 0
    )

    nb_doublons = df_analyse.duplicated().sum()

    colonnes_constantes = [
        col
        for col in df_analyse.columns
        if df_analyse[col].nunique(dropna=False) <= 1
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
            df_analyse.columns,

        "Valeurs manquantes":
            df_analyse.isna().sum().values,

        "Pourcentage manquant":
            (
                df_analyse.isna().mean().values
                * 100
            ),

        "Valeurs uniques":
            df_analyse.nunique(dropna=True).values

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

        doublons = df_analyse[
            df_analyse.duplicated(keep=False)
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
            df_analyse.columns,

        "Nombre de valeurs uniques":
            df_analyse.nunique(dropna=True).values,

        "Pourcentage de valeurs uniques":
            (
                df_analyse.nunique(dropna=True).values
                / len(df_analyse)
                * 100
            )
            if len(df_analyse) > 0
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

# ============================================================
# ONGLET : ANALYSE AUTOMATIQUE
# ============================================================

with tab_auto:

    st.header("✨ Première analyse automatique")

    st.write(
        """
        Cette section réalise une première lecture automatique
        de l'ensemble du jeu de données.

        Elle adapte les méthodes utilisées au type des variables :
        **numériques, catégorielles, booléennes et temporelles**.

        L'objectif est de faire ressortir rapidement les éléments
        qui semblent mériter une analyse plus approfondie dans
        les autres onglets.
        """
    )

    st.info(
        """
        💡 Les résultats ci-dessous sont des **pistes d'exploration**.
        Une association statistique n'implique pas nécessairement
        une relation de causalité et une valeur atypique n'est pas
        forcément une erreur.
        """
    )


    # ========================================================
    # PARAMÈTRES DE DÉTECTION
    # ========================================================

    with st.expander("⚙️ Paramètres de détection"):

        seuil_manquants_auto = st.slider(
            "Seuil de valeurs manquantes considéré comme problématique (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="seuil_manquants_auto"
        )

        seuil_corr_num_auto = st.slider(
            "Seuil minimum pour les corrélations numériques",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key="seuil_corr_num_auto"
        )

        seuil_cramer_auto = st.slider(
            "Seuil minimum du V de Cramér",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="seuil_cramer_auto"
        )

        seuil_eta_auto = st.slider(
            "Seuil minimum de η²",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            key="seuil_eta_auto"
        )

        st.caption(
            """
            Ces seuils servent uniquement à sélectionner les éléments
            affichés dans l'analyse automatique. Ils ne modifient pas
            les données.
            """
        )

    # ========================================================
    # IDENTIFICATION DES TYPES DE VARIABLES
    # ========================================================

    colonnes_num_auto = [
        col
        for col in df_analyse.columns
        if (
            pd.api.types.is_numeric_dtype(df_analyse[col])
            and not pd.api.types.is_bool_dtype(df_analyse[col])
        )
    ]

    colonnes_date_auto = [
        col
        for col in df_analyse.columns
        if pd.api.types.is_datetime64_any_dtype(df_analyse[col])
    ]

    colonnes_cat_auto = [
        col
        for col in df_analyse.columns
        if (
            col not in colonnes_num_auto
            and col not in colonnes_date_auto
        )
    ]

    # Les variables à très forte cardinalité sont utiles à signaler,
    # mais sont volontairement exclues de certaines comparaisons
    # catégorielles afin d'éviter des tableaux gigantesques.
    colonnes_cat_relations = [
        col
        for col in colonnes_cat_auto
        if (
            2 <= df_analyse[col].nunique(dropna=True) <= 30
        )
    ]

    # ========================================================
    # 1. DIAGNOSTIC GÉNÉRAL
    # ========================================================

    st.subheader("1️⃣ Diagnostic général")

    nb_lignes_auto = len(df_analyse)
    nb_colonnes_auto = df_analyse.shape[1]
    nb_manquants_auto = int(
        df_analyse.isna().sum().sum()
    )
    nb_doublons_auto = int(
        df_analyse.duplicated().sum()
    )

    taux_manquants_auto = (
        nb_manquants_auto / df_analyse.size * 100
        if df_analyse.size > 0
        else 0
    )

    colonnes_constantes_auto = [
        col
        for col in df_analyse.columns
        if df_analyse[col].nunique(dropna=False) <= 1
    ]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Observations",
            f"{nb_lignes_auto:,}".replace(",", " ")
        )

    with k2:
        st.metric(
            "Variables",
            nb_colonnes_auto
        )

    with k3:
        st.metric(
            "Valeurs manquantes",
            f"{taux_manquants_auto:.2f} %"
        )

    with k4:
        st.metric(
            "Doublons",
            nb_doublons_auto
        )

    st.write(
        f"""
        **Types actuellement utilisés :**
        {len(colonnes_num_auto)} numérique(s) ·
        {len(colonnes_cat_auto)} catégorielle(s) / booléenne(s) ·
        {len(colonnes_date_auto)} date(s)
        """
    )

    # ========================================================
    # 2. POINTS D'ATTENTION
    # ========================================================

    st.subheader("2️⃣ Points d'attention")

    points_attention = []

    taux_na_auto = (
        df_analyse
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    colonnes_incompletes_auto = taux_na_auto[
        taux_na_auto >= seuil_manquants_auto
    ]

    if len(colonnes_incompletes_auto) > 0:
        points_attention.append(
            f"**{len(colonnes_incompletes_auto)} colonne(s)** "
            "possèdent au moins {seuil_manquants_auto:.0f} % de valeurs manquantes."
        )

    if nb_doublons_auto > 0:
        points_attention.append(
            f"**{nb_doublons_auto} ligne(s)** complètement dupliquée(s) "
            "ont été détectées."
        )

    if colonnes_constantes_auto:
        points_attention.append(
            f"**{len(colonnes_constantes_auto)} colonne(s) constante(s)** "
            "n'apportent pratiquement aucune information discriminante."
        )

    variables_identifiantes_auto = []

    for col in df_analyse.columns:

        serie = df_analyse[col]

        nb_non_nuls = serie.notna().sum()

        if nb_non_nuls == 0:
            continue

        taux_unique = (
            serie.nunique(dropna=True)
            / nb_non_nuls
            * 100
        )

        if taux_unique >= 95:

            variables_identifiantes_auto.append({
                "Colonne": col,
                "Valeurs uniques (%)": round(
                    taux_unique,
                    2
                ),
                "Type": str(serie.dtype)
            })

    if variables_identifiantes_auto:
        points_attention.append(
            f"**{len(variables_identifiantes_auto)} variable(s)** "
            "possèdent au moins 95 % de valeurs uniques et peuvent "
            "correspondre à des identifiants ou champs libres."
        )

    if points_attention:

        for point in points_attention:
            st.warning(point)

    else:
        st.success(
            "Aucun problème majeur n'a été détecté lors de ce premier diagnostic."
        )

    with st.expander(
        "Voir le détail des colonnes à surveiller"
    ):

        if len(colonnes_incompletes_auto) > 0:

            st.write("#### Colonnes très incomplètes")

            tableau_incomplet = (
                colonnes_incompletes_auto
                .rename("Valeurs manquantes (%)")
                .reset_index()
                .rename(columns={"index": "Colonne"})
            )

            st.dataframe(
                tableau_incomplet,
                use_container_width=True,
                hide_index=True
            )

        if colonnes_constantes_auto:

            st.write("#### Colonnes constantes")

            st.write(
                colonnes_constantes_auto
            )

        if variables_identifiantes_auto:

            st.write(
                "#### Variables potentiellement identifiantes"
            )

            st.dataframe(
                pd.DataFrame(
                    variables_identifiantes_auto
                ),
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # 3. VARIABLES NUMÉRIQUES REMARQUABLES
    # ========================================================

    st.subheader("3️⃣ Variables numériques remarquables")

    analyses_num_auto = []

    for col in colonnes_num_auto:

        serie_num = pd.to_numeric(
            df_analyse[col],
            errors="coerce"
        ).dropna()

        if len(serie_num) < 5:
            continue

        if serie_num.nunique() <= 1:
            continue

        asymetrie = serie_num.skew()

        q1 = serie_num.quantile(0.25)
        q3 = serie_num.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            taux_outliers = 0.0
        else:
            borne_basse = q1 - 1.5 * iqr
            borne_haute = q3 + 1.5 * iqr

            taux_outliers = (
                (
                    (serie_num < borne_basse)
                    | (serie_num > borne_haute)
                ).mean()
                * 100
            )

        if abs(asymetrie) >= 1 or taux_outliers >= 5:

            if abs(asymetrie) >= 1 and taux_outliers >= 5:
                remarque = "Asymétrie forte et valeurs atypiques"
            elif abs(asymetrie) >= 1:
                remarque = "Distribution fortement asymétrique"
            else:
                remarque = "Nombre notable de valeurs atypiques"

            analyses_num_auto.append({
                "Variable": col,
                "Asymétrie": round(asymetrie, 2),
                "Outliers IQR (%)": round(taux_outliers, 2),
                "Observation": remarque
            })

    if analyses_num_auto:

        df_num_remarquables = (
            pd.DataFrame(analyses_num_auto)
            .sort_values(
                "Outliers IQR (%)",
                ascending=False
            )
        )

        st.dataframe(
            df_num_remarquables,
            use_container_width=True,
            hide_index=True
        )

    elif colonnes_num_auto:

        st.success(
            "Aucune variable numérique particulièrement atypique "
            "n'a été détectée avec les seuils utilisés."
        )

    else:

        st.info(
            "Aucune variable numérique n'est disponible."
        )

    # ========================================================
    # 4. VARIABLES CATÉGORIELLES REMARQUABLES
    # ========================================================

    st.subheader("4️⃣ Variables catégorielles remarquables")

    analyses_cat_auto = []

    for col in colonnes_cat_auto:

        serie_cat = df_analyse[col].dropna()

        if len(serie_cat) == 0:
            continue

        frequences = serie_cat.value_counts(
            normalize=True
        )

        nb_modalites = serie_cat.nunique()

        modalite_principale = frequences.index[0]
        part_principale = frequences.iloc[0] * 100

        remarques = []

        if nb_modalites == 1:
            remarques.append("Variable constante")

        elif part_principale >= 80:
            remarques.append(
                "Variable fortement déséquilibrée"
            )

        taux_unique_cat = (
            nb_modalites
            / len(serie_cat)
            * 100
        )

        if taux_unique_cat >= 95:
            remarques.append(
                "Très forte cardinalité / identifiant potentiel"
            )

        elif nb_modalites > 30:
            remarques.append(
                "Forte cardinalité"
            )

        if remarques:

            analyses_cat_auto.append({
                "Variable": col,
                "Modalités": nb_modalites,
                "Modalité principale": str(
                    modalite_principale
                )[:80],
                "Part principale (%)": round(
                    part_principale,
                    2
                ),
                "Observation": " ; ".join(remarques)
            })

    if analyses_cat_auto:

        st.dataframe(
            pd.DataFrame(analyses_cat_auto),
            use_container_width=True,
            hide_index=True
        )

    elif colonnes_cat_auto:

        st.success(
            "Aucune variable catégorielle particulièrement remarquable "
            "n'a été détectée avec les seuils utilisés."
        )

    else:

        st.info(
            "Aucune variable catégorielle ou booléenne n'est disponible."
        )

    # ========================================================
    # 5. NUMÉRIQUE × NUMÉRIQUE
    # ========================================================

    st.subheader("5️⃣ Relations Numérique × Numérique")

    relations_num_num = []

    if len(colonnes_num_auto) >= 2:

        matrice_pearson_auto = (
            df_analyse[colonnes_num_auto]
            .corr(method="pearson")
        )

        matrice_spearman_auto = (
            df_analyse[colonnes_num_auto]
            .corr(method="spearman")
        )

        for i, var1 in enumerate(colonnes_num_auto):

            for var2 in colonnes_num_auto[i + 1:]:

                pearson = matrice_pearson_auto.loc[
                    var1,
                    var2
                ]

                spearman = matrice_spearman_auto.loc[
                    var1,
                    var2
                ]

                force_max = np.nanmax(
                    np.abs([pearson, spearman])
                )

                if (
                    not pd.isna(force_max)
                    and force_max >= seuil_corr_num_auto
                ):

                    relations_num_num.append({
                        "Variable 1": var1,
                        "Variable 2": var2,
                        "Pearson": round(pearson, 3),
                        "Spearman": round(spearman, 3),
                        "Force maximale": round(
                            force_max,
                            3
                        )
                    })

        if relations_num_num:

            df_rel_num = (
                pd.DataFrame(relations_num_num)
                .sort_values(
                    "Force maximale",
                    ascending=False
                )
            )

            st.dataframe(
                df_rel_num,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"""
                Sont affichées ici les relations dont la valeur absolue
                de Pearson ou Spearman atteint au moins
                **{seuil_corr_num_auto:.2f}**.
                """
            )

        else:

            st.success(
                f"Aucune relation numérique forte détectée "
                f"avec le seuil de {seuil_corr_num_auto:.2f}."
            )

    else:

        st.info(
            "Au moins deux variables numériques sont nécessaires."
        )

    # ========================================================
    # 6. CATÉGORIELLE × CATÉGORIELLE
    # ========================================================

    st.subheader("6️⃣ Relations Catégorielle × Catégorielle")

    st.write(
        """
        Le **V de Cramér** mesure l'intensité de l'association
        entre deux variables catégorielles.

        Pour préserver les performances et éviter des tableaux croisés
        gigantesques, l'analyse automatique utilise ici uniquement
        les variables comportant entre **2 et 30 modalités**.
        """
    )

    relations_cat_cat = []

    # Garde-fou supplémentaire pour Streamlit Cloud :
    # si le fichier contient beaucoup de variables catégorielles,
    # on limite automatiquement le nombre de variables testées.
    cat_testees = colonnes_cat_relations[:25]

    for i, var1 in enumerate(cat_testees):

        for var2 in cat_testees[i + 1:]:

            valeur_v = cramers_v(
                df_analyse[var1],
                df_analyse[var2]
            )

            if (
                not pd.isna(valeur_v)
                and valeur_v >= seuil_cramer_auto
            ):

                relations_cat_cat.append({
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "V de Cramér": round(
                        valeur_v,
                        3
                    ),
                    "Intensité": niveau_association(
                        valeur_v
                    )
                })

    if relations_cat_cat:

        df_rel_cat = (
            pd.DataFrame(relations_cat_cat)
            .sort_values(
                "V de Cramér",
                ascending=False
            )
        )

        st.dataframe(
            df_rel_cat,
            use_container_width=True,
            hide_index=True
        )

    elif len(cat_testees) >= 2:

        st.success(
            f"Aucune association catégorielle forte détectée "
            f"avec le seuil de {seuil_cramer_auto:.2f}."
        )

    else:

        st.info(
            "Pas assez de variables catégorielles adaptées "
            "pour calculer le V de Cramér."
        )

    if len(colonnes_cat_relations) > 25:

        st.caption(
            """
            ⚙️ Pour préserver les performances, seules les
            25 premières variables catégorielles adaptées ont
            été comparées automatiquement.
            """
        )

    # ========================================================
    # 7. CATÉGORIELLE × NUMÉRIQUE
    # ========================================================

    st.subheader("7️⃣ Relations Catégorielle × Numérique")

    st.write(
        """
        Le rapport de corrélation **η² (eta carré)** indique quelle
        part de la variation d'une variable numérique peut être
        associée aux groupes d'une variable catégorielle.

        Une valeur élevée indique que les groupes présentent des
        niveaux numériques sensiblement différents.
        """
    )

    relations_mixtes = []

    # Même principe : limitation du nombre de variables testées
    # afin de conserver une application fluide sur Streamlit Cloud.
    cat_mixtes = colonnes_cat_relations[:20]
    num_mixtes = colonnes_num_auto[:30]

    for variable_cat in cat_mixtes:

        for variable_num in num_mixtes:

            eta2 = eta_squared(
                df_analyse[variable_cat],
                df_analyse[variable_num]
            )

            if (
                not pd.isna(eta2)
                and eta2 >= seuil_eta_auto
            ):

                relations_mixtes.append({
                    "Variable catégorielle": variable_cat,
                    "Variable numérique": variable_num,
                    "η²": round(eta2, 3),
                    "Intensité": niveau_association(
                        eta2
                    )
                })

    if relations_mixtes:

        df_rel_mixtes = (
            pd.DataFrame(relations_mixtes)
            .sort_values(
                "η²",
                ascending=False
            )
        )

        st.dataframe(
            df_rel_mixtes,
            use_container_width=True,
            hide_index=True
        )

    elif cat_mixtes and num_mixtes:

        st.success(
            f"Aucune relation mixte particulièrement forte détectée "
            f"avec le seuil η² ≥ {seuil_eta_auto:.2f}."
        )

    else:

        st.info(
            "Cette analyse nécessite au moins une variable numérique "
            "et une variable catégorielle adaptée."
        )

    if (
        len(colonnes_cat_relations) > 20
        or len(colonnes_num_auto) > 30
    ):

        st.caption(
            """
            ⚙️ Pour préserver les performances, l'analyse automatique
            limite le nombre de combinaisons testées. L'onglet
            « Analyse bivariée » permet ensuite d'étudier précisément
            n'importe quelle paire de variables.
            """
        )

    # ========================================================
    # 8. ANALYSE TEMPORELLE
    # ========================================================

    st.subheader("8️⃣ Variables temporelles")

    analyses_dates = []

    for col in colonnes_date_auto:

        serie_date = (
            df_analyse[col]
            .dropna()
            .sort_values()
        )

        if len(serie_date) == 0:
            continue

        date_min = serie_date.min()
        date_max = serie_date.max()

        duree_jours = (
            date_max - date_min
        ).days

        analyses_dates.append({
            "Variable": col,
            "Première date": date_min.date(),
            "Dernière date": date_max.date(),
            "Période couverte (jours)": duree_jours,
            "Valeurs renseignées": len(serie_date),
            "Dates uniques": serie_date.nunique()
        })

    if analyses_dates:

        st.dataframe(
            pd.DataFrame(analyses_dates),
            use_container_width=True,
            hide_index=True
        )

        variable_date_graphique = st.selectbox(
            "Variable temporelle à visualiser",
            options=colonnes_date_auto,
            key="date_auto_graphique"
        )

        serie_temporelle = (
            df_analyse[
                [variable_date_graphique]
            ]
            .dropna()
            .copy()
        )

        if not serie_temporelle.empty:

            serie_temporelle["Période"] = (
                serie_temporelle[
                    variable_date_graphique
                ]
                .dt.to_period("M")
                .astype(str)
            )

            evolution_dates = (
                serie_temporelle
                .groupby("Période")
                .size()
                .reset_index(name="Nombre d'observations")
            )

            fig_dates_auto = px.line(
                evolution_dates,
                x="Période",
                y="Nombre d'observations",
                markers=True,
                title=(
                    f"Répartition temporelle de "
                    f"{variable_date_graphique}"
                )
            )

            st.plotly_chart(
                fig_dates_auto,
                use_container_width=True
            )

    else:

        st.info(
            """
            Aucune variable n'est actuellement reconnue comme une date.

            Vous pouvez utiliser l'onglet **Types des colonnes**
            pour convertir une colonne en type **Date**.
            """
        )

    # ========================================================
    # 9. SYNTHÈSE AUTOMATIQUE
    # ========================================================

    st.subheader("9️⃣ Synthèse : que regarder en priorité ?")

    synthese = []

    if len(colonnes_incompletes_auto) > 0:
        synthese.append(
            f"🧹 Examiner les **{len(colonnes_incompletes_auto)} "
            "colonne(s) très incomplète(s)** avant toute analyse approfondie."
        )

    if variables_identifiantes_auto:
        synthese.append(
            f"🆔 Vérifier le rôle métier de **{len(variables_identifiantes_auto)} "
            "variable(s) à très forte cardinalité**, qui peuvent être des identifiants."
        )

    if analyses_num_auto:
        synthese.append(
            f"📊 Approfondir **{len(analyses_num_auto)} variable(s) numérique(s)** "
            "présentant une asymétrie importante ou des valeurs atypiques."
        )

    if analyses_cat_auto:
        synthese.append(
            f"🔤 Examiner **{len(analyses_cat_auto)} variable(s) catégorielle(s)** "
            "présentant un déséquilibre ou une cardinalité particulière."
        )

    if relations_num_num:
        meilleure = max(
            relations_num_num,
            key=lambda x: x["Force maximale"]
        )

        synthese.append(
            f"🔗 La relation numérique la plus marquée détectée concerne "
            f"**{meilleure['Variable 1']}** et **{meilleure['Variable 2']}** "
            f"(force ≈ **{meilleure['Force maximale']:.3f}**)."
        )

    if relations_cat_cat:
        meilleure = max(
            relations_cat_cat,
            key=lambda x: x["V de Cramér"]
        )

        synthese.append(
            f"🔗 L'association catégorielle la plus marquée concerne "
            f"**{meilleure['Variable 1']}** et **{meilleure['Variable 2']}** "
            f"(V de Cramér = **{meilleure['V de Cramér']:.3f}**)."
        )

    if relations_mixtes:
        meilleure = max(
            relations_mixtes,
            key=lambda x: x["η²"]
        )

        synthese.append(
            f"🔗 La relation mixte la plus marquée relie "
            f"**{meilleure['Variable catégorielle']}** à "
            f"**{meilleure['Variable numérique']}** "
            f"(η² = **{meilleure['η²']:.3f}**)."
        )

    if colonnes_date_auto:
        synthese.append(
            f"📅 **{len(colonnes_date_auto)} variable(s) temporelle(s)** "
            "peuvent être utilisées pour rechercher des évolutions dans le temps."
        )

    if synthese:

        for element in synthese:
            st.write(element)

    else:

        st.success(
            """
            Le diagnostic automatique ne fait ressortir aucun élément
            particulièrement marqué avec les seuils actuels.

            Vous pouvez poursuivre avec les analyses univariées et
            bivariées pour explorer le jeu de données plus finement.
            """
        )

    st.caption(
        """
        Les seuils sélectionnés servent à faire ressortir les relations
        les plus visibles. Ils ne constituent pas des règles statistiques
        universelles et doivent toujours être interprétés dans le contexte
        métier des données.
        """
    )


# ============================================================
# ONGLET 3 : ANALYSE UNIVARIÉE
# ============================================================

with tab_univariee:

    st.header("📊 Analyse univariée")

    st.write(
        """
        L'analyse univariée consiste à étudier les variables
        **une par une**.

        Elle permet notamment de comprendre :

        - la distribution d'une variable ;
        - ses valeurs les plus fréquentes ;
        - sa dispersion ;
        - la présence éventuelle de valeurs extrêmes ;
        - la quantité de données manquantes.
        """
    )

    # --------------------------------------------------------
    # SÉLECTION DE LA VARIABLE
    # --------------------------------------------------------

    variable = st.selectbox(
        "Sélectionnez une variable à analyser",
        options=df_analyse.columns
    )

    serie = df_analyse[variable]

    # --------------------------------------------------------
    # DÉTECTION DU TYPE
    # --------------------------------------------------------

    est_numerique = (
    pd.api.types.is_numeric_dtype(serie)
    and not pd.api.types.is_bool_dtype(serie)
)

    if est_numerique:

        st.success(
            "🔢 Variable numérique détectée"
        )

    else:

        st.success(
            "🔤 Variable catégorielle / textuelle détectée"
        )

    # --------------------------------------------------------
    # INFORMATIONS GÉNÉRALES SUR LA VARIABLE
    # --------------------------------------------------------

    nb_valeurs = len(serie)
    nb_manquantes = serie.isna().sum()
    nb_uniques = serie.nunique(dropna=True)

    taux_manquant = (
        nb_manquantes / nb_valeurs * 100
        if nb_valeurs > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Observations", nb_valeurs)

    with col2:
        st.metric("Valeurs uniques", nb_uniques)

    with col3:
        st.metric("Valeurs manquantes", nb_manquantes)

    with col4:
        st.metric("Taux manquant", f"{taux_manquant:.2f} %")

    # ========================================================
    # ANALYSE NUMÉRIQUE
    # ========================================================

    if est_numerique:

        st.subheader(f"Analyse numérique : {variable}")

        serie_num = pd.to_numeric(
            serie.dropna(),
            errors="coerce"
        ).dropna()

        if len(serie_num) == 0:
            st.warning("Cette variable ne contient aucune valeur exploitable.")

        else:
            # Statistiques descriptives
            moyenne = serie_num.mean()
            mediane = serie_num.median()
            minimum = serie_num.min()
            maximum = serie_num.max()
            ecart_type = serie_num.std()
            q1 = serie_num.quantile(0.25)
            q3 = serie_num.quantile(0.75)

            st.subheader("📐 Statistiques descriptives")

            statistiques = pd.DataFrame({
                "Statistique": [
                    "Moyenne", "Médiane", "Écart-type", "Minimum",
                    "1er quartile", "3e quartile", "Maximum"
                ],
                "Valeur": [
                    moyenne, mediane, ecart_type, minimum,
                    q1, q3, maximum
                ]
            })

            st.dataframe(
                statistiques,
                use_container_width=True,
                hide_index=True
            )

            # Histogramme
            st.subheader("📊 Distribution")

            nb_classes = st.slider(
                "Nombre de classes de l'histogramme",
                min_value=5,
                max_value=100,
                value=30,
                key="slider_histogramme"
            )

            fig_hist = px.histogram(
                df_analyse,
                x=variable,
                nbins=nb_classes,
                title=f"Distribution de {variable}"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            st.info(
                """
                💡 **Comment lire ce graphique ?**

                Un histogramme représente la répartition des valeurs
                d'une variable numérique. Les barres les plus hautes
                correspondent aux zones dans lesquelles les observations
                sont les plus nombreuses.
                """
            )

            # Interprétation de l'asymétrie
            asymetrie = serie_num.skew()
            st.write("### 🤖 Interprétation automatique")

            if abs(asymetrie) < 0.5:
                st.write("La distribution semble relativement **symétrique**.")
            elif asymetrie >= 0.5:
                st.write(
                    "La distribution présente une **asymétrie vers les valeurs élevées**. "
                    "Quelques observations importantes peuvent tirer la moyenne vers le haut."
                )
            else:
                st.write(
                    "La distribution présente une **asymétrie vers les valeurs faibles**. "
                    "Quelques observations faibles peuvent tirer la moyenne vers le bas."
                )

            # Boxplot
            st.subheader("📦 Boxplot")

            fig_box = px.box(
                df_analyse,
                x=variable,
                points="outliers",
                title=f"Boxplot de {variable}"
            )

            st.plotly_chart(fig_box, use_container_width=True)

            st.info(
                """
                💡 **Comment lire un boxplot ?**

                La boîte représente la partie centrale de la distribution.
                Le trait situé dans la boîte correspond à la médiane.
                Les points isolés peuvent correspondre à des valeurs atypiques.

                Attention : une valeur atypique n'est pas nécessairement une erreur.
                """
            )

            # Détection des outliers par IQR
            iqr = q3 - q1
            borne_basse = q1 - 1.5 * iqr
            borne_haute = q3 + 1.5 * iqr

            outliers = serie_num[
                (serie_num < borne_basse) |
                (serie_num > borne_haute)
            ]

            nb_outliers = len(outliers)
            taux_outliers = nb_outliers / len(serie_num) * 100

            st.write("### 🔎 Valeurs atypiques")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Valeurs atypiques", nb_outliers)

            with col2:
                st.metric("Part des observations", f"{taux_outliers:.2f} %")

            if nb_outliers > 0:
                st.warning(
                    f"""
                    La méthode IQR détecte **{nb_outliers} valeur(s) potentiellement atypique(s)**.

                    Les valeurs situées en dessous de **{borne_basse:.2f}** ou au-dessus de
                    **{borne_haute:.2f}** sont considérées comme atypiques selon cette méthode.
                    """
                )
            else:
                st.success("Aucune valeur atypique détectée avec la méthode IQR.")

    # ========================================================
    # ANALYSE CATÉGORIELLE / TEXTUELLE
    # ========================================================

    else:

        st.subheader(f"Analyse catégorielle : {variable}")

        serie_cat = serie.dropna()

        if len(serie_cat) == 0:
            st.warning("Cette variable ne contient aucune valeur exploitable.")

        else:
            frequences = (
                serie_cat
                .value_counts()
                .reset_index()
            )
            frequences.columns = ["Valeur", "Effectif"]
            frequences["Pourcentage"] = (
                frequences["Effectif"] / len(serie_cat) * 100
            ).round(2)

            st.subheader("🏆 Valeurs les plus fréquentes")

            max_modalites = min(50, len(frequences))

            if max_modalites > 1:
                nb_modalites = st.slider(
                    "Nombre de modalités à afficher",
                    min_value=1,
                    max_value=max_modalites,
                    value=min(10, max_modalites),
                    key="slider_modalites"
                )
            else:
                nb_modalites = 1

            st.dataframe(
                frequences.head(nb_modalites),
                use_container_width=True,
                hide_index=True
            )

            top_frequences = (
                frequences
                .head(nb_modalites)
                .sort_values("Effectif")
            )

            fig_cat = px.bar(
                top_frequences,
                x="Effectif",
                y="Valeur",
                orientation="h",
                title=f"Valeurs les plus fréquentes — {variable}"
            )

            st.plotly_chart(fig_cat, use_container_width=True)

            valeur_principale = frequences.iloc[0]["Valeur"]
            effectif_principal = frequences.iloc[0]["Effectif"]
            pourcentage_principal = frequences.iloc[0]["Pourcentage"]

            st.write("### 🤖 Interprétation automatique")
            st.write(
                f"""
                La modalité la plus fréquente est **{valeur_principale}**.

                Elle apparaît **{effectif_principal} fois** et représente environ
                **{pourcentage_principal:.2f} %** des observations renseignées.
                """
            )

            taux_unique = serie_cat.nunique() / len(serie_cat) * 100

            if taux_unique >= 95:
                st.warning(
                    """
                    ⚠️ Cette variable possède presque uniquement des valeurs différentes.

                    Elle pourrait correspondre à un **identifiant**, un numéro de référence
                    ou un champ texte libre. Une analyse de fréquence est alors généralement
                    peu pertinente.
                    """
                )
# ============================================================
# ONGLET 4 : ANALYSE BIVARIÉE
# ============================================================

with tab_bivariee:

    st.header("🔗 Analyse bivariée")

    st.write(
        """
        L'analyse bivariée consiste à étudier la relation
        entre **deux variables**.

        Le type d'analyse dépend de la nature des variables :

        - numérique + numérique → nuage de points et corrélation ;
        - numérique + catégorielle → comparaison des distributions ;
        - catégorielle + catégorielle → tableau croisé.
        """
    )

    # --------------------------------------------------------
    # SÉLECTION DES VARIABLES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        variable_x = st.selectbox(
            "Variable X",
            options=df_analyse.columns,
            key="bivariee_x"
        )

    with col2:
        variable_y = st.selectbox(
            "Variable Y",
            options=df_analyse.columns,
            index=min(1, len(df_analyse.columns) - 1),
            key="bivariee_y"
        )

    if variable_x == variable_y:

        st.warning(
            "⚠️ Sélectionnez deux variables différentes."
        )

    else:

        serie_x = df_analyse[variable_x]
        serie_y = df_analyse[variable_y]

        # Les booléens sont considérés comme catégoriels
        x_numerique = (
            pd.api.types.is_numeric_dtype(serie_x)
            and not pd.api.types.is_bool_dtype(serie_x)
        )

        y_numerique = (
            pd.api.types.is_numeric_dtype(serie_y)
            and not pd.api.types.is_bool_dtype(serie_y)
        )


        # ====================================================
        # CAS 1 : NUMÉRIQUE + NUMÉRIQUE
        # ====================================================

        if x_numerique and y_numerique:

            st.subheader("🔢 Numérique × Numérique")

            donnees_bivariees = (
                df_analyse[[variable_x, variable_y]]
                .dropna()
                .copy()
            )

            if len(donnees_bivariees) < 2:

                st.warning(
                    "Pas assez de données pour analyser cette relation."
                )

            else:

                # --------------------------------------------
                # CORRÉLATION DE PEARSON
                # --------------------------------------------

                correlation = (
                    donnees_bivariees[variable_x]
                    .corr(donnees_bivariees[variable_y])
                )

                st.metric(
                    "Corrélation de Pearson",
                    f"{correlation:.3f}"
                )

                # --------------------------------------------
                # SCATTER PLOT
                # --------------------------------------------

                fig_scatter = px.scatter(
                    donnees_bivariees,
                    x=variable_x,
                    y=variable_y,
                    title=f"{variable_x} × {variable_y}",
                    opacity=0.7
                )

                st.plotly_chart(
                    fig_scatter,
                    use_container_width=True
                )

                # --------------------------------------------
                # INTERPRÉTATION
                # --------------------------------------------

                st.write("### 🤖 Interprétation automatique")

                if pd.isna(correlation):

                    st.info(
                        """
                        La corrélation ne peut pas être calculée.
                        Cela peut notamment arriver lorsqu'une
                        variable est constante.
                        """
                    )

                else:

                    force = abs(correlation)

                    if force < 0.2:
                        interpretation = "très faible"

                    elif force < 0.4:
                        interpretation = "faible"

                    elif force < 0.6:
                        interpretation = "modérée"

                    elif force < 0.8:
                        interpretation = "forte"

                    else:
                        interpretation = "très forte"

                    if correlation > 0:
                        direction = "positive"
                    elif correlation < 0:
                        direction = "négative"
                    else:
                        direction = "nulle"

                    st.write(
                        f"""
                        La relation linéaire entre **{variable_x}**
                        et **{variable_y}** est **{interpretation}**
                        et **{direction}**.

                        Coefficient de Pearson : **{correlation:.3f}**.
                        """
                    )

                    st.info(
                        """
                        💡 Une corrélation ne démontre pas une causalité.

                        Deux variables peuvent évoluer ensemble sans
                        que l'une soit directement responsable de l'autre.
                        """
                    )


        # ====================================================
        # CAS 2 : NUMÉRIQUE + CATÉGORIELLE
        # ====================================================

        elif x_numerique != y_numerique:

            st.subheader("📦 Numérique × Catégorielle")

            if x_numerique:
                variable_num = variable_x
                variable_cat = variable_y
            else:
                variable_num = variable_y
                variable_cat = variable_x

            donnees_bivariees = (
                df_analyse[[variable_num, variable_cat]]
                .dropna()
                .copy()
            )

            nb_categories = (
                donnees_bivariees[variable_cat]
                .nunique()
            )

            st.write(
                f"""
                **Variable numérique :** {variable_num}

                **Variable catégorielle :** {variable_cat}

                **Nombre de catégories :** {nb_categories}
                """
            )

            # --------------------------------------------
            # TROP DE CATÉGORIES
            # --------------------------------------------

            if nb_categories > 30:

                st.warning(
                    """
                    ⚠️ Cette variable possède plus de 30 catégories.

                    Un boxplot contenant autant de groupes serait
                    difficile à lire.

                    Sélectionnez de préférence une variable
                    catégorielle comportant moins de modalités.
                    """
                )

            else:

                # --------------------------------------------
                # BOXPLOT
                # --------------------------------------------

                fig_box_biv = px.box(
                    donnees_bivariees,
                    x=variable_cat,
                    y=variable_num,
                    points="outliers",
                    title=(
                        f"Distribution de {variable_num} "
                        f"selon {variable_cat}"
                    )
                )

                st.plotly_chart(
                    fig_box_biv,
                    use_container_width=True
                )

                # --------------------------------------------
                # STATISTIQUES PAR GROUPE
                # --------------------------------------------

                stats_groupes = (
                    donnees_bivariees
                    .groupby(variable_cat)[variable_num]
                    .agg([
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max"
                    ])
                    .reset_index()
                )

                stats_groupes.columns = [
                    variable_cat,
                    "Effectif",
                    "Moyenne",
                    "Médiane",
                    "Écart-type",
                    "Minimum",
                    "Maximum"
                ]

                st.subheader(
                    "📐 Statistiques par catégorie"
                )

                st.dataframe(
                    stats_groupes,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    """
                    💡 Le boxplot permet de comparer la distribution
                    d'une variable numérique entre plusieurs groupes.

                    Des différences de médiane ou de dispersion peuvent
                    révéler des comportements différents selon les catégories.
                    """
                )


        # ====================================================
        # CAS 3 : CATÉGORIELLE + CATÉGORIELLE
        # ====================================================

        else:

            st.subheader("🔤 Catégorielle × Catégorielle")

            donnees_bivariees = (
                df_analyse[[variable_x, variable_y]]
                .dropna()
                .copy()
            )

            nb_x = donnees_bivariees[variable_x].nunique()
            nb_y = donnees_bivariees[variable_y].nunique()

            if nb_x > 30 or nb_y > 30:

                st.warning(
                    """
                    ⚠️ Au moins une des deux variables possède
                    plus de 30 modalités.

                    Le tableau croisé risquerait d'être très volumineux.
                    Essayez des variables avec moins de catégories.
                    """
                )

            else:

                # --------------------------------------------
                # TABLEAU CROISÉ
                # --------------------------------------------

                tableau_croise = pd.crosstab(
                    donnees_bivariees[variable_x],
                    donnees_bivariees[variable_y]
                )

                st.subheader("🧮 Tableau croisé")

                st.dataframe(
                    tableau_croise,
                    use_container_width=True
                )

                # --------------------------------------------
                # POURCENTAGES
                # --------------------------------------------

                tableau_pourcentage = pd.crosstab(
                    donnees_bivariees[variable_x],
                    donnees_bivariees[variable_y],
                    normalize="index"
                ) * 100

                tableau_pourcentage = (
                    tableau_pourcentage.round(2)
                )

                st.subheader(
                    "📊 Répartition en pourcentage"
                )

                st.dataframe(
                    tableau_pourcentage,
                    use_container_width=True
                )

                st.info(
                    """
                    💡 Le tableau croisé permet d'observer comment
                    les modalités de deux variables catégorielles
                    se répartissent les unes par rapport aux autres.

                    Les pourcentages facilitent la comparaison lorsque
                    les groupes n'ont pas les mêmes effectifs.
                    """
                )

# ============================================================
# ONGLET 5 : ANALYSE DES CORRÉLATIONS
# ============================================================

with tab_correlations:

    st.header("🔥 Analyse des corrélations")

    st.write(
        """
        Une corrélation mesure la manière dont deux variables
        numériques évoluent ensemble.

        Le coefficient varie entre **-1 et +1** :

        - proche de **+1** → forte relation positive ;
        - proche de **0** → faible relation ;
        - proche de **-1** → forte relation négative.

        ⚠️ Une corrélation ne signifie pas nécessairement
        qu'une variable est la cause de l'autre.
        """
    )

    # --------------------------------------------------------
    # SÉLECTION DES VARIABLES NUMÉRIQUES
    # --------------------------------------------------------

    colonnes_num_corr = [
        col for col in df_analyse.columns
        if (
            pd.api.types.is_numeric_dtype(df_analyse[col])
            and not pd.api.types.is_bool_dtype(df_analyse[col])
        )
    ]

    if len(colonnes_num_corr) < 2:

        st.warning(
            """
            Au moins deux variables numériques sont nécessaires
            pour calculer une matrice de corrélation.
            """
        )

    else:

        st.write(
            f"""
            **{len(colonnes_num_corr)} variables numériques**
            peuvent être utilisées pour cette analyse.
            """
        )

        # ----------------------------------------------------
        # CHOIX DE LA MÉTHODE
        # ----------------------------------------------------

        methode = st.radio(
            "Méthode de corrélation",
            options=["Pearson", "Spearman"],
            horizontal=True,
            key="methode_correlation"
        )

        if methode == "Pearson":
            methode_pandas = "pearson"

            st.info(
                """
                **Pearson** mesure principalement les relations
                **linéaires** entre deux variables numériques.

                C'est généralement la méthode la plus connue.
                """
            )

        else:
            methode_pandas = "spearman"

            st.info(
                """
                **Spearman** repose sur le classement des valeurs.

                Elle permet de détecter des relations monotones
                qui ne sont pas nécessairement parfaitement linéaires.
                """
            )

        # ----------------------------------------------------
        # CHOIX DES COLONNES
        # ----------------------------------------------------

        variables_selectionnees = st.multiselect(
            "Variables à inclure dans la matrice",
            options=colonnes_num_corr,
            default=colonnes_num_corr[:min(10, len(colonnes_num_corr))],
            key="variables_correlation"
        )

        if len(variables_selectionnees) < 2:

            st.warning(
                "Sélectionnez au moins deux variables."
            )

        else:

            # =================================================
            # CALCUL DE LA MATRICE
            # =================================================

            matrice_corr = (
                df_analyse[variables_selectionnees]
                .corr(method=methode_pandas)
            )

            # =================================================
            # HEATMAP
            # =================================================

            st.subheader("🌡️ Matrice de corrélation")

            fig_corr = px.imshow(
                matrice_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=(
                    f"Matrice de corrélation "
                    f"— méthode {methode}"
                )
            )

            fig_corr.update_layout(
                xaxis_title="Variables",
                yaxis_title="Variables"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

            st.info(
                """
                💡 **Comment lire cette matrice ?**

                Chaque case représente la corrélation entre
                deux variables.

                La diagonale vaut toujours **1**, puisqu'une
                variable est parfaitement corrélée avec elle-même.

                Il faut donc surtout observer les cases situées
                en dehors de cette diagonale.
                """
            )

            # =================================================
            # TRANSFORMATION DE LA MATRICE EN LISTE
            # =================================================

            correlations = (
                matrice_corr
                .stack()
                .reset_index()
            )

            correlations.columns = [
                "Variable 1",
                "Variable 2",
                "Corrélation"
            ]

            # Suppression des relations d'une variable avec elle-même
            correlations = correlations[
                correlations["Variable 1"]
                != correlations["Variable 2"]
            ].copy()

            # -------------------------------------------------
            # ÉVITER LES DOUBLONS
            # -------------------------------------------------
            #
            # La matrice contient :
            #
            # Age / Salaire
            # et
            # Salaire / Age
            #
            # Il s'agit de la même relation.
            # Nous n'en conservons donc qu'une seule.

            correlations["paire"] = correlations.apply(
                lambda ligne: tuple(
                    sorted([
                        ligne["Variable 1"],
                        ligne["Variable 2"]
                    ])
                ),
                axis=1
            )

            correlations = (
                correlations
                .drop_duplicates("paire")
                .drop(columns="paire")
            )

            correlations["Corrélation absolue"] = (
                correlations["Corrélation"].abs()
            )

            correlations = correlations.sort_values(
                "Corrélation absolue",
                ascending=False
            )

            # =================================================
            # CLASSEMENT DES CORRÉLATIONS
            # =================================================

            st.subheader(
                "🏆 Relations les plus fortes"
            )

            nb_relations = st.slider(
                "Nombre de relations à afficher",
                min_value=1,
                max_value=min(
                    30,
                    len(correlations)
                ),
                value=min(
                    10,
                    len(correlations)
                ),
                key="nb_relations_correlation"
            )

            top_correlations = (
                correlations
                .head(nb_relations)
                .copy()
            )

            top_correlations["Corrélation"] = (
                top_correlations["Corrélation"]
                .round(3)
            )

            top_correlations[
                "Corrélation absolue"
            ] = (
                top_correlations[
                    "Corrélation absolue"
                ]
                .round(3)
            )

            st.dataframe(
                top_correlations[
                    [
                        "Variable 1",
                        "Variable 2",
                        "Corrélation"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # INTERPRÉTATION DE LA PLUS FORTE RELATION
            # =================================================

            if not correlations.empty:

                relation_max = correlations.iloc[0]

                var1 = relation_max["Variable 1"]
                var2 = relation_max["Variable 2"]
                corr_max = relation_max["Corrélation"]

                force = abs(corr_max)

                if force < 0.2:
                    niveau = "très faible"

                elif force < 0.4:
                    niveau = "faible"

                elif force < 0.6:
                    niveau = "modérée"

                elif force < 0.8:
                    niveau = "forte"

                else:
                    niveau = "très forte"

                if corr_max > 0:
                    direction = "positive"

                elif corr_max < 0:
                    direction = "négative"

                else:
                    direction = "nulle"

                st.write(
                    "### 🤖 Interprétation automatique"
                )

                st.write(
                    f"""
                    Parmi les variables sélectionnées, la relation
                    la plus importante est observée entre
                    **{var1}** et **{var2}**.

                    Leur coefficient de corrélation est de
                    **{corr_max:.3f}**.

                    Cette relation peut être qualifiée de
                    **{niveau} et {direction}**.
                    """
                )

                # =============================================
                # VISUALISATION AUTOMATIQUE
                # =============================================

                st.subheader(
                    "🔎 Visualisation de cette relation"
                )

                fig_relation = px.scatter(
                    df_analyse,
                    x=var1,
                    y=var2,
                    opacity=0.6,
                    title=(
                        f"{var1} × {var2} "
                        f"(corrélation = {corr_max:.3f})"
                    )
                )

                st.plotly_chart(
                    fig_relation,
                    use_container_width=True
                )
# ============================================================
# ONGLET 6 : ANALYSE AVANCÉE
# ============================================================

with tab_avancee:

    st.header("🧠 Analyse avancée")

    st.write(
        """
        Cette partie présente quelques méthodes de Machine Learning
        non supervisé.

        Elles permettent d'explorer les données sans disposer
        d'une variable cible à prédire.

        Trois techniques sont proposées :

        - **PCA / ACP** : réduire le nombre de dimensions ;
        - **K-Means** : rechercher automatiquement des groupes ;
        - **Isolation Forest** : détecter des observations atypiques.
        """
    )

    # --------------------------------------------------------
    # COLONNES NUMÉRIQUES DISPONIBLES
    # --------------------------------------------------------

    colonnes_num_avance = [
        col for col in df_analyse.columns
        if (
            pd.api.types.is_numeric_dtype(df_analyse[col])
            and not pd.api.types.is_bool_dtype(df_analyse[col])
        )
    ]

    if len(colonnes_num_avance) < 2:

        st.warning(
            """
            Au moins deux variables numériques sont nécessaires
            pour réaliser une analyse avancée.
            """
        )

    else:

        variables_ml = st.multiselect(
            "Variables numériques à utiliser",
            options=colonnes_num_avance,
            default=colonnes_num_avance[:min(5, len(colonnes_num_avance))],
            key="variables_ml"
        )

        if len(variables_ml) < 2:

            st.warning(
                "Sélectionnez au moins deux variables numériques."
            )

        else:

            # =================================================
            # PRÉPARATION COMMUNE DES DONNÉES
            # =================================================

            X_ml = df_analyse[variables_ml].copy()

            # Imputation des valeurs manquantes par la médiane
            imputer = SimpleImputer(
                strategy="median"
            )

            X_impute = imputer.fit_transform(X_ml)

            # Standardisation
            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(
                X_impute
            )

            st.success(
                f"""
                {len(variables_ml)} variables utilisées sur
                {len(df_analyse)} observations.

                Les valeurs manquantes sont remplacées par la médiane,
                puis les variables sont standardisées.
                """
            )

            st.divider()

            # =================================================
            # PCA / ACP
            # =================================================

            st.subheader("1️⃣ PCA / ACP")

            st.write(
                """
                L'Analyse en Composantes Principales (ACP / PCA) permet
                de résumer plusieurs variables numériques en un nombre
                plus réduit de **composantes principales**.

                Chaque composante est une combinaison des variables
                d'origine et capture une partie de la variabilité totale
                du jeu de données.

                L'objectif est de conserver suffisamment d'information
                tout en réduisant le nombre de dimensions.
                """
            )

            # -------------------------------------------------
            # PCA COMPLÈTE POUR ÉTUDIER LA VARIANCE EXPLIQUÉE
            # -------------------------------------------------

            nb_composantes_max = min(
                X_scaled.shape[0],
                X_scaled.shape[1]
            )

            pca_complete = PCA(
                n_components=nb_composantes_max
            )

            pca_complete.fit(X_scaled)

            variance_expliquee = (
                pca_complete.explained_variance_ratio_
                * 100
            )

            variance_cumulee = np.cumsum(
                variance_expliquee
            )

            valeurs_propres = (
                pca_complete.explained_variance_
            )

            df_eboulis = pd.DataFrame({
                "Composante": [
                    f"PC{i + 1}"
                    for i in range(nb_composantes_max)
                ],
                "Numéro composante": np.arange(
                    1,
                    nb_composantes_max + 1
                ),
                "Valeur propre": valeurs_propres,
                "Variance expliquée (%)": variance_expliquee,
                "Variance cumulée (%)": variance_cumulee
            })

            # -------------------------------------------------
            # ÉBOULIS DES VALEURS PROPRES
            # -------------------------------------------------

            st.subheader(
                "📉 Éboulis des valeurs propres"
            )

            st.write(
                """
                L'éboulis aide à choisir combien de composantes conserver.

                Une composante ayant une valeur propre importante
                contient davantage d'information. On cherche généralement
                le point à partir duquel les valeurs propres diminuent
                beaucoup plus lentement : le **coude**.
                """
            )

            fig_eboulis = px.line(
                df_eboulis,
                x="Numéro composante",
                y="Valeur propre",
                markers=True,
                title="Éboulis des valeurs propres"
            )

            fig_eboulis.update_layout(
                xaxis_title="Nombre de composantes",
                yaxis_title="Valeur propre"
            )

            st.plotly_chart(
                fig_eboulis,
                use_container_width=True
            )

            # -------------------------------------------------
            # VARIANCE EXPLIQUÉE CUMULÉE
            # -------------------------------------------------

            fig_variance = px.line(
                df_eboulis,
                x="Numéro composante",
                y="Variance cumulée (%)",
                markers=True,
                title="Variance expliquée cumulée"
            )

            fig_variance.update_layout(
                xaxis_title="Nombre de composantes",
                yaxis_title="Variance cumulée (%)",
                yaxis_range=[0, 105]
            )

            st.plotly_chart(
                fig_variance,
                use_container_width=True
            )

            # -------------------------------------------------
            # CHOIX DU NOMBRE DE COMPOSANTES
            # -------------------------------------------------

            # Cas limite : s'il n'existe qu'une seule dimension exploitable
            if nb_composantes_max < 2:

                nb_composantes_pca = 1

                st.warning(
                    """
                    La PCA nécessite au moins deux dimensions exploitables
                    pour produire une projection graphique en deux axes.
                    """
                )

            else:

                nb_composantes_pca = st.slider(
                    "Nombre de composantes PCA à conserver",
                    min_value=2,
                    max_value=nb_composantes_max,
                    value=min(2, nb_composantes_max),
                    step=1,
                    key="nb_composantes_pca"
                )

                pca = PCA(
                    n_components=nb_composantes_pca
                )

                X_pca = pca.fit_transform(
                    X_scaled
                )

                variance_totale = (
                    pca.explained_variance_ratio_.sum()
                    * 100
                )

                st.metric(
                    "Variance totale conservée",
                    f"{variance_totale:.2f} %"
                )

                st.write(
                    f"""
                    Avec **{nb_composantes_pca} composantes**, l'ACP conserve
                    environ **{variance_totale:.2f} %** de la variance totale
                    présente dans les variables sélectionnées.
                    """
                )

                # -------------------------------------------------
                # TABLEAU DE VARIANCE PAR COMPOSANTE
                # -------------------------------------------------

                variance_selectionnee = pd.DataFrame({
                    "Composante": [
                        f"PC{i + 1}"
                        for i in range(nb_composantes_pca)
                    ],
                    "Variance expliquée (%)": (
                        pca.explained_variance_ratio_
                        * 100
                    ).round(2),
                    "Variance cumulée (%)": (
                        np.cumsum(
                            pca.explained_variance_ratio_
                        )
                        * 100
                    ).round(2)
                })

                st.dataframe(
                    variance_selectionnee,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # CHOIX DES AXES À PROJETER
                # -------------------------------------------------

                st.subheader(
                    "🗺️ Projection des observations"
                )

                composantes_disponibles = [
                    f"PC{i + 1}"
                    for i in range(nb_composantes_pca)
                ]

                col_axe1, col_axe2 = st.columns(2)

                with col_axe1:

                    axe_x = st.selectbox(
                        "Axe horizontal",
                        options=composantes_disponibles,
                        index=0,
                        key="axe_pca_x"
                    )

                with col_axe2:

                    axe_y = st.selectbox(
                        "Axe vertical",
                        options=composantes_disponibles,
                        index=min(
                            1,
                            len(composantes_disponibles) - 1
                        ),
                        key="axe_pca_y"
                    )

                if axe_x == axe_y:

                    st.warning(
                        """
                        Sélectionnez deux composantes différentes
                        pour obtenir une projection en deux dimensions.
                        """
                    )

                else:

                    index_x = (
                        int(axe_x.replace("PC", ""))
                        - 1
                    )

                    index_y = (
                        int(axe_y.replace("PC", ""))
                        - 1
                    )

                    df_pca = pd.DataFrame({
                        axe_x: X_pca[:, index_x],
                        axe_y: X_pca[:, index_y]
                    })

                    fig_pca = px.scatter(
                        df_pca,
                        x=axe_x,
                        y=axe_y,
                        title=(
                            f"Projection PCA — "
                            f"{axe_x} × {axe_y}"
                        ),
                        opacity=0.7
                    )

                    st.plotly_chart(
                        fig_pca,
                        use_container_width=True
                    )

                    st.info(
                        f"""
                        💡 Chaque point correspond à une observation.

                        **{axe_x}** explique
                        **{pca.explained_variance_ratio_[index_x] * 100:.2f} %**
                        de la variance et **{axe_y}** en explique
                        **{pca.explained_variance_ratio_[index_y] * 100:.2f} %**.

                        Deux observations proches sur cette projection
                        possèdent généralement des profils similaires
                        au regard des variables utilisées pour l'ACP.
                        """
                    )

                    # =================================================
                    # CONTRIBUTION / LOADINGS DES VARIABLES
                    # =================================================

                    st.subheader(
                        "📊 Importance des variables dans les axes"
                    )

                    st.write(
                        """
                        Les barres ci-dessous représentent les **coefficients
                        (loadings)** des variables dans chaque composante.

                        - une valeur **positive** signifie que la variable
                          contribue dans le sens positif de l'axe ;
                        - une valeur **négative** signifie qu'elle contribue
                          dans le sens opposé ;
                        - plus la valeur absolue est grande, plus la variable
                          joue un rôle important dans la construction de l'axe.

                        Le signe ne signifie donc pas qu'une variable est
                        "bonne" ou "mauvaise" : il indique uniquement son
                        orientation sur la composante.
                        """
                    )

                    # ---------------------------------------------
                    # LOADINGS AXE X
                    # ---------------------------------------------

                    loadings_x = pd.DataFrame({
                        "Variable": variables_ml,
                        "Coefficient": (
                            pca.components_[index_x]
                        )
                    })

                    loadings_x["Importance absolue"] = (
                        loadings_x["Coefficient"].abs()
                    )

                    loadings_x = (
                        loadings_x
                        .sort_values(
                            "Importance absolue",
                            ascending=True
                        )
                    )

                    fig_loadings_x = px.bar(
                        loadings_x,
                        x="Coefficient",
                        y="Variable",
                        orientation="h",
                        title=(
                            f"Contribution des variables à {axe_x}"
                        )
                    )

                    fig_loadings_x.add_vline(
                        x=0,
                        line_width=1
                    )

                    st.plotly_chart(
                        fig_loadings_x,
                        use_container_width=True
                    )

                    # ---------------------------------------------
                    # LOADINGS AXE Y
                    # ---------------------------------------------

                    loadings_y = pd.DataFrame({
                        "Variable": variables_ml,
                        "Coefficient": (
                            pca.components_[index_y]
                        )
                    })

                    loadings_y["Importance absolue"] = (
                        loadings_y["Coefficient"].abs()
                    )

                    loadings_y = (
                        loadings_y
                        .sort_values(
                            "Importance absolue",
                            ascending=True
                        )
                    )

                    fig_loadings_y = px.bar(
                        loadings_y,
                        x="Coefficient",
                        y="Variable",
                        orientation="h",
                        title=(
                            f"Contribution des variables à {axe_y}"
                        )
                    )

                    fig_loadings_y.add_vline(
                        x=0,
                        line_width=1
                    )

                    st.plotly_chart(
                        fig_loadings_y,
                        use_container_width=True
                    )

                    # -------------------------------------------------
                    # VARIABLES LES PLUS IMPORTANTES PAR AXE
                    # -------------------------------------------------

                    variable_principale_x = (
                        loadings_x
                        .sort_values(
                            "Importance absolue",
                            ascending=False
                        )
                        .iloc[0]
                    )

                    variable_principale_y = (
                        loadings_y
                        .sort_values(
                            "Importance absolue",
                            ascending=False
                        )
                        .iloc[0]
                    )

                    st.write(
                        "### 🤖 Lecture automatique des axes"
                    )

                    st.write(
                        f"""
                        Pour **{axe_x}**, la variable ayant le coefficient
                        le plus important en valeur absolue est
                        **{variable_principale_x['Variable']}**
                        avec un coefficient de
                        **{variable_principale_x['Coefficient']:.3f}**.

                        Pour **{axe_y}**, la variable ayant le coefficient
                        le plus important en valeur absolue est
                        **{variable_principale_y['Variable']}**
                        avec un coefficient de
                        **{variable_principale_y['Coefficient']:.3f}**.
                        """
                    )

            st.divider()

            # =================================================
            # K-MEANS
            # =================================================

            st.subheader("2️⃣ K-Means")

            st.write(
                """
                K-Means cherche automatiquement des groupes
                d'observations ayant des caractéristiques similaires.

                Le nombre de groupes, appelé **K**, doit être choisi
                avant l'entraînement du modèle.
                """
            )

            max_k = min(
                10,
                len(df_analyse) - 1
            )

            if max_k < 2:

                st.warning(
                    "Pas assez d'observations pour lancer K-Means."
                )

            else:

                # ---------------------------------------------
                # MÉTHODE DU COUDE
                # ---------------------------------------------

                st.subheader("📉 Méthode du coude")

                st.write(
                    """
                    La méthode du coude consiste à entraîner K-Means
                    avec plusieurs valeurs de **K** et à observer
                    l'évolution de l'**inertie**.

                    L'inertie représente la somme des distances au carré
                    entre les observations et le centre de leur cluster.

                    Plus K augmente, plus l'inertie diminue. L'objectif
                    est de repérer un point où ajouter de nouveaux clusters
                    n'apporte plus qu'une amélioration limitée : le
                    **coude** de la courbe.
                    """
                )

                # On teste K de 2 jusqu'à max_k.
                valeurs_k = list(
                    range(2, max_k + 1)
                )

                inerties = []
                silhouettes = []

                for k_test in valeurs_k:

                    modele_test = KMeans(
                        n_clusters=k_test,
                        random_state=42,
                        n_init=10
                    )

                    labels_test = (
                        modele_test
                        .fit_predict(X_scaled)
                    )

                    inerties.append(
                        modele_test.inertia_
                    )

                    # Le score de silhouette n'est calculable que
                    # si plusieurs clusters différents existent.
                    if len(np.unique(labels_test)) > 1:

                        silhouettes.append(
                            silhouette_score(
                                X_scaled,
                                labels_test
                            )
                        )

                    else:

                        silhouettes.append(
                            np.nan
                        )

                df_choix_k = pd.DataFrame({
                    "K": valeurs_k,
                    "Inertie": inerties,
                    "Score de silhouette": silhouettes
                })

                # ---------------------------------------------
                # GRAPHIQUE DU COUDE
                # ---------------------------------------------

                fig_coude = px.line(
                    df_choix_k,
                    x="K",
                    y="Inertie",
                    markers=True,
                    title="Méthode du coude — inertie selon le nombre de clusters"
                )

                fig_coude.update_layout(
                    xaxis_title="Nombre de clusters K",
                    yaxis_title="Inertie"
                )

                st.plotly_chart(
                    fig_coude,
                    use_container_width=True
                )

                st.info(
                    """
                    💡 **Comment lire le graphique ?**

                    Cherchez l'endroit où la courbe commence à
                    s'aplatir nettement.

                    Avant ce point, ajouter un cluster améliore fortement
                    la séparation. Après ce point, le gain devient beaucoup
                    plus faible.

                    Le coude n'est pas toujours parfaitement visible :
                    le score de silhouette ci-dessous peut alors compléter
                    l'analyse.
                    """
                )

                # ---------------------------------------------
                # SCORE DE SILHOUETTE PAR K
                # ---------------------------------------------

                fig_silhouette_k = px.line(
                    df_choix_k,
                    x="K",
                    y="Score de silhouette",
                    markers=True,
                    title="Score de silhouette selon le nombre de clusters"
                )

                fig_silhouette_k.update_layout(
                    xaxis_title="Nombre de clusters K",
                    yaxis_title="Score de silhouette"
                )

                st.plotly_chart(
                    fig_silhouette_k,
                    use_container_width=True
                )

                # Suggestion automatique fondée sur le meilleur
                # score de silhouette.
                if (
                    df_choix_k[
                        "Score de silhouette"
                    ]
                    .notna()
                    .any()
                ):

                    ligne_meilleur_k = (
                        df_choix_k
                        .loc[
                            df_choix_k[
                                "Score de silhouette"
                            ].idxmax()
                        ]
                    )

                    k_suggere = int(
                        ligne_meilleur_k["K"]
                    )

                    silhouette_suggeree = (
                        ligne_meilleur_k[
                            "Score de silhouette"
                        ]
                    )

                    st.write(
                        f"""
                        🤖 **Indication automatique :**
                        le meilleur score de silhouette parmi les valeurs
                        testées est obtenu avec **K = {k_suggere}**
                        (silhouette = **{silhouette_suggeree:.3f}**).

                        Cette valeur est une aide au choix et ne remplace
                        pas l'interprétation métier des clusters.
                        """
                    )

                # ---------------------------------------------
                # CHOIX FINAL DE K PAR L'UTILISATEUR
                # ---------------------------------------------

                k = st.slider(
                    "Nombre de clusters K à utiliser",
                    min_value=2,
                    max_value=max_k,
                    value=(
                        k_suggere
                        if "k_suggere" in locals()
                        else min(3, max_k)
                    ),
                    key="kmeans_k"
                )

                kmeans = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                clusters = kmeans.fit_predict(
                    X_scaled
                )

                score_silhouette = silhouette_score(
                    X_scaled,
                    clusters
                )

                st.metric(
                    "Score de silhouette",
                    f"{score_silhouette:.3f}"
                )

                st.info(
                    """
                    💡 Le score de silhouette est compris
                    approximativement entre -1 et 1.

                    Plus il est élevé, plus les groupes sont
                    bien séparés.
                    """
                )

                # ---------------------------------------------
                # VISUALISATION SUR LA PCA
                # ---------------------------------------------

                df_clusters = pd.DataFrame({
                    "Composante 1": X_pca[:, 0],
                    "Composante 2": X_pca[:, 1],
                    "Cluster": clusters.astype(str)
                })

                fig_clusters = px.scatter(
                    df_clusters,
                    x="Composante 1",
                    y="Composante 2",
                    color="Cluster",
                    title=f"K-Means — {k} clusters",
                    opacity=0.7
                )

                st.plotly_chart(
                    fig_clusters,
                    use_container_width=True
                )

                # ---------------------------------------------
                # TAILLE DES GROUPES
                # ---------------------------------------------

                effectifs_clusters = (
                    pd.Series(clusters)
                    .value_counts()
                    .sort_index()
                    .rename_axis("Cluster")
                    .reset_index(name="Nombre d'observations")
                )

                st.subheader(
                    "Répartition des clusters"
                )

                st.dataframe(
                    effectifs_clusters,
                    use_container_width=True,
                    hide_index=True
                )

                # ---------------------------------------------
                # PROFIL DES CLUSTERS
                # ---------------------------------------------

                df_profils = df_analyse[variables_ml].copy()

                df_profils["Cluster"] = clusters

                profils_clusters = (
                    df_profils
                    .groupby("Cluster")
                    [variables_ml]
                    .mean()
                    .round(2)
                )

                st.subheader(
                    "🧬 Profil moyen des clusters"
                )

                st.dataframe(
                    profils_clusters,
                    use_container_width=True
                )

                st.write(
                    """
                    Ce tableau permet de comprendre quelles
                    caractéristiques distinguent les différents groupes.

                    Par exemple, un cluster peut présenter des montants
                    moyens plus élevés ou des durées plus importantes.
                    """
                )

            st.divider()

            # =================================================
            # ISOLATION FOREST
            # =================================================

            st.subheader("3️⃣ Isolation Forest")

            st.write(
                """
                Isolation Forest recherche les observations
                qui se distinguent fortement du reste du jeu de données.

                Ces observations sont appelées **anomalies** ou
                **outliers multivariés**.
                """
            )

            contamination = st.slider(
                "Proportion estimée d'anomalies (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="contamination_iso"
            )

            modele_iso = IsolationForest(
                contamination=(
                    contamination / 100
                ),
                random_state=42
            )

            predictions_iso = (
                modele_iso
                .fit_predict(X_scaled)
            )

            scores_iso = (
                modele_iso
                .decision_function(X_scaled)
            )

            anomalies = (
                predictions_iso == -1
            )

            nb_anomalies = (
                anomalies.sum()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Anomalies détectées",
                    nb_anomalies
                )

            with col2:

                st.metric(
                    "Part des observations",
                    f"{nb_anomalies / len(df_analyse) * 100:.2f} %"
                )

            # -------------------------------------------------
            # VISUALISATION DES ANOMALIES
            # -------------------------------------------------

            df_iso = pd.DataFrame({
                "Composante 1": X_pca[:, 0],
                "Composante 2": X_pca[:, 1],
                "Statut": np.where(
                    anomalies,
                    "Anomalie",
                    "Normal"
                ),
                "Score anomalie": scores_iso
            })

            fig_iso = px.scatter(
                df_iso,
                x="Composante 1",
                y="Composante 2",
                color="Statut",
                hover_data=["Score anomalie"],
                title="Détection d'anomalies — Isolation Forest",
                opacity=0.7
            )

            st.plotly_chart(
                fig_iso,
                use_container_width=True
            )

            st.warning(
                """
                ⚠️ Une anomalie détectée automatiquement
                n'est pas nécessairement une erreur.

                Cela signifie uniquement que l'observation présente
                un comportement inhabituel par rapport aux autres.

                Une vérification humaine reste nécessaire.
                """
            )

            if nb_anomalies > 0:

                st.subheader(
                    "🔎 Observations détectées comme atypiques"
                )

                df_anomalies = (
                    df_analyse.loc[anomalies]
                    .copy()
                )

                df_anomalies[
                    "Score_anomalie"
                ] = scores_iso[anomalies]

                df_anomalies = (
                    df_anomalies
                    .sort_values(
                        "Score_anomalie"
                    )
                )

                st.dataframe(
                    df_anomalies.head(100),
                    use_container_width=True
                )

# ============================================================
# ONGLET 7 : NETTOYAGE ET EXPORT
# ============================================================

with tab_nettoyage:

    st.header("🛠️ Nettoyage et export")

    st.write(
        """
        Cette partie permet d'appliquer quelques opérations
        de nettoyage simples au jeu de données.

        Les modifications sont appliquées sur une **copie**
        du DataFrame original afin de ne jamais modifier
        directement les données chargées.
        """
    )

    # --------------------------------------------------------
    # COPIE DU DATAFRAME
    # --------------------------------------------------------

    df_nettoye = df_analyse.copy()

    st.info(
        f"""
        Jeu de données initial :
        **{df_analyse.shape[0]} lignes** et **{df_analyse.shape[1]} colonnes**.
        """
    )

    st.divider()

    # ========================================================
    # 1. SUPPRESSION DES DOUBLONS
    # ========================================================

    st.subheader("1️⃣ Doublons")

    nb_doublons_nettoyage = df_nettoye.duplicated().sum()

    st.write(
        f"""
        Nombre de doublons complets détectés :
        **{nb_doublons_nettoyage}**
        """
    )

    supprimer_doublons = st.checkbox(
        "Supprimer les lignes dupliquées",
        value=False,
        key="supprimer_doublons"
    )

    if supprimer_doublons:

        df_nettoye = (
            df_nettoye
            .drop_duplicates()
            .copy()
        )

        st.success(
            f"""
            Doublons supprimés.

            Le jeu de données contient maintenant
            **{len(df_nettoye)} lignes**.
            """
        )

    st.divider()

    # ========================================================
    # 2. SUPPRESSION DE COLONNES
    # ========================================================

    st.subheader("2️⃣ Suppression de colonnes")

    st.write(
        """
        Certaines colonnes peuvent être inutiles pour l'analyse,
        par exemple :

        - identifiants techniques ;
        - variables constantes ;
        - colonnes presque entièrement vides ;
        - champs non pertinents pour l'objectif étudié.
        """
    )

    colonnes_a_supprimer = st.multiselect(
        "Colonnes à supprimer",
        options=df_nettoye.columns.tolist(),
        key="colonnes_a_supprimer"
    )

    if colonnes_a_supprimer:

        df_nettoye = (
            df_nettoye
            .drop(
                columns=colonnes_a_supprimer
            )
            .copy()
        )

        st.success(
            f"""
            {len(colonnes_a_supprimer)}
            colonne(s) supprimée(s).
            """
        )

    st.divider()

    # ========================================================
    # 3. VALEURS MANQUANTES
    # ========================================================

    st.subheader("3️⃣ Gestion des valeurs manquantes")

    nb_manquants_nettoyage = (
        df_nettoye
        .isna()
        .sum()
        .sum()
    )

    st.write(
        f"""
        Le jeu de données contient actuellement
        **{nb_manquants_nettoyage} valeurs manquantes**.
        """
    )

    strategie_manquants = st.selectbox(
        "Choisissez une stratégie",
        options=[
            "Ne rien modifier",
            "Supprimer les lignes avec des valeurs manquantes",
            "Supprimer les colonnes trop incomplètes",
            "Imputer les variables numériques",
            "Imputer les variables catégorielles",
            "Imputer toutes les variables"
        ],
        key="strategie_manquants"
    )

    # --------------------------------------------------------
    # STRATÉGIE 1 : SUPPRESSION DES LIGNES
    # --------------------------------------------------------

    if strategie_manquants == "Supprimer les lignes avec des valeurs manquantes":

        avant = len(df_nettoye)

        df_nettoye = (
            df_nettoye
            .dropna()
            .copy()
        )

        apres = len(df_nettoye)

        st.warning(
            f"""
            {avant - apres} ligne(s) supprimée(s).

            Il reste **{apres} lignes**.
            """
        )

    elif strategie_manquants == "Supprimer les colonnes trop incomplètes":

        seuil_suppression = st.slider(
            "Supprimer les colonnes ayant au moins ce pourcentage de valeurs manquantes",
            min_value=10,
            max_value=100,
            value=70,
            step=5,
            key="seuil_suppression_colonnes"
        )

        taux_manquants_colonnes = (
            df_nettoye
            .isna()
            .mean()
            * 100
        )

        colonnes_trop_vides = (
            taux_manquants_colonnes[
                taux_manquants_colonnes >= seuil_suppression
            ]
            .index
            .tolist()
        )

        if colonnes_trop_vides:

            st.write(
                "Colonnes concernées :"
            )

            st.write(
                colonnes_trop_vides
            )

            df_nettoye = (
                df_nettoye
                .drop(
                    columns=colonnes_trop_vides
                )
                .copy()
            )

            st.success(
                f"""
                {len(colonnes_trop_vides)}
                colonne(s) supprimée(s).
                """
            )

        else:

            st.info(
                """
                Aucune colonne ne dépasse
                le seuil sélectionné.
                """
            )

    elif strategie_manquants == "Imputer les variables numériques":

        st.info(
            """
            💡 **Imputer** signifie remplacer une valeur manquante
            par une valeur calculée.

            Ici, les variables numériques sont complétées
            avec leur **médiane**.

            La médiane est souvent préférée à la moyenne
            lorsqu'il existe des valeurs extrêmes.
            """
        )

        colonnes_num = [
            col
            for col in df_nettoye.columns
            if (
                pd.api.types.is_numeric_dtype(df_nettoye[col])
                and not pd.api.types.is_bool_dtype(df_nettoye[col])
            )
        ]

        for col in colonnes_num:

            mediane = (
                df_nettoye[col]
                .median()
            )

            df_nettoye[col] = (
                df_nettoye[col]
                .fillna(mediane)
            )

        st.success(
            f"""
            Imputation réalisée sur
            **{len(colonnes_num)} variable(s) numérique(s)**.
            """
        )

    elif strategie_manquants == "Imputer les variables catégorielles":

        st.info(
            """
            Les valeurs manquantes des variables catégorielles
            sont remplacées par la modalité la plus fréquente,
            appelée **mode**.
            """
        )

        colonnes_cat = [
            col
            for col in df_nettoye.columns
            if not pd.api.types.is_numeric_dtype(df_nettoye[col])
            or pd.api.types.is_bool_dtype(df_nettoye[col])
        ]

        for col in colonnes_cat:

            mode = (
                df_nettoye[col]
                .mode(dropna=True)
            )

            if not mode.empty:

                df_nettoye[col] = (
                    df_nettoye[col]
                    .fillna(mode.iloc[0])
                )

        st.success(
            f"""
            Imputation réalisée sur
            **{len(colonnes_cat)} variable(s) catégorielle(s)**.
            """
        )

    elif strategie_manquants == "Imputer toutes les variables":

        colonnes_num = [
            col
            for col in df_nettoye.columns
            if (
                pd.api.types.is_numeric_dtype(df_nettoye[col])
                and not pd.api.types.is_bool_dtype(df_nettoye[col])
            )
        ]

        colonnes_cat = [
            col
            for col in df_nettoye.columns
            if col not in colonnes_num
        ]

        # Variables numériques → médiane
        for col in colonnes_num:

            mediane = (
                df_nettoye[col]
                .median()
            )

            df_nettoye[col] = (
                df_nettoye[col]
                .fillna(mediane)
            )

        # Variables catégorielles → mode
        for col in colonnes_cat:

            mode = (
                df_nettoye[col]
                .mode(dropna=True)
            )

            if not mode.empty:

                df_nettoye[col] = (
                    df_nettoye[col]
                    .fillna(mode.iloc[0])
                )

        st.success(
            """
            Toutes les valeurs manquantes pouvant être imputées
            ont été traitées.
            """
        )

    st.divider()

    # ========================================================
    # 4. RÉSUMÉ DU NETTOYAGE
    # ========================================================

    st.subheader("4️⃣ Résumé du nettoyage")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Lignes",
            df_nettoye.shape[0],
            delta=df_nettoye.shape[0] - df_analyse.shape[0]
        )

    with col2:

        st.metric(
            "Colonnes",
            df_nettoye.shape[1],
            delta=df_nettoye.shape[1] - df_analyse.shape[1]
        )

    with col3:

        valeurs_manquantes_restantes = (
            df_nettoye
            .isna()
            .sum()
            .sum()
        )

        st.metric(
            "Valeurs manquantes",
            valeurs_manquantes_restantes
        )

    with col4:

        doublons_restants = (
            df_nettoye
            .duplicated()
            .sum()
        )

        st.metric(
            "Doublons",
            doublons_restants
        )

    st.subheader("🔎 Aperçu du jeu de données nettoyé")

    st.dataframe(
        df_nettoye.head(50),
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # 5. EXPORT
    # ========================================================

    st.subheader("5️⃣ Export du fichier nettoyé")

    csv_nettoye = (
        df_nettoye
        .to_csv(
            index=False,
            sep=";",
            encoding="utf-8-sig"
        )
        .encode("utf-8-sig")
    )

    st.download_button(
        label="⬇️ Télécharger le CSV nettoyé",
        data=csv_nettoye,
        file_name="dataset_nettoye.csv",
        mime="text/csv"
    )