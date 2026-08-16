import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


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

tab_general, tab_qualite, tab_univariee = st.tabs([
    "📋 Vue générale",
    "🧹 Qualité des données",
    "📊 Analyse univariée"
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
        options=df.columns
    )

    serie = df[variable]

    # --------------------------------------------------------
    # DÉTECTION DU TYPE
    # --------------------------------------------------------

    est_numerique = pd.api.types.is_numeric_dtype(serie)

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

        st.metric(
            "Observations",
            nb_valeurs
        )


    with col2:

        st.metric(
            "Valeurs uniques",
            nb_uniques
        )


    with col3:

        st.metric(
            "Valeurs manquantes",
            nb_manquantes
        )


    with col4:

        st.metric(
            "Taux manquant",
            f"{taux_manquant:.2f} %"
        )

        st.subheader(
            f"Analyse numérique : {variable}"
        )


        # ----------------------------------------------------
        # SUPPRESSION TEMPORAIRE DES NaN
        # ----------------------------------------------------

        serie_num = serie.dropna()


        if len(serie_num) == 0:

            st.warning(
                "Cette variable ne contient aucune valeur exploitable."
            )

        else:

            # ------------------------------------------------
            # STATISTIQUES DESCRIPTIVES
            # ------------------------------------------------

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
                    "Moyenne",
                    "Médiane",
                    "Écart-type",
                    "Minimum",
                    "1er quartile",
                    "3e quartile",
                    "Maximum"
                ],

                "Valeur": [
                    moyenne,
                    mediane,
                    ecart_type,
                    minimum,
                    q1,
                    q3,
                    maximum
                ]
            })


            st.dataframe(
                statistiques,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # HISTOGRAMME
            # ------------------------------------------------

            st.subheader("📊 Distribution")


            nb_classes = st.slider(
                "Nombre de classes de l'histogramme",
                min_value=5,
                max_value=100,
                value=30
            )


            fig_hist = px.histogram(
                df,
                x=variable,
                nbins=nb_classes,
                title=f"Distribution de {variable}"
            )


            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

            st.info(
                """
                💡 **Comment lire ce graphique ?**

                Un histogramme représente la répartition
                des valeurs d'une variable numérique.

                Les barres les plus hautes correspondent
                aux zones dans lesquelles les observations
                sont les plus nombreuses.
                """
            )

            asymetrie = serie_num.skew()

            st.write("### 🤖 Interprétation automatique")


            if abs(asymetrie) < 0.5:

                st.write(
                    """
                    La distribution semble relativement
                    **symétrique**.
                    """
                )


            elif asymetrie >= 0.5:

                st.write(
                    """
                    La distribution présente une
                    **asymétrie vers les valeurs élevées**.

                    Quelques observations importantes peuvent
                    tirer la moyenne vers le haut.
                    """
                )


            else:

                st.write(
                    """
                    La distribution présente une
                    **asymétrie vers les valeurs faibles**.

                    Quelques observations faibles peuvent
                    tirer la moyenne vers le bas.
                    """
                )

            # ------------------------------------------------
            # BOXPLOT
            # ------------------------------------------------

            st.subheader("📦 Boxplot")


            fig_box = px.box(
                df,
                x=variable,
                points="outliers",
                title=f"Boxplot de {variable}"
            )


            st.plotly_chart(
                fig_box,
                use_container_width=True
            )

            st.info(
                """
                💡 **Comment lire un boxplot ?**

                La boîte représente la partie centrale
                de la distribution.

                Le trait situé dans la boîte correspond
                à la médiane.

                Les points isolés peuvent correspondre
                à des valeurs atypiques.

                Attention : une valeur atypique n'est pas
                nécessairement une erreur.
                """
            )

            # ------------------------------------------------
            # DÉTECTION DES OUTLIERS PAR IQR
            # ------------------------------------------------

            iqr = q3 - q1

            borne_basse = q1 - 1.5 * iqr
            borne_haute = q3 + 1.5 * iqr


            outliers = serie_num[
                (serie_num < borne_basse)
                |
                (serie_num > borne_haute)
            ]


            nb_outliers = len(outliers)


            taux_outliers = (
                nb_outliers
                / len(serie_num)
                * 100
            )


            st.write("### 🔎 Valeurs atypiques")


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Valeurs atypiques",
                    nb_outliers
                )


            with col2:

                st.metric(
                    "Part des observations",
                    f"{taux_outliers:.2f} %"
                )

            if nb_outliers > 0:

                st.warning(
                    f"""
                    La méthode IQR détecte **{nb_outliers}
                    valeur(s) potentiellement atypique(s)**.

                    Les valeurs situées en dessous de
                    **{borne_basse:.2f}** ou au-dessus de
                    **{borne_haute:.2f}** sont considérées
                    comme atypiques selon cette méthode.
                    """
                )

            else:

                st.success(
                    """
                    Aucune valeur atypique détectée
                    avec la méthode IQR.
                    """
                )

if est_numerique:

    else:

        st.subheader(
            f"Analyse catégorielle : {variable}"
        )


        serie_cat = serie.dropna()


        if len(serie_cat) == 0:

            st.warning(
                "Cette variable ne contient aucune valeur exploitable."
            )

        else:

            # ------------------------------------------------
            # VALEURS LES PLUS FRÉQUENTES
            # ------------------------------------------------

            frequences = (
                serie_cat
                .value_counts()
                .reset_index()
            )

            frequences.columns = [
                "Valeur",
                "Effectif"
            ]


            frequences["Pourcentage"] = (
                frequences["Effectif"]
                / len(serie_cat)
                * 100
            ).round(2)

            st.subheader(
                "🏆 Valeurs les plus fréquentes"
            )


            nb_modalites = st.slider(
                "Nombre de modalités à afficher",
                min_value=5,
                max_value=min(
                    50,
                    len(frequences)
                ),
                value=min(
                    10,
                    len(frequences)
                )
            )


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


            st.plotly_chart(
                fig_cat,
                use_container_width=True
            )

            valeur_principale = frequences.iloc[0]["Valeur"]

            effectif_principal = frequences.iloc[0]["Effectif"]

            pourcentage_principal = frequences.iloc[0]["Pourcentage"]

            st.write(
                "### 🤖 Interprétation automatique"
            )


            st.write(
                f"""
                La modalité la plus fréquente est
                **{valeur_principale}**.

                Elle apparaît **{effectif_principal} fois**
                et représente environ
                **{pourcentage_principal:.2f} %**
                des observations renseignées.
                """
            )

            taux_unique = (
                serie_cat.nunique()
                / len(serie_cat)
                * 100
            )


            if taux_unique >= 95:

                st.warning(
                    """
                    ⚠️ Cette variable possède presque
                    uniquement des valeurs différentes.

                    Elle pourrait correspondre à un
                    **identifiant**, un numéro de référence
                    ou un champ texte libre.

                    Une analyse de fréquence est alors
                    généralement peu pertinente.
                    """
                )