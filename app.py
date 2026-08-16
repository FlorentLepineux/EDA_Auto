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

tab_general, tab_qualite, tab_univariee, tab_bivariee, tab_correlations = st.tabs([
    "📋 Vue générale",
    "🧹 Qualité des données",
    "📊 Analyse univariée",
    "🔗 Analyse bivariée",
    "🔥 Corrélations"
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
                df,
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
                df,
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
            options=df.columns,
            key="bivariee_x"
        )

    with col2:
        variable_y = st.selectbox(
            "Variable Y",
            options=df.columns,
            index=min(1, len(df.columns) - 1),
            key="bivariee_y"
        )

    if variable_x == variable_y:

        st.warning(
            "⚠️ Sélectionnez deux variables différentes."
        )

    else:

        serie_x = df[variable_x]
        serie_y = df[variable_y]

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
                df[[variable_x, variable_y]]
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
                df[[variable_num, variable_cat]]
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
                df[[variable_x, variable_y]]
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
        col for col in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
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
                df[variables_selectionnees]
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
                    df,
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