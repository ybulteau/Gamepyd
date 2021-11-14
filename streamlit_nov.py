# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:51:42 2021

"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


from sklearn.linear_model import  LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.decomposition import PCA
from PIL import Image
    
# Importation du dataframe






st.title('Gamepyd Project')

menu = st.sidebar.radio('Menu', ['Visualisation', 
                                 'Web Scraping', 
                                 'Modèle : Ventes mondiales', 
                                 'Modèle : Ventes par régions'])

if menu == 'Visualisation':
    
    st.header('Visualisation')
    st.markdown('Nous avons travaillé avec deux datasets. Le premier a été '
                'obtenu sur kaggle, et vient du webscraping '
                'du site vgchartz.com. Le deuxième est issu de la fusion '
                'entre le premier et les données '
                'que nous avons nous-mêmes scrapé sur le site '
                'www.metacritic.com (voir partie Web Scraping). '
                'Le premier contient plus de jeux, tandis que le deuxième a '
                'l\'avantage de contenir les notes '
                'des jeux (variable rating).')
    
    
    # Choix du dataset
    data_choice = st.radio('Dataset', ['VGchartz', 
                                       'VGchartz fusionné avec Metacritic'])
    if data_choice == 'VGchartz':
        df = pd.read_csv('vgsales.csv', index_col = 0)
    elif data_choice == 'VGchartz fusionné avec Metacritic':
        df = pd.read_csv('df_merged.csv', index_col = 0)
    
    # Suppression des valeurs manquantes
    df = df.dropna()
    
    # Affichage de la taille du dataframe
    st.write('20 jeux les plus vendus:')
    st.dataframe(df.head(20))
    st.write("df.shape :", df.shape)
    
    # Choix du type de graph
    graph_type = st.selectbox('Type de graph', ['Dispersion', 
                                                'Ventes Moyennes', 
                                                'Ventes Moyennes par Année '
                                                '(top 10)'])
    
    
    
    if graph_type == 'Dispersion':
        
        # Choix de la variable
        variable = st.selectbox('Variable', ['Genre', 
                                             'Platform', 
                                             'Global_Sales', 
                                             'NA_Sales', 
                                             'EU_Sales', 
                                             'JP_Sales', 
                                             'rating', 
                                             'Year'])
        
        # Affichage du graph
        if (variable == 'Genre') or (variable == 'Platform'):
            fig = plt.figure()
            sns.countplot(x = variable, data = df, 
                          order = df[variable].value_counts().index)
            plt.xticks(rotation = 45);
            st.pyplot(fig)
        else:
            fig = sns.displot(x = variable, data = df)
            st.pyplot(fig)
            
    if graph_type == 'Ventes Moyennes':
        
        # Choix de la variable
        variable = st.selectbox('Variable', ['Genre', 'Platform', 'Publisher'])
        
        # Calcul de la moyenne et tri
        df_me = df[['Global_Sales', variable]].groupby(variable).mean()
        df_me = df_me.sort_values(by = 'Global_Sales', ascending = False)
        
        # Si la variable est 'Publisher' : Selection des 15 plus grandes 
        # valeurs
        if variable == 'Publisher':
            df_me = df_me.iloc[:15]
            
        # Affichage du graph
        fig, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(ax = ax1, x = df_me.index, y = df_me['Global_Sales'])
        plt.xticks(rotation = 45);
        st.pyplot(fig)
        
    if graph_type == 'Ventes Moyennes par Année (top 10)':
        
        # Choix de la variable
        variable = st.selectbox('Variable', ['Genre', 
                                             'Platform', 
                                             'Publisher'])
        
        # Calcul de la moyenne et tri, selection des  plus grandes valeurs
        df_me = df[['Global_Sales', variable]].groupby(variable).mean()
        df_me = df_me.sort_values(by = 'Global_Sales', ascending = False)
        df_top = df_me.iloc[:10]
        
        # Calcul de la moyenne pour chaque année
        df_y = df[['Global_Sales', 'Year', variable]]
        df_y = df_y[df_y[variable].isin(df_top.index)]
        df_y_m = df_y.groupby([variable, 'Year']).mean()
        
        # Affichage du graph
        if variable == 'Publisher':
            fig = sns.relplot(x = 'Year', y = 'Global_Sales', data = df_y_m, 
                              hue = variable, kind = 'scatter')
        else:
            fig = sns.relplot(x = 'Year', y = 'Global_Sales', data = df_y_m, 
                              hue = variable, kind = 'line')
        st.pyplot(fig)
            
    

if menu == 'Modèle : Ventes mondiales':
    
    st.header('Modèle : Ventes mondiales')
    
    # Choix du dataset
    data_choice = st.radio('Dataset', ['VGchartz', 
                                       'VGchartz merged with Metacritic'])
    if data_choice == 'VGchartz':
        df = pd.read_csv('vgsales.csv', index_col = 0)
    elif data_choice == 'VGchartz merged with Metacritic':
        df = pd.read_csv('df_merged.csv', index_col = 0)
  
    st.write("df.shape :", df.shape)  
  
    # Suppression des valeurs manquantes et de la variable Name
    df = df.dropna()
    df = df.drop(['Name','EU_Sales', 'JP_Sales', 'NA_Sales', 'Other_Sales'], 
                 axis = 1)
    
    
    # Choix du nombre d variables à retenir
    sk_num = st.select_slider("Nombre de variable à retenir", 
                              [50, 100, 150, 200])
    
    # Choix du modèle de régression
    model_key = st.selectbox('Model', ['Linear Regression', 
                                       'Lasso', 
                                       'Random Forest Regressor', 
                                       'Voting Regressor'])
    
    # Séparation en variables cible et explicatives
    target = df['Global_Sales']
    feats = df.drop(['Global_Sales'], axis = 1)
    
    # One-hot encoding
    feats = feats.join(pd.get_dummies(feats[['Platform', 
                                             'Genre', 
                                             'Publisher']]))
    feats = feats.drop(['Platform', 'Genre', 'Publisher'], axis = 1)
    
     # Séparation en jeux d'entrainement et de test
    X_train, X_test, y_train, y_test, \
    i_train, i_test = train_test_split(feats, 
                                       target, 
                                       feats.index, 
                                       test_size=0.2, 
                                       random_state=50)
    sk = SelectKBest(f_regression, k=sk_num)
    
    # Création des modèles
    # Régression linéaire
    linreg = LinearRegression()
    
    # Régression lasso
    alpha = [50, 25, 10,1,0.1, 0.05, 0.01 , 0.005 ,0.001, 0.0005, 0.0001]
    model_las = LassoCV(alphas = alpha, cv=8)
    
    # Random Forest
    model_rf = RandomForestRegressor()
    
    # Voting Regressor
    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = LinearRegression()
    model_VR = VotingRegressor(estimators=[('gb', reg1), 
                                           ('rf', reg2), 
                                           ('lr', reg3)])
    
    # Création d'un dictionnaire de modèles
    model_dict = {'Linear Regression' : linreg, 
                  'Lasso' : model_las, 
                  'Random Forest Regressor': model_rf, 
                  'Voting Regressor': model_VR}
    
    model = model_dict[model_key]
    
    # Création et entrainement de la pipeline
    model_pipe = Pipeline(steps = [('selectkbest', sk), 
                                   ('linear_regression', model)])
    model_pipe.fit(X_train, y_train)
    
    # Affichage des scores
    st.write("R2 et RMSE pour l'ensemble train")
    st.write(model_pipe.score(X_train, y_train),
             mean_squared_error(model_pipe.predict(X_train), y_train))

    st.write("R2 et RMSE pour l'ensemble test")
    st.write(model_pipe.score(X_test, y_test),
             mean_squared_error(model_pipe.predict(X_test), y_test))
        
if menu == 'Web Scraping':
    
    st.header('Web Scraping')
    st.markdown('Notre code de webscraping nous permet d\'extraire '
                'l\'ensemble '
                'des jeux vidéos contenus dans la base de donnée du site '
                'www.metacritic.com '
                'grâce à la librairie Beautiful Soup. Ici, vous pouvez '
                'choisir une page de la liste des jeux de metacritic et '
                'afficher les données obtenus par le scraping.', )

    # Initialisation des listes
    game = []
    platform = []
    rating = []
    date = []
    
    # Choix du numéro de page
    pa = st.text_input('Numéro de la page :', value = 1)
    pa = int(pa) -1
    
    # Connexion au site
    url = "https://www.metacritic.com/browse/games/score/metascore/all/all" \
          "/filtered?page=%d" % pa
    req = Request(url, headers = {'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(urlopen(req).read(), 'html.parser')
    
    # Extraction des informations correspondantes
    games = soup.find_all(name = 'a', attrs = {'class' : 'title'})
    platforms = soup.find_all(name = 'span', attrs = {'class' : 'data'})
    ratings = soup.find_all(name = 'div', 
                            attrs = {'class' : re.compile('metascore_w '\
                                                          'large game*')})
    dates = soup.find_all(attrs = {'class' : 'clamp-details'})
    
    # Nettoyage du texte
    for g, p, r, d in zip(games, platforms, ratings, dates):
        game.append(g.text)
        platform.append(p.
                        text.
                        replace('\n', '').
                        replace('                                        ',
                                '').
                        replace('                                    ', ''))
        rating.append(r.text)
        date.append(d.text[-5:-1])
        
    # Création et affichage d'une dataframe contant les données 
    df = pd.DataFrame({'game' : game, 
                       'platform' : platform, 
                       'rating' : rating, 
                       'date' : date})
    st.write(url)
    st.dataframe(df)

if menu == 'Modèle : Ventes par régions':
    
    @st.cache(suppress_st_warning=True)
    def selecteur_fichier(selected_data):
         
        # Importation du jeu de données
        if selected_data == 'VGchartz':
            df = pd.read_csv('vgsales.csv', index_col=0)
        elif selected_data == 'VGchartz fusionné avec Metacritic':
            df = pd.read_csv('df_merged.csv', index_col=0)
        
        df = df.dropna()
        return df
    
    
    @st.cache(suppress_st_warning=True, allow_output_mutation=True) 
    def modelisation(df, selected_target, selected_k, selected_model):      
        # Séparation de la variable cible et des features.
        table_target = dict({"Europe": "EU_Sales", 
                             "Amerique": "NA_Sales", 
                             "Japon": "JP_Sales", 
                             "Reste du monde": "Other_Sales"})
        cible=table_target[selected_target]
        target = df[cible]
        feats = df.drop(cible, axis = 1)
    
        # Dichotomisation des variables catégorielles.
        feats = feats.join(pd.get_dummies(feats[['Platform', 
                                                 'Genre', 
                                                 'Publisher']]))
        feats = feats.drop(['Platform', 'Genre', 'Publisher'], axis = 1)
    
        # Séparation en jeu d'entrainement et de test.
        X_train, X_test, y_train, y_test,\
        i_train, i_test = train_test_split(feats, 
                                           target, 
                                           feats.index, 
                                           test_size=0.2, 
                                           random_state=50)
    
        # Standardisation des données
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = pd.Series(scaler.fit_transform(y_train.values.
                                                 reshape(-1, 1))[:, 0])
        y_test = pd.Series(scaler.transform(y_test.values.
                                            reshape(-1, 1))[:, 0])
    
        # Selection des meilleures features
        sk = SelectKBest(f_regression, k=selected_k).fit(X_train, y_train)
        sk_train = sk.transform(X_train)
        sk_test = sk.transform(X_test)
    
        # Modelisation
        if selected_model == "Linear Regression":
            model = LinearRegression()
        elif selected_model == "Lasso":
            alpha = [50, 25, 10,1,0.1, 0.05, 0.01,
                     0.005 ,0.001, 0.0005, 0.0001]
            model = LassoCV(alphas = alpha, cv=8)
        elif selected_model == "Voting Regressor":
            reg1 = GradientBoostingRegressor(random_state=1)
            reg2 = RandomForestRegressor(random_state=1)
            reg3 = LinearRegression()
            model = VotingRegressor(estimators=[('gb', reg1), 
                                                ('rf', reg2), 
                                                ('lr', reg3)])
    
        model.fit(sk_train, y_train)
        y_pred_train = model.predict(sk_train)
        y_pred_test = model.predict(sk_test) 
        return scaler, model, X_train, X_test, sk_train, sk_test, y_train,\
               y_test, y_pred_train, y_pred_test, i_train, i_test
    
        
    @st.cache(suppress_st_warning=True) 
    def print_pca(X_train, y_train, i_train, scaler):
        pca = PCA(n_components=2)
        pca.fit(X_train)
        data_2D = pca.transform(X_train)  
        data_pca = pd.DataFrame(data_2D, columns=['ax_1', 'ax_2'], 
                                index=i_train)
        data_pca["log(1+sales)"] = np.log(1+scaler.
                                          inverse_transform(
                                              y_train.values.reshape(-1, 1) ))
        return data_pca

    @st.cache(suppress_st_warning=True) 
    def show_image(selected_target):
        image = Image.open(selected_target +'.png')
        return image
    
    st.header('Modèle : Ventes par régions')
    # Affichage dans du dataframe dans streamlit
    selected_data = st.radio('Dataset', ['VGchartz', 
                                         'VGchartz fusionné avec Metacritic'])
    
    df = selecteur_fichier(selected_data)
    st.write("df.shape :", df.shape)
    
    # Suppression des variables non utilisées
    df = df.drop(['Name', 'Global_Sales'], axis = 1)
    
    
    # Choix parametres et target pour la modélisation
    st.header("Prédiction des ventes par région")
    
    selected_target = st.radio("Cible", ["Europe", 
                                         "Amerique", 
                                         "Japon", 
                                         "Reste du monde"])

    st.image(show_image(selected_target))
    
    selected_k = st.select_slider("Nombre de variable à retenir", 
                                  [50, 100, 150, 200])
    
    selected_model = st.selectbox("Model", ["Linear Regression", 
                                            "Lasso", 
                                            "Voting Regressor"])
    
    # Modélisation
    scaler, model, X_train, X_test, sk_train, sk_test, \
    y_train, y_test, y_pred_train, y_pred_test, \
    i_train, i_test = modelisation(df, 
                                   selected_target, 
                                   selected_k, 
                                   selected_model)
    
    # Affichage des résultats    
    st.write("R2 et RMSE pour l'ensemble train")
    st.write(model.score(sk_train, y_train), 
             mean_squared_error(y_pred_train, y_train))
    
    st.write("R2 et RMSE pour l'ensemble test")
    st.write(model.score(sk_test, y_test), 
             mean_squared_error(y_pred_test, y_test))
        
    # Matrice de correlation
    if st.button("Matrice de Corrélation"):
        corr = df.drop(['Platform', 'Genre', 'Publisher'], axis=1).corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, center=0, annot=True, cmap='RdBu_r');
        st.pyplot(fig);
    
    # Affichage du graphe PCA
    st.write("_____")
    st.header("PCA 2D")
    
    data_pca = print_pca(X_train, y_train, i_train, scaler)
    
    fig = sns.relplot(data=data_pca, 
                      x='ax_1', 
                      y='ax_2', 
                      hue="log(1+sales)", 
                      kind = 'scatter', 
                      palette = "Spectral_r", 
                      alpha=0.8);
    st.pyplot(fig);