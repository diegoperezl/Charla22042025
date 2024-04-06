from flask import Flask, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import psutil
import os

# Configuración de Flask
app = Flask(__name__)

@app.route('/', methods=['POST'])
def barra():
    raise Exception()
        

##############################################
############### Item to Item #################
##############################################

################## Síncrono #################

@app.route('/cosine', methods=['POST'])
def cosine():
    """
        Calculate the similarity of similar movies
    """

    if request.is_json:
        # Nombre de los archivos de datos
        data_path = 'movielens1M.csv'
        movies_path = 'movies_data.csv'

        # Cargar los archivos de datos
        ratings = pd.read_csv(data_path, sep=':', names=['user_id', 'item_id', 'rating', 'timestamp'])
        movies = pd.read_csv(movies_path, sep='::', names=['item_id', 'title', 'genres'], engine='python')

        # Obtener un mapa para los identificadores de las películas
        item_id_map = sorted(ratings['item_id'].unique())

        # Calcular la matriz de co-ocurrencia de películas (matriz de correlación)
        co_ocurrence_matrix = pd.pivot_table(ratings, values='rating', index='user_id', columns='item_id', fill_value=0)

        # Calcular la similitud de coseno entre películas
        item_similarity = cosine_similarity(co_ocurrence_matrix.T)

        # Obtener datos JSON de la solicitud
        data = request.get_json()  
        
        # Obtener el índice de la película
        pos = item_id_map.index(data['movie'])

        # Total de resultados a mostrar
        n = 5

        # Obtener los resultados más altos del cálculo del coseno
        similar_indices = item_similarity[pos].argsort()[-n-1:-1][::-1]
        similar_similarities = item_similarity[pos][similar_indices]

        # Transformar el tipo de dato para poder convertirlo a JSON
        similar_similarities = similar_similarities.astype(float)
        similar_indices = similar_indices.astype(int)
        
        # Crear el JSON en base a los resultados
        similar_movies_json = []
        for index, similarity in zip(similar_indices, similar_similarities):
            title = movies.loc[movies['item_id'] == item_id_map[index], 'title'].values[0]
            similar_movie_json = {"movie_id": int(item_id_map[index]), "similarity": float(similarity), "title": title}
            similar_movies_json.append(similar_movie_json)
        
        # Obtener el consumo de memoria del proceso ejecutado
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        # Crear el diccionario final con el formato JSON
        title = movies.loc[movies['item_id'] == data['movie'], 'title'].values[0]
        result_json = {"movie_id": data['movie'], "title":title, "similarities": similar_movies_json, "RAM":memory_usage}

        return result_json


##############################################
#################### Main ####################
##############################################

if __name__ == '__main__':
    app.run(host='localhost', port=9090, debug=False)