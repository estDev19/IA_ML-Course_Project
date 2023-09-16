# Importación de todas las librerías necesarias, NumPy y pandas para el manejo de datos, Matplotlib para visualización (gráfico),
# y las clases y funciones necesarias de la biblioteca Surprise, que se utilizarán para la construcción del sistema de recomendación.
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt  
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV, cross_validate

# Carga de los datos desde un archivo CSV llamado "restautant_data.CSV" usando pd.read_csv de pandas. Se le llama df de "data frame"
df = pd.read_csv('C:\\Users\\esteb\\Desktop\\IA&ML-Proyecto\\restaurant_data.csv')

# Creación  de un lector de surprise para interpretar las calificaciones en un rango de 1-5
reader = Reader(rating_scale=(1, 5))

# Carga de los datos desde el DataFrame de pandas a un dataset de surprise, de manera que surprise los pueda manipular. 
# Usa los atributos name  = nombre de usuario, restaurant_name = nombre del restaurante y rating = calificación
data = Dataset.load_from_df(df[['name', 'restaurant_name', 'rating']], reader)

# Se define el modelo a utilizzar como SVD(), es decir se utilizará el modelo de filtrado colaborativo llamado
# Singular Value Descomposition = Descomposición de Valores Singulares
model = SVD()

# Definición de un conjunto de hiperparámetros para probar con Grid Search = búsqueda en cuadrícula,
# para buscar la combinación óptima de hiperparámetros **Cuidado con sobreajuste al modificarlos
param_grid = {'n_factors': [150, 200, 250],  #Factores latentes, las características ocultas que el modelo aprenderá, prueba entre 150, 200 y 250
              'n_epochs' : [100,200,300],    #Cantidad de iteraciones (épocas) de entrenamiento que se llevarán a cabo, prueba entre 100, 200, 300 épocas
              'lr_all': [0.010, 0.015, 0.020], #Tamaño de los pasos que el algoritmo toma para ajustar los parámetros del modelo durante el entrenamiento prueba entre 0.010, 0.015, 0.020
              'reg_all': [0.010, 0.020, 0.030]} #Regularización de todos los parámetros, para prevenir sobreajustes, penaliza valores extremadamente altos

# Se realizará una búsqueda en cuadrícula con validación cruzada para encontrar los mejores hiperparámetros
# Divide en 5 folds, uno se usa para prueba y el resto para entrenamiento, una proporción 80/20, 20% de prueba y 80% de entramiento
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5) #Usa el error cuadrático medio (RMSE) y el error absoluto medio (MAE) para evaluar el rendimiento del modelo
grid_search.fit(data) #Entrenamiento para buscar la combinación óptima de hiperparámetros parea el modelo

# Se extraen los mejores hiperparámetros
best_params = grid_search.best_params['rmse'] #Se extraen de acuerdo a la métrica RMSE, la del error cuadrático medio, para penalizar más los errores más grandes en las predicciones.
print("Mejores hiperparámetros:", best_params)

# Se crea el modelo SVD previamente definido, utilizando los mejores hiperparámetros
model = SVD(n_factors=best_params['n_factors'],
             n_epochs=best_params['n_epochs'], 
             lr_all=best_params['lr_all'], 
             reg_all=best_params['reg_all'])

# Evaluar el rendimiento del modelo utilizando validación cruzada
cross_val_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Resultados de validación cruzada:")
for key, values in cross_val_results.items():
    print(key, sum(values) / len(values))  # Calcular el promedio manualmente

# Entrenar el modelo con todos los datos disponibles
trainset = data.build_full_trainset()
model.fit(trainset)


# Realizar una recomendación para un usuario específico (por ejemplo, 'Gustavo')
user_id = 'Ana'
user_items = df[df['name'] == user_id]['restaurant_name'].tolist()
user_unrated_items = [item for item in df['restaurant_name'].unique() if item not in user_items]

recommendations = [(item, model.predict(user_id, item).est) for item in user_unrated_items]
recommendations.sort(key=lambda x: x[1], reverse=True)

# Obtener los mejores N restaurantes recomendados
N = 5  # Puedes ajustar este valor según la cantidad de recomendaciones que desees
top_n_recommendations = recommendations[:N]

print(f'Recomendaciones para {user_id}:')
for i, (restaurant_name, score) in enumerate(top_n_recommendations, start=1):
    print(f"{i}. Restaurante: {restaurant_name}, Puntuación: {score:.2f}")



#GRAPH
# Obtener las puntuaciones exactas de las recomendaciones
puntuaciones_exactas = [score for restaurant_name, score in top_n_recommendations]

# Escalar las puntuaciones de 1-5 a una escala de 0-100
puntuaciones_scaled = [(score - 1) / 4 * 100 for score in puntuaciones_exactas]

# Obtener los nombres de los restaurantes recomendados
restaurantes = [restaurant_name for restaurant_name, score in top_n_recommendations]

# Ajustar el ancho de las barras
ancho_barras = 0.4

# Crear un rango de posiciones para las barras
posiciones = np.arange(len(restaurantes))

# Crear una figura con estilo personalizado
plt.figure(figsize=(12, 8))  # Tamaño de la figura
plt.style.use('seaborn')  # Estilo personalizado

# Crear un gráfico de barras verticales con ancho ajustado y colores agradables
bars = plt.bar(posiciones, puntuaciones_scaled, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightseagreen'], width=ancho_barras)

# Agregar etiquetas de porcentaje dentro de las barras
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}%',  # Texto de la etiqueta (porcentaje)
                xy=(bar.get_x() + bar.get_width() / 2, height),  # Coordenadas de la etiqueta
                xytext=(0, 3),  # Desplazamiento vertical de la etiqueta
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

# Establecer el rango del eje y de 0 a 105 para dejar espacio en la parte superior
plt.ylim(0, 105)  

plt.xlabel('Restaurante', fontsize=14, fontweight='bold')  # Etiqueta del eje x
plt.ylabel('Puntuación (%)', fontsize=14, fontweight='bold')  # Etiqueta del eje y
plt.title(f'Recomendaciones para {user_id}', fontsize=18, fontweight='bold')  # Título del gráfico

# Etiquetas del eje x y ubicación
plt.xticks(posiciones, restaurantes, rotation=45, ha='right', fontsize=12)

# Ajustar las configuraciones de espaciado
plt.subplots_adjust(left=0.252, bottom=0.233, right=0.8, top=0.912, wspace=0.2, hspace=0.2)

plt.show()  # Mostrar el gráfico en la ventana

