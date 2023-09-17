#Importación de todas las librerías necesarias, numpy y pandas para el manejo de datos, matplotlib y seaborn para visualización (gráfico),
#y las clases y funciones necesarias de la biblioteca surprise, que se utilizarán para la construcción del sistema de recomendación.
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV, cross_validate

#Cargar los datos desde un archivo CSV llamado "restautant_data.CSV" usando pd.read_csv de pandas. Se le llama df de "data frame"
df = pd.read_csv("restaurant_data.csv")

#Se crea  de un lector de surprise para interpretar las calificaciones en un rango de 1-5
reader = Reader(rating_scale=(1, 5))

#Cargar los datos desde el DataFrame de pandas a un dataset de surprise, de manera que surprise los pueda manipular. 
#Usa los atributos name  = nombre de usuario, restaurant_name = nombre del restaurante y rating = calificación
data = Dataset.load_from_df(df[['name', 'restaurant_name', 'rating']], reader)

#Se define el modelo a utilizar como SVD(), es decir se utilizará el modelo de filtrado colaborativo llamado
#Singular Value Descomposition = Descomposición de Valores Singulares
model = SVD()

#Se define de un conjunto de hiperparámetros para probar con Grid Search = búsqueda en cuadrícula,
#para buscar la combinación óptima de hiperparámetros **Cuidado con sobreajuste al modificarlos
param_grid = {'n_factors': [150, 200, 250],  #Factores latentes, las características ocultas que el modelo aprenderá, prueba entre 150, 200 y 250
              'n_epochs' : [100,200,300],    #Cantidad de iteraciones (épocas) de entrenamiento que se llevarán a cabo, prueba entre 100, 200, 300 épocas
              'lr_all': [0.010, 0.015, 0.020], #Tamaño de los pasos que el algoritmo toma para ajustar los parámetros del modelo durante el entrenamiento prueba entre 0.010, 0.015, 0.020
              'reg_all': [0.010, 0.020, 0.030]} #Regularización de todos los parámetros, para prevenir sobreajustes, penaliza valores extremadamente altos

#Se realizará una búsqueda en cuadrícula con validación cruzada para encontrar los mejores hiperparámetros
#Está dividido en 5 folds, uno se usa para prueba y el resto para entrenamiento, una proporción 80/20, 20% de prueba y 80% de entrenamiento
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5) #Usa el error cuadrático medio (RMSE) y el error absoluto medio (MAE) para evaluar el rendimiento del modelo
grid_search.fit(data) #Entrenamiento para buscar la combinación óptima de hiperparámetros parea el modelo

#Se extraen los mejores hiperparámetros
best_params = grid_search.best_params['rmse'] #Se extraen de acuerdo a la métrica RMSE, la del error cuadrático medio, para penalizar más los errores más grandes en las predicciones.
print("Mejores hiperparámetros:", best_params)

#Crea el modelo SVD previamente definido, utilizando los mejores hiperparámetros n_factors, n_epochs, lr_all y reg_all
model = SVD(n_factors=best_params['n_factors'],
             n_epochs=best_params['n_epochs'], 
             lr_all=best_params['lr_all'], 
             reg_all=best_params['reg_all'])

#Se evalúa el rendimiento del modelo utilizando validación cruzada, dividiendo en 5 folds con la propiedad verbose para mostrar los datos 
cross_val_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Resultados de validación cruzada:")
for key, values in cross_val_results.items():
    print(key, sum(values) / len(values))  #En cada fold imprime en consola RMSE y MAE 

#Se entrena el modelo con todos los datos disponibles en el conjunto de datos, provenientes del CSV
trainset = data.build_full_trainset()
model.fit(trainset)


#Se realiza una recomendación para un usuario específico (por ejemplo, 'Ana')
user_id = 'Ana'
user_items = df[df['name'] == user_id]['restaurant_name'].tolist() #Los restaurantes que ya ha calificado el usuario
user_unrated_items = [item for item in df['restaurant_name'].unique() if item not in user_items] #Los restaurantes que NO ha calificado el usuario

recommendations = [(item, model.predict(user_id, item).est) for item in user_unrated_items] #Se hace una predicción de a acuerdo a los 
recommendations.sort(key=lambda x: x[1], reverse=True)   #estaurantes aún no visitados y se almacena en una lista de recomendaciones, 
#se ordenan de acuerdo a la calificación estimada, de mayor a menor


# Se obtienen los mejores N restaurantes recomendados (las N mejores recomendaciones)
N = 10  #Se puede ajustar el N de acuerdo a la cantidad de recomendaciones que se deseen
top_n_recommendations = recommendations[:N]

print(f'Recomendaciones para {user_id}:') #Se imprimen en consola las recomendaciones enumeradas, con su respectiva puntuación de 1 a 5
for i, (restaurant_name, score) in enumerate(top_n_recommendations, start=1):   #para el usuario que se haya definido anteriormente.
    print(f"{i}. Restaurante: {restaurant_name}, Puntuación: {score:.2f}")      #Está redondeado a dos decimales cada puntuación


#Aquí se crea un gráfico con Matplotlib con la intención de visualizar los resultados de manera más ilustrativa,
#lo cuál puede facilitar su comprensión.
#El gráfico está en una escala de 1 a 100% (el valor se representa sobre cada bara, su equivalente de 1 a 5 se aprecia debajo de
#la barra en color gris claro), representa las puntuaciones de las recomendaciones dadas por el modelo

# Se obtienen las puntuaciones exactas de las recomendaciones
puntuaciones_exactas = [score for restaurant_name, score in top_n_recommendations]

# Se escalan las puntuaciones de 1-5 a una escala de 1-100 con dos decimales de precisión
puntuaciones_scaled = [(score / 5) * 100 for score in puntuaciones_exactas]

# Se obtienen los nombres de los restaurantes recomendados
restaurantes = [restaurant_name for restaurant_name, score in top_n_recommendations]

# Ancho de las barras del gráfico
ancho_barras = 0.4

# Se crea un rango de posiciones para las barras
posiciones = np.arange(len(restaurantes))

# Se crea una figura con estilo personalizado
plt.figure(figsize=(12, 8))  # Tamaño de la figura
sns.set(style="whitegrid")  # Estilo personalizado

# Se crea un gráfico de barras verticales con ancho ajustado y colores suaves para cada barra,
# pensado para 5 barras, si son más, se repiten los colores
bars = plt.bar(posiciones, puntuaciones_scaled, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightseagreen'], width=ancho_barras)

# Se agregan las etiquetas del porcentaje que alcanza cada una de las barras con dos decimales
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%',  # El texto de la etiqueta en cada barra, con dos decimales
                xy=(bar.get_x() + bar.get_width() / 2, height),  # Coordenadas de la etiqueta en cada barra
                xytext=(0, 3),  # Desplazamiento vertical de la etiqueta
                textcoords="offset points",  # La posición se establece desde un punto de referencia
                ha='center', va='bottom', fontsize=12)  # Alineación del texto
    
    # Se agrega el equivalente en la escala de 1 a 5 en la parte inferior de cada barra con dos decimales
    plt.annotate(f'{(height / 100 * 5):.2f}',  
                xy=(bar.get_x() + bar.get_width() / 2, 0),  # Coordenadas de la etiqueta en la parte inferior de cada barra
                xytext=(0, -10),  # Desplazamiento vertical hacia abajo
                textcoords="offset points",  # La posición se establece desde un punto de referencia
                ha='center', va='bottom', fontsize=10, color='gray')  # Alineación del texto y color
    
# Se establece el rango del eje y de 0 a 105 (5 después del 100 para dejar espacio en la parte superior)
plt.ylim(0, 105)

plt.xlabel('Restaurante', fontsize=14, fontweight='bold')  # Etiqueta del eje x
plt.ylabel('Puntuación (%)', fontsize=14, fontweight='bold')  # Etiqueta del eje y
plt.title(f'Recomendaciones para {user_id}', fontsize=18, fontweight='bold')  # Título del gráfico

# Etiquetas del eje x y ubicación
plt.xticks(posiciones, restaurantes, rotation=45, ha='right', fontsize=12)  # Rotación de 45 grados para que se aprecien los nombres sin interferir con los que están a los lados

# Se ajustan las configuraciones de espaciado
plt.subplots_adjust(left=0.252, bottom=0.26, right=0.8, top=0.912, wspace=0.2, hspace=0.2)

plt.show()  # Se despliega el gráfico en la ventana