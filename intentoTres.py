import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV, cross_validate

# Cargar los datos desde un archivo CSV
df = pd.read_csv('C:\\Users\\esteb\\Desktop\\IA&ML-Proyecto\\restaurant_data.csv')

# Crear un lector para interpretar las calificaciones en el rango de 1-5
reader = Reader(rating_scale=(1, 5))

# Cargar los datos desde el DataFrame de pandas
data = Dataset.load_from_df(df[['nombre', 'restaurante', 'calificacion']], reader)

# Definir el modelo SVD
model = SVD()

# Definir un conjunto de hiperparámetros para probar con Grid Search
param_grid = {'n_factors': [50, 100, 150],
              'n_epochs': [40, 50],
              'lr_all': [0.015, 0.020],
              'reg_all': [0.05, 0.01]}

# Realizar una búsqueda en cuadrícula con validación cruzada para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
grid_search.fit(data)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params['rmse']
print("Mejores hiperparámetros:", best_params)

# Crear el modelo SVD con los mejores hiperparámetros
model = SVD(n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'],
            lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])

# Evaluar el rendimiento del modelo utilizando validación cruzada
# Evaluar el rendimiento del modelo con validación cruzada
cross_val_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Resultados de validación cruzada:")
for key, values in cross_val_results.items():
    print(key, sum(values) / len(values))  # Calcular el promedio manualmente


# Entrenar el modelo con todos los datos disponibles
trainset = data.build_full_trainset()
model.fit(trainset)

# Realizar una recomendación para un usuario específico (por ejemplo, 'Gustavo')
user_id = 'Gustavo'
user_items = df[df['nombre'] == user_id]['restaurante'].tolist()
user_unrated_items = [item for item in df['restaurante'].unique() if item not in user_items]

recommendations = [(item, model.predict(user_id, item).est) for item in user_unrated_items]
recommendations.sort(key=lambda x: x[1], reverse=True)

# Obtener los mejores N restaurantes recomendados
N = 5  # Puedes ajustar este valor según la cantidad de recomendaciones que desees
top_n_recommendations = recommendations[:N]

print(f'Recomendaciones para {user_id}:')
for i, (restaurant, score) in enumerate(top_n_recommendations, start=1):
    print(f"{i}. Restaurante: {restaurant}, Puntuación: {score:.2f}")
