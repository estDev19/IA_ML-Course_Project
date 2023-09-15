import random

# Lista de usuarios y sus preferencias de comida
usuarios = [
    {"nombre": "Rafael", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Veronica", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Gustavo", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Jennifer", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Oscar", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]}
]

# Lista de restaurantes y tipos de comida
restaurantes = [
    {"nombre": "McDonalds", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Green leaf", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "El novillo alegre", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Italianissimo", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Sakura Japanise", "tipo_comida": ["sushi"]},
    {"nombre": "Taj Mahal", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Ajua restaurante", "tipo_comida": ["comida_mexicana"]},
    {"nombre": "TacoBell", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Patio verde", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "La Parrilla Argentina", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Benvenuti Italia", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Sushi Sensei", "tipo_comida": ["sushi"]},
    {"nombre": "Asian Flavor Fusion", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Mexicanos ", "tipo_comida": ["comida_mexicana"]}
]

# Función para generar calificaciones coherentes para un usuario
def generar_calificaciones(usuario, registros):
    calificaciones = []
    for _ in range(7):
        # Elegir un restaurante al azar que no haya sido calificado previamente por el usuario
        restaurantes_no_calificados = [restaurante for restaurante in restaurantes if restaurante["nombre"] not in registros[usuario["nombre"]]]
        if not restaurantes_no_calificados:
            break  # Si el usuario ha calificado todos los restaurantes, salir del bucle
        
        restaurante = random.choice(restaurantes_no_calificados)
        registros[usuario["nombre"]].append(restaurante["nombre"])  # Registrar el restaurante calificado
        
        # Verificar si el restaurante es coherente con las preferencias del usuario
        for preferencia in usuario["preferencias"]:
            if preferencia in restaurante["tipo_comida"]:
                calificacion = random.randint(4, 5)  # Calificación positiva
                break
        else:
            calificacion = random.randint(1, 3)  # Calificación neutral o negativa
        
        calificaciones.append({"usuario": usuario["nombre"], "restaurante": restaurante["nombre"], "calificacion": calificacion})
    
    return calificaciones

# Inicializar registros vacíos para cada usuario
registros = {usuario["nombre"]: [] for usuario in usuarios}

# Generar registros para cada usuario
registros_totales = []
for usuario in usuarios:
    registros_totales.extend(generar_calificaciones(usuario, registros))

# Imprimir los registros generados en formato CSV en la consola
print("nombre,restaurante,calificacion")
for registro in registros_totales:
    print(f"{registro['usuario']},{registro['restaurante']},{registro['calificacion']}")
