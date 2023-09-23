#Autor: Esteban Castro Oviedo, setiembre 2023
import random

# Lista de usuarios y sus preferencias de comida
usuarios = [
    {"nombre": "Rafael", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Veronica", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Gustavo", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Jennifer", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Oscar", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Mauricio", "preferencias": ["comida_rapida", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Raquel", "preferencias": ["comida_vegetariana", "comida_italiana", "sushi"]},
    {"nombre": "Minor", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Daniela", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Jose", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Luis", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Elena", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Carlos", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Maria", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Andres", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Laura", "preferencias": ["comida_rapida", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Pedro", "preferencias": ["comida_vegetariana", "comida_italiana", "sushi"]},
    {"nombre": "Isabel", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Antonio", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Ana", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Alberto", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Beatriz", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Diana", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Eduardo", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Fernanda", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Gabriel", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Hector", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Irene", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Javier", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Karla", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Luisa", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Mario", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Natalia", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Omar", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]},
    {"nombre": "Patricia", "preferencias": ["comida_rapida", "comida_italiana", "sushi"]},
    {"nombre": "Quintín", "preferencias": ["comida_vegetariana", "comida_italiana", "comida_mexicana"]},
    {"nombre": "Rosa", "preferencias": ["comida_vegetariana", "comida_asiatica", "sushi"]},
    {"nombre": "Sergio", "preferencias": ["comida_carnes", "comida_mexicana", "comida_italiana"]},
    {"nombre": "Tania", "preferencias": ["comida_rapida", "comida_asiatica", "comida_mexicana"]}
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
    {"nombre": "Mexicanos", "tipo_comida": ["comida_mexicana"]},
     {"nombre": "KFC", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Vegan food", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "Braza novillo", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Venezia qui", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Noguchi sushi", "tipo_comida": ["sushi"]},
    {"nombre": "Thai cuisine", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Plato Azteca", "tipo_comida": ["comida_mexicana"]},
    {"nombre": "Burger King", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Vegetarian Delight", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "Steakhouse Paradise", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Mama Mia", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Sushi Heaven", "tipo_comida": ["sushi"]},
    {"nombre": "Asian Fusion Delights", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Mexican Fiesta", "tipo_comida": ["comida_mexicana"]},
    {"nombre": "Chicken Express", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Green Oasis", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "Grill Masters", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Italian Elegance", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Sushi Masters", "tipo_comida": ["sushi"]},
    {"nombre": "Far East Flavors", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Mexican Delights", "tipo_comida": ["comida_mexicana"]},
    {"nombre": "Fried Chicken Express", "tipo_comida": ["comida_rapida"]},
    {"nombre": "Veggie Delights", "tipo_comida": ["comida_vegetariana"]},
    {"nombre": "Grill House", "tipo_comida": ["comida_carnes"]},
    {"nombre": "Italiano Bella", "tipo_comida": ["comida_italiana"]},
    {"nombre": "Sushi Sensations", "tipo_comida": ["sushi"]},
    {"nombre": "Taste of Asia", "tipo_comida": ["comida_asiatica"]},
    {"nombre": "Guadalajara Platter", "tipo_comida": ["comida_mexicana"]}

]

# Esta función genera calificaciones coherentes para un usuario
def generar_calificaciones(usuario, registros):
    calificaciones = []
    for _ in range(26):
        # Elige un restaurante al azar que no haya sido calificado previamente por el usuario
        restaurantes_no_calificados = [restaurante for restaurante in restaurantes if restaurante["nombre"] not in registros[usuario["nombre"]]]
        if not restaurantes_no_calificados:
            break  # Si el usuario ha calificado todos los restaurantes, salir del bucle
        
        restaurante = random.choice(restaurantes_no_calificados)
        registros[usuario["nombre"]].append(restaurante["nombre"])  # Registrar el restaurante calificado
        
        # Verifica si el restaurante es coherente con las preferencias del usuario
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
