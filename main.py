import pandas as pd
import requests
from ortools.linear_solver import pywraplp
import folium
import numpy as np  

# 1 Récupérer les données depuis deux fichiers CSV
def load_data(passengers_file, vehicles_file):
    passengers = pd.read_csv(passengers_file)
    vehicles = pd.read_csv(vehicles_file)
    return passengers.to_dict(orient='records'), vehicles.to_dict(orient='records')

# 2 Calculer les distances avec OpenRoute Service
def calculate_distance_matrix(passengers, vehicles, api_key):
    base_url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    coords = [f"{p['longitude']},{p['latitude']}" for p in passengers] + [f"{v['longitude']},{v['latitude']}" for v in vehicles]
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        "locations": [list(map(float, coord.split(','))) for coord in coords],
        "metrics": ["distance"],
        "units": "km"
    }
    response = requests.post(base_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erreur API: {response.status_code}, {response.text}")
    matrix = response.json()['distances']
    
    # Convertir la matrice en un tableau NumPy
    matrix = np.array(matrix)
    
    # Extraire la sous-matrice correspondant aux distances entre passagers et véhicules
    return matrix[:len(passengers), len(passengers):]

# 3 Modéliser et résoudre le problème d'assignation avec OR-Tools
def solve_assignment_problem(distance_matrix, passengers, vehicles):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver non disponible.")

    num_passengers = len(passengers)
    num_vehicles = len(vehicles)

    # Variables de décision : x[i][j] = 1 telque i passager et j véhicule, si le passager i est assigné au véhicule j
    x = {}
    for i in range(num_passengers):
        for j in range(num_vehicles):
            x[i, j] = solver.IntVar(0, 1, f'x[{i},{j}]')

    # Contraintes : On a chaque passager doit être assigné à exactement un véhicule
    for i in range(num_passengers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_vehicles)]) == 1)

    # Fonction objectif : Minimiser la somme des distances
    objective_terms = []
    for i in range(num_passengers):
        for j in range(num_vehicles):
            objective_terms.append(distance_matrix[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    # Résoudre le problème
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        assignments = []
        for i in range(num_passengers):
            for j in range(num_vehicles):
                if x[i, j].solution_value() > 0.5:
                    assignments.append((passengers[i]['id'], vehicles[j]['id']))
        return assignments
    else:
        raise Exception("Aucune solution trouvée.")

# 4 Afficher les résultats sur une carte
def display_results_on_map(passengers, vehicles, assignments):
    m = folium.Map(location=[35.3055, -1.1402], zoom_start=12)  # Centré sur Aïn Témouchent

    # Ajouter les passagers
    for passenger in passengers:
        folium.Marker(
            location=[passenger['latitude'], passenger['longitude']],
            popup=f"Passenger {passenger['id']}",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    # Ajouter les véhicules
    for vehicle in vehicles:
        folium.Marker(
            location=[vehicle['latitude'], vehicle['longitude']],
            popup=f"Vehicle {vehicle['id']}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # Ajouter les lignes pour les assignations
    for passenger_id, vehicle_id in assignments:
        passenger = next(p for p in passengers if p['id'] == passenger_id)
        vehicle = next(v for v in vehicles if v['id'] == vehicle_id)
        folium.PolyLine(
            locations=[[passenger['latitude'], passenger['longitude']], [vehicle['latitude'], vehicle['longitude']]],
            color="green"
        ).add_to(m)

    # Sauvegarder la carte dans un fichier HTML
    m.save("result_map.html")

# 5 Exécution du script
if __name__ == "__main__":
    # Chemins vers les fichiers CSV
    passengers_file = "passengers.csv"
    vehicles_file = "vehicles.csv"

    # Clé API OpenRoute Service (remplacez par votre clé)
    api_key = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjlmN2RkZWRkNmQ2NjQ0MjJiYjhkMzIwNDcyYmMzOWMxIiwiaCI6Im11cm11cjY0In0="

    # Charger les données
    passengers, vehicles = load_data(passengers_file, vehicles_file)

    # Calculer la matrice des distances
    distance_matrix = calculate_distance_matrix(passengers, vehicles, api_key)

    # Résoudre le problème d'assignation
    assignments = solve_assignment_problem(distance_matrix, passengers, vehicles)

    # Afficher les résultats
    print("Assignations :", assignments)

    # Afficher les résultats sur une carte
    display_results_on_map(passengers, vehicles, assignments)