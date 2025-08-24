import os
import pandas as pd
import requests
from ortools.linear_solver import pywraplp
import folium
import numpy as np
from typing import List, Dict, Tuple

"""
Script d'assignation Passagers → Véhicules
- Lit deux CSV: passengers.csv et vehicles.csv (colonnes requises: id, latitude, longitude)
- Calcule une matrice de distances (km) via l'API ORS Matrix hébergée sur Deploily.cloud
- Résout un problème d'assignation (chaque passager est affecté à exactement un véhicule)
- Affiche les résultats sur une carte Folium (result_map.html)

Docs Deploily (ORS Matrix): https://hub.deploily.cloud/blog/deploily-apis-3/boost-your-logistics-efficiency-using-ors-matrix-time-distance-solver-18
"""

# 1) Récupérer les données depuis deux fichiers CSV (passsengers et vehicles)

def load_data(passengers_file: str, vehicles_file: str) -> Tuple[List[Dict], List[Dict]]:
    passengers_df = pd.read_csv(passengers_file)
    vehicles_df = pd.read_csv(vehicles_file)

    required_cols = {"id", "latitude", "longitude"}
    for name, df in (("passengers", passengers_df), ("vehicles", vehicles_df)):
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Le CSV '{name}' doit contenir les colonnes: {sorted(required_cols)}. Manquantes: {sorted(missing)}")

    return passengers_df.to_dict(orient='records'), vehicles_df.to_dict(orient='records')


# 2) Calculer les distances avec Deploily (ORS Matrix)

def calculate_distance_matrix(passengers: List[Dict], vehicles: List[Dict], api_key: str) -> np.ndarray:
    if not api_key:
        raise ValueError("Clé API manquante. Définissez DEPLOILY_API_KEY.")

    # Url API Deploily 
    base_url = "https://api.deploily.cloud/ors/v2/matrix/driving-car"

    # Construire la liste des lieux: d'abord tous les passagers, puis tous les véhicules
    # IMPORTANT: ORS attend les coordonnées au format [lon, lat] et pas [lat, lon]
    passenger_coords = [[float(p['longitude']), float(p['latitude'])] for p in passengers]
    vehicle_coords = [[float(v['longitude']), float(v['latitude'])] for v in vehicles]
    locations = passenger_coords + vehicle_coords

    n_pass = len(passengers)
    sources = list(range(n_pass))
    destinations = list(range(n_pass, n_pass + len(vehicles)))

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Deploily attend l'en-tête 'apikey' 
        "apikey": api_key,
    }

    payload = {
        "locations": locations,
        "metrics": ["distance"],  # en (km)
        "units": "km",
        "sources": sources,
        "destinations": destinations,
    }

    resp = requests.post(base_url, json=payload, headers=headers, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Aider au debug avec un message clair
        raise RuntimeError(f"Erreur API Deploily/ORS ({resp.status_code}): {resp.text}") from e

    data = resp.json()
    if 'distances' not in data:
        raise KeyError("La réponse ne contient pas le champ 'distances'. Réponse: " + str(data)[:500])

    matrix = np.array(data['distances'], dtype=float)
    # La matrice renvoyée a la forme [len(sources) x len(destinations)] = [passagers x véhicules]
    return matrix


# 3) Modéliser et résoudre le problème d'assignation avec OR-Tools

def solve_assignment_problem(distance_matrix: np.ndarray, passengers: List[Dict], vehicles: List[Dict]) -> List[Tuple]:
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Solver OR-Tools 'SCIP' non disponible.")

    num_passengers = len(passengers)
    num_vehicles = len(vehicles)

    # Variables de décision : x[i,j] = 1 si le passager i est assigné au véhicule j
    x = {(i, j): solver.IntVar(0, 1, f"x[{i},{j}]") for i in range(num_passengers) for j in range(num_vehicles)}

    # Contrainte: chaque passager est assigné à exactement un véhicule
    for i in range(num_passengers):
        solver.Add(solver.Sum(x[i, j] for j in range(num_vehicles)) == 1)

        
    objective_terms = []
    for i in range(num_passengers):
        for j in range(num_vehicles):
            objective_terms.append(float(distance_matrix[i, j]) * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Aucune solution trouvée par OR-Tools.")

    assignments = []
    for i in range(num_passengers):
        for j in range(num_vehicles):
            if x[i, j].solution_value() > 0.5:
                assignments.append((passengers[i]['id'], vehicles[j]['id']))
    return assignments


# 4) Afficher les résultats sur une carte

def display_results_on_map(passengers: List[Dict], vehicles: List[Dict], assignments: List[Tuple], outfile: str = "result_map.html") -> None:
    # Centrer sur la moyenne des points si dispo, sinon fallback Ain Témouchent
    if passengers or vehicles:
        all_lat = [p['latitude'] for p in passengers] + [v['latitude'] for v in vehicles]
        all_lon = [p['longitude'] for p in passengers] + [v['longitude'] for v in vehicles]
        center = [float(np.mean(all_lat)), float(np.mean(all_lon))]
    else:
        center = [35.3055, -1.1402]

    m = folium.Map(location=center, zoom_start=12)

    for passenger in passengers:
        folium.Marker(
            location=[passenger['latitude'], passenger['longitude']],
            popup=f"Passenger {passenger['id']}",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    for vehicle in vehicles:
        folium.Marker(
            location=[vehicle['latitude'], vehicle['longitude']],
            popup=f"Vehicle {vehicle['id']}",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    for passenger_id, vehicle_id in assignments:
        passenger = next(p for p in passengers if p['id'] == passenger_id)
        vehicle = next(v for v in vehicles if v['id'] == vehicle_id)
        folium.PolyLine(
            locations=[[passenger['latitude'], passenger['longitude']], [vehicle['latitude'], vehicle['longitude']]],
            color="green",
        ).add_to(m)

    m.save(outfile)


# 5) Exécution du script
if __name__ == "__main__":
    # Chemins vers les fichiers CSV
    passengers_file = "passengers.csv"
    vehicles_file = "vehicles.csv"

    # Clé API Deploily (ne pas commiter votre clé !)
    api_key = "4397b1d58eaf470cb190835aa0131305"

    # Charger les données
    passengers, vehicles = load_data(passengers_file, vehicles_file)

    # Calculer la matrice des distances (km) via Deploily
    distance_matrix = calculate_distance_matrix(passengers, vehicles, api_key)

    # Résoudre le problème d'assignation
    assignments = solve_assignment_problem(distance_matrix, passengers, vehicles)

    print("Assignations :", assignments)

    # Afficher les résultats sur une carte
    display_results_on_map(passengers, vehicles, assignments)
    print("Carte enregistrée → result_map.html")
