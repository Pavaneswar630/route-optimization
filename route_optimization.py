import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_traffic_data(num_records=1000):
    """Generates synthetic historical data for training the ML model."""
    np.random.seed(42)
    data = {
        'distance_km': np.random.uniform(1, 20, num_records),
        'hour_of_day': np.random.randint(0, 24, num_records),
        'weather_condition': np.random.randint(0, 3, num_records), # 0: Clear, 1: Rain, 2: Fog
        'traffic_density': np.random.uniform(0.1, 1.0, num_records)
    }
    df = pd.DataFrame(data)
    
    # Target variable: actual travel time in minutes
    # Base time (assuming 40km/h speed) + delays based on traffic and weather
    df['travel_time_min'] = (df['distance_km'] / 40.0) * 60 
    df['travel_time_min'] += (df['traffic_density'] * 15) + (df['weather_condition'] * 5)
    
    return df

def train_predictive_model(df):
    """Trains a Random Forest Regressor to predict travel times."""
    print("--- Training Machine Learning Model ---")
    X = df[['distance_km', 'hour_of_day', 'weather_condition', 'traffic_density']]
    y = df['travel_time_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Validation metrics
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Model Performance -> MAE: {mae:.2f} mins, RMSE: {rmse:.2f} mins\n")
    return model

def build_city_graph(model, current_hour, weather):
    """Builds a predictive route graph for a simulated city grid."""
    print("--- Constructing Predictive Network Graph ---")
    G = nx.Graph()
    
    # Define a simple grid of delivery nodes (0 to 5)
    edges = [
        (0, 1, 5.2), (0, 2, 3.1), (1, 3, 4.0), 
        (2, 3, 6.5), (2, 4, 2.0), (3, 5, 3.8), (4, 5, 7.1)
    ]
    
    for start_node, end_node, distance in edges:
        # Simulate current traffic for this edge
        current_traffic = np.random.uniform(0.2, 0.9) 
        
        # Prepare feature array for prediction
        features = pd.DataFrame({
            'distance_km': [distance],
            'hour_of_day': [current_hour],
            'weather_condition': [weather],
            'traffic_density': [current_traffic]
        })
        
        # Predict the travel time (edge weight)
        predicted_time = model.predict(features)[0]
        G.add_edge(start_node, end_node, weight=predicted_time, distance=distance)
        
    return G

def find_optimal_route(G, source, destination):
    """Finds the fastest route using Dijkstra's algorithm based on ML weights."""
    print(f"--- Optimizing Route from Node {source} to Node {destination} ---")
    try:
        optimal_path = nx.dijkstra_path(G, source=source, target=destination, weight='weight')
        total_time = nx.dijkstra_path_length(G, source=source, target=destination, weight='weight')
        
        print(f"Optimal Path Selected: {' -> '.join(map(str, optimal_path))}")
        print(f"Estimated Travel Time: {total_time:.2f} minutes")
    except nx.NetworkXNoPath:
        print("No valid path found between the specified nodes.")

if __name__ == "__main__":
    # 1. Prepare Data
    traffic_data = generate_synthetic_traffic_data(2000)
    
    # 2. Train Model
    rf_model = train_predictive_model(traffic_data)
    
    # 3. Simulate routing request (e.g., 5 PM rush hour, rainy weather)
    city_graph = build_city_graph(rf_model, current_hour=17, weather=1)
    
    # 4. Optimize
    find_optimal_route(city_graph, source=0, destination=5)
