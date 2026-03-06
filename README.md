# Route Optimization Using Machine Learning Algorithms

## Overview
This repository contains a predictive modeling approach to route optimization. The objective of the project is to minimize travel costs, time, and distance by integrating machine learning techniques into traditional graph-based pathfinding algorithms. 

Instead of relying on static distance metrics, this system dynamically calculates edge weights (travel times) using a **Random Forest Regressor** trained on historical traffic, weather, and temporal data.

## Methodology
1. **Data Preprocessing:** Handled high-dimensional simulated traffic datasets (features include weather conditions, temporal data, and traffic density).
2. **Predictive Modeling:** Trained an ML regression model to predict Point-A to Point-B latency.
3. **Graph Construction:** Utilized `NetworkX` to build a dynamic city node structure.
4. **Algorithmic Optimization:** Applied Dijkstra's algorithm against the ML-generated edge weights to calculate the mathematically optimal route.

## Performance Metrics
- **Mean Absolute Error (MAE):** Validated to ensure minimal deviation in real-world time estimation.
- **Root Mean Squared Error (RMSE):** Used to penalize large prediction variances in high-traffic scenarios.

## Tech Stack
- **Languages:** Python
- **Libraries:** Scikit-Learn (Machine Learning), Pandas & NumPy (Data Manipulation), NetworkX (Graph Theory & Optimization)

## How to Run
```bash
# Clone the repository
git clone [https://github.com/Pavaneswar630/Route-Optimization-ML.git](https://github.com/Pavaneswar630/Route-Optimization-ML.git)

# Install dependencies
pip install pandas numpy scikit-learn networkx

# Execute the predictive model
python route_optimization.py
