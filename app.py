from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

app = Flask(__name__,template_folder='')

def optimize_weights(v, A):
    n = len(v)
    
    # Objective function to minimize the sum of elements in w_1 and w_2
    def objective(w):
        w_1 = w[:n]
        w_2 = w[n:]
        return np.sum(w_1) + np.sum(w_2)

    # Nonlinear constraint function
    def constraint_func(w):
        w_1 = w[:n]
        w_2 = w[n:]
        return v + w_1 + w_2 + 0.5 * A @ w_2

    # Define the constraints using NonlinearConstraint
    nonlinear_constraint = NonlinearConstraint(constraint_func, np.full(n, 2), np.full(n, 3))

    # Bounds for weights (non-negative)
    bounds = [(0, 3) for _ in range(2 * n)]

    # Initial guess for weights
    initial_guess = np.ones(2 * n)

    # Minimize the objective function using SLSQP method
    result = minimize(objective, initial_guess, method='SLSQP', constraints=[nonlinear_constraint], bounds=bounds)

    if result.success:
        w = result.x
        w_1 = w[:n]
        w_2 = w[n:]
        return w_1, w_2
    else:
        raise ValueError("No suitable weight distribution found")

def generate_adjacency_matrix(n, edges):
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    v = np.array(data['v'])
    edges_input = data.get('edges')
    print(v)
    print(len(v))
    print(edges_input)
    weight = data['weight']

    # Convert edges_input to a list of tuples
    edges = []
    for edge in edges_input:
        if len(edge) == 2:
            try:
                u_1, u_2 = int(edge[0]), int(edge[1])
                edges.append((u_1, u_2))
            except ValueError:
                continue  # skip invalid edges

    default_edges = [(i, i+1) for i in range(len(v)-1)]
    edges = edges + default_edges
    A = generate_adjacency_matrix(len(v), edges)

    try:
        w_1, w_2 = optimize_weights(v, A)
        np.set_printoptions(precision=2)
        result = {
            "w_1": np.round(w_1, 2).tolist(),
            "w_2": np.round(w_2, 2).tolist(),
            "weight_sum": np.round((np.sum(w_1) + np.sum(w_2)) * weight, 2),
            "resulting_vector": np.round(v + w_1 + w_2 + 0.5 * A @ w_2, 2).tolist()
        }
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
