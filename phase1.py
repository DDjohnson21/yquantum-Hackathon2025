

import pandas as pd
import numpy as np
import math
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: LOAD AND INSPECT THE DATA
# =============================================================================
csv_file = 'tornado_severity_data.csv'
df = pd.read_csv(csv_file)

# Verify required columns exist
required_cols = ['CAT Severity Code', 'ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV file must contain 'CAT Severity Code', 'ACC_STD_LAT_NBR', and 'ACC_STD_LON_NBR' columns.")

print("Data columns:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Make sure the severity column is integer typed.
df['CAT Severity Code'] = df['CAT Severity Code'].astype(int)

# =============================================================================
# STEP 2: CLUSTER THE GEOGRAPHIC DATA
# =============================================================================
# Use geographic coordinates for clustering.
coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]
n_clusters = 5  # You can adjust this number as needed.
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(coords)

# Optional: Visualize the clustering result.
plt.figure(figsize=(8, 6))
plt.scatter(df['ACC_STD_LON_NBR'], df['ACC_STD_LAT_NBR'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Clustering of Claims")
plt.show()

# =============================================================================
# STEP 3: COMPUTE CLUSTER-SPECIFIC CLAIM VOLUMES
# =============================================================================
clusters = sorted(df['cluster'].unique())
severity_levels = sorted(df['CAT Severity Code'].unique())

# Build a nested dictionary: cluster_claim_volume[cluster][severity] = count
cluster_claim_volume = {c: {} for c in clusters}
for c in clusters:
    for s in severity_levels:
        count = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s)].shape[0]
        cluster_claim_volume[c][s] = count

print("Cluster-specific claim volumes:")
print(cluster_claim_volume)

# =============================================================================
# STEP 4: PARAMETERS AND DUMMY DATA (For Cost, Productivity, etc.)
# =============================================================================
# Target resolution days for 90% of claims (example values).
target_days = {
    1: 5,
    2: 7,
    3: 10,
    4: 12,
    5: 15
}

# Cost parameters: fixed cost X and additional cost per extra day.
X = 1000           # fixed cost per handler
daily_cost = 200   # cost for every additional day (n-1)

# Productivity rates: handler of skill s on claims of severity j.
productivity = {
    (1, 1): 0.8, (1, 2): 0.7, (1, 3): 0.5, (1, 4): 0.3, (1, 5): 0.2,
    (2, 1): 1.0, (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.4, (2, 5): 0.3,
    (3, 1): 1.2, (3, 2): 1.1, (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.5,
    (4, 1): 1.5, (4, 2): 1.3, (4, 3): 1.1, (4, 4): 0.8, (4, 5): 0.7,
    (5, 1): 1.8, (5, 2): 1.6, (5, 3): 1.4, (5, 4): 1.2, (5, 5): 1.0,
}

# Penalty multiplier: large constant to enforce meeting resolution constraints.
A = 10000

# Maximum number of handlers allowed per (skill, severity, cluster) combination.
# Using 4 bits to encode counts up to 15.
max_count = 10
num_bits = 4  # 2^4 = 16, allowing numbers 0-15.

# =============================================================================
# STEP 5: EXTEND THE BINARY VARIABLE MAPPING (Including Geographic Clustering)
# =============================================================================
# Create binary variables for every combination (skill, severity, cluster, bit)
binary_vars = {}   # maps global index to a dict with keys: s, j, cluster, bit, weight, prod
# Create an index mapping for grouping by (cluster, severity)
indices_by_cluster_severity = {(c, s): [] for c in clusters for s in severity_levels}

var_idx = 0
for s in range(1, 6):  # skill levels from 1 to 5
    for j in severity_levels:  # claim severities (assumed 1 to 5)
        for c in clusters:  # geographic clusters from KMeans
            for b in range(num_bits):  # binary digits for count
                weight = 2 ** b
                binary_vars[var_idx] = {
                    's': s,
                    'j': j,
                    'cluster': c,
                    'bit': b,
                    'weight': weight,
                    'prod': productivity[(s, j)]
                }
                indices_by_cluster_severity[(c, j)].append(var_idx)
                var_idx += 1

num_vars = var_idx
print(f"Total binary variables (with geography): {num_vars}")

# =============================================================================
# STEP 6: BUILD THE EXTENDED QUBO MATRIX
# =============================================================================
# Store QUBO as a dictionary mapping (i,j) with i<=j to a real coefficient.
Q = {}

# Helper function: add a coefficient value to Q[(i,j)]
def add_to_Q(i, j, value):
    key = (i, j) if i <= j else (j, i)
    Q[key] = Q.get(key, 0) + value

# For each binary variable add linear cost and local penalty contributions.
for i in range(num_vars):
    info_i = binary_vars[i]
    s = info_i['s']
    j = info_i['j']
    c = info_i['cluster']
    b = info_i['bit']
    weight = info_i['weight']
    prod_rate = info_i['prod']

    # Staffing cost for one handler: use the formula X + daily_cost*(target_days[j]-1)
    cost_term = X + daily_cost * (target_days[j] - 1)
    lin_cost = cost_term * weight

    # Local penalty: enforce that staffing in cluster c meets the local claim volume.
    local_claim = cluster_claim_volume[c][j]
    # Expand the squared penalty: A * (local_claim - target_days[j] * N_j)^2, where 
    # N_j is the effective number of handlers = sum(weight * prod_rate * x)
    pen_lin = -2 * A * local_claim * target_days[j] * (prod_rate * weight)
    pen_self = A * (target_days[j] ** 2) * ((prod_rate * weight) ** 2)
    add_to_Q(i, i, lin_cost + pen_lin + pen_self)

# Add quadratic (pairwise) penalty terms for variables within the same (cluster, severity) group.
for c in clusters:
    for j in severity_levels:
        group = indices_by_cluster_severity[(c, j)]
        for idx1 in range(len(group)):
            i = group[idx1]
            info_i = binary_vars[i]
            for idx2 in range(idx1 + 1, len(group)):
                k = group[idx2]
                info_k = binary_vars[k]
                coeff = A * (target_days[j] ** 2) * (info_i['prod'] * info_i['weight']) * (info_k['prod'] * info_k['weight'])
                add_to_Q(i, k, coeff)

# =============================================================================
# STEP 7: SOLVE THE QUBO USING SIMULATED ANNEALING
# =============================================================================
def qubo_energy(solution, Q):
    """Compute the QUBO energy for a binary solution vector."""
    energy = 0
    for (i, j), coeff in Q.items():
        energy += coeff * solution[i] * solution[j]
    return energy

def simulated_annealing(Q, num_vars, num_steps=5000, init_temp=10.0, cooling=0.995):
    # Start with a random binary solution.
    solution = [random.choice([0, 1]) for _ in range(num_vars)]
    current_energy = qubo_energy(solution, Q)
    best_solution = solution[:]
    best_energy = current_energy
    T = init_temp

    for step in range(num_steps):
        # Flip a random bit.
        i = random.randint(0, num_vars - 1)
        new_solution = solution[:]
        new_solution[i] = 1 - new_solution[i]
        new_energy = qubo_energy(new_solution, Q)
        delta = new_energy - current_energy
        # Accept new solution probabilistically.
        if delta < 0 or random.random() < math.exp(-delta / T):
            solution = new_solution
            current_energy = new_energy
            if current_energy < best_energy:
                best_solution = solution[:]
                best_energy = current_energy
        T *= cooling

    return best_solution, best_energy

print("Solving the extended QUBO with simulated annealing...")
solution, energy = simulated_annealing(Q, num_vars)
print(f"Best energy found: {energy}")

# =============================================================================
# STEP 8: DECODE THE SOLUTION
# =============================================================================
# For each (skill, severity, cluster), decode the number of handlers from the binary solution.
staffing_solution = {}  # key: (s, j, cluster) -> handler count
for s in range(1, 6):
    for j in severity_levels:
        for c in clusters:
            staffing_solution[(s, j, c)] = 0

for i in range(num_vars):
    info = binary_vars[i]
    if solution[i] == 1:
        key = (info['s'], info['j'], info['cluster'])
        staffing_solution[key] += info['weight']

# =============================================================================
# STEP 9: PRINT THE FINAL OPTIMAL STAFFING SOLUTION
# =============================================================================
print("\nOptimal staffing solution by skill, claim severity, and geographic cluster:")
for c in clusters:
    print(f"\nCluster {c}:")
    for j in severity_levels:
        print(f"  Severity {j} claims (Local volume: {cluster_claim_volume[c][j]}, Target days: {target_days[j]}):")
        for s in range(1, 6):
            count = staffing_solution[(s, j, c)]
            if count > 0:
                print(f"    Skill {s} handlers: {count}")