# import pandas as pd
# import numpy as np
# import math
# import random
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # =============================================================================
# # STEP 1: LOAD AND INSPECT THE DATA
# # =============================================================================
# csv_file = 'tornado_severity_data.csv'
# df = pd.read_csv(csv_file)

# # Verify required columns exists
# required_cols = ['CAT Severity Code', 'ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']
# if not all(col in df.columns for col in required_cols):
#     raise ValueError("CSV file must contain 'CAT Severity Code', 'ACC_STD_LAT_NBR', and 'ACC_STD_LON_NBR' columns.")

# # Ensure each claim has a type: on_site or virtual (for simulation purposes)
# if 'CLAIM_TYPE' not in df.columns:
#     np.random.seed(42)  # for reproducibility
#     df['CLAIM_TYPE'] = np.random.choice(['on_site', 'virtual'], size=len(df), p=[0.5, 0.5])

# print("Data columns:", df.columns.tolist())
# print("First few rows:")
# print(df.head())

# # Make sure the severity column is integer typed.
# df['CAT Severity Code'] = df['CAT Severity Code'].astype(int)

# # =============================================================================
# # STEP 2: CLUSTER THE GEOGRAPHIC DATA USING THE ELBOW METHOD
# # =============================================================================
# # Use the provided geographic columns for clustering.
# coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]

# # Determine optimal k using the Elbow Method
# k_range = range(1, 11)  # Try between 1 and 10 clusters
# inertias = []
# for k in k_range:
#     kmeans_temp = KMeans(n_clusters=k, random_state=42)
#     kmeans_temp.fit(coords)
#     inertias.append(kmeans_temp.inertia_)

# # Optionally, plot the elbow curve for visual inspection.
# plt.figure(figsize=(8, 4))
# plt.plot(k_range, inertias, marker='o')
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method for Optimal k")
# plt.show()

# # Automatically select k based on relative improvement threshold.
# optimal_k = 1
# for i in range(1, len(inertias)):
#     improvement = (inertias[i-1] - inertias[i]) / inertias[i-1]
#     if improvement < 0.1:  # if improvement is less than 10%, select the previous k value.
#         optimal_k = i
#         break
# if optimal_k < 2:
#     optimal_k = 2

# print(f"Optimal number of clusters determined by the elbow method: {optimal_k}")
# n_clusters = optimal_k

# # Perform final clustering with the determined optimal number of clusters.
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# df['cluster'] = kmeans.fit_predict(coords)


# # =============================================================================
# # STEP 3: COMPUTE CLUSTER-SPECIFIC CLAIM VOLUMES
# # =============================================================================
# clusters = sorted(df['cluster'].unique())
# severity_levels = sorted(df['CAT Severity Code'].unique())

# # Build a nested dictionary: cluster_claim_volume[cluster][severity] = count
# cluster_claim_volume = {c: {} for c in clusters}
# for c in clusters:
#     for s in severity_levels:
#         count = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s)].shape[0]
#         cluster_claim_volume[c][s] = count

# print("Cluster-specific claim volumes:")
# print(cluster_claim_volume)

# # =============================================================================
# # STEP 3b: COMPUTE CLAIM VOLUMES BY HANDLING TYPE
# # =============================================================================
# cluster_claim_volume_on_site = {c: {} for c in clusters}
# cluster_claim_volume_virtual = {c: {} for c in clusters}

# for c in clusters:
#     for s in severity_levels:
#         cluster_claim_volume_on_site[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'on_site')].shape[0]
#         cluster_claim_volume_virtual[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'virtual')].shape[0]

# print('On-site claim volumes per cluster:', cluster_claim_volume_on_site)
# print('Virtual claim volumes per cluster:', cluster_claim_volume_virtual)

# # =============================================================================
# # STEP 3c: DEFINE DRIVE TIME IMPACTS FOR ON-SITE CLAIMS
# # (20% productivity loss per 30 minutes of drive time)
# if len(clusters) >= 5:
#     cluster_drive_time = {0: 30, 1: 20, 2: 40, 3: 30, 4: 15}  
# else:
#     cluster_drive_time = {c: 30 for c in clusters}  

# # =============================================================================
# # STEP 4: PARAMETERS AND DUMMY DATA (For Cost, Productivity, etc.)
# # =============================================================================
# # Target resolution days for 90% of claims (example values).
# target_days = {
#     1: 7,
#     2: 14,
#     3: 21,
#     4: 28,
#     5: 28
# }

# # Cost parameters: fixed cost X and additional cost per extra day.
# X = 1000           # fixed cost per handler
# daily_cost = 200   # cost for every additional day (n-1)

# # Productivity rates: handler of skill s on claims of severity j.
# productivity = {
#     (1, 1): 0.8, (1, 2): 0.7, (1, 3): 0.5, (1, 4): 0.3, (1, 5): 0.2,
#     (2, 1): 1.0, (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.4, (2, 5): 0.3,
#     (3, 1): 1.2, (3, 2): 1.1, (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.5,
#     (4, 1): 1.5, (4, 2): 1.3, (4, 3): 1.1, (4, 4): 0.8, (4, 5): 0.7,
#     (5, 1): 1.8, (5, 2): 1.6, (5, 3): 1.4, (5, 4): 1.2, (5, 5): 1.0,
# }

# # Penalty multiplier: large constant to enforce meeting resolution constraints.
# A = 10000

# # Maximum number of handlers allowed per (skill, severity, cluster) combination.
# # Using 4 bits to encode counts up to 15.
# max_count = 10
# num_bits = 4  # 2^4 = 16, allowing numbers 0-15.

# # =============================================================================
# # STEP 5: EXTEND THE BINARY VARIABLE MAPPING (Including Geographic Clustering and Dynamic Handling)
# # =============================================================================
# # Create binary variables for every combination (skill, severity, cluster, bit)
# binary_vars = {}   # maps global index to a dict with keys: s, j, cluster, bit, weight, prod
# # Create an index mapping for grouping by (cluster, severity)
# indices_by_cluster_severity = {(c, s): [] for c in clusters for s in severity_levels}

# var_idx = 0
# for s in range(1, 6):  # skill levels from 1 to 5
#     for j in severity_levels:  # claim severities (assumed 1 to 5)
#         for c in clusters:  # geographic clusters from KMeans
#             # Compute effective productivity multiplier for the (cluster, severity) group
#             total_claims = cluster_claim_volume_on_site[c][j] + cluster_claim_volume_virtual[c][j]
#             if total_claims > 0:
#                 # For on-site claims, reduce productivity by 20% per 30 minutes drive time
#                 drive_multiplier = 1 - 0.2 * (cluster_drive_time[c] / 30)
#                 # Weighted average multiplier based on claim type distribution
#                 multiplier = ((cluster_claim_volume_virtual[c][j] * 1.0) + (cluster_claim_volume_on_site[c][j] * drive_multiplier)) / total_claims
#             else:
#                 multiplier = 1.0
            
#             for b in range(num_bits):  # binary digits for count
#                 weight = 2 ** b
#                 # Adjust the base productivity using the computed multiplier
#                 effective_prod = productivity[(s, j)] * multiplier
#                 binary_vars[var_idx] = {
#                     's': s,
#                     'j': j,
#                     'cluster': c,
#                     'bit': b,
#                     'weight': weight,
#                     'prod': effective_prod
#                 }
#                 indices_by_cluster_severity[(c, j)].append(var_idx)
#                 var_idx += 1

# num_vars = var_idx
# print(f"Total binary variables (with geography): {num_vars}")

# # =============================================================================
# # STEP 6: BUILD THE EXTENDED QUBO MATRIX
# # =============================================================================
# # Store QUBO as a dictionary mapping (i,j) with i<=j to a real coefficient.
# Q = {}

# # Helper function: add a coefficient value to Q[(i,j)]
# def add_to_Q(i, j, value):
#     key = (i, j) if i <= j else (j, i)
#     Q[key] = Q.get(key, 0) + value

# # For each binary variable add linear cost and local penalty contributions.
# for i in range(num_vars):
#     info_i = binary_vars[i]
#     s = info_i['s']
#     j = info_i['j']
#     c = info_i['cluster']
#     b = info_i['bit']
#     weight = info_i['weight']
#     prod_rate = info_i['prod']

#     # Staffing cost for one handler: use the formula X + daily_cost*(target_days[j]-1)
#     cost_term = X + daily_cost * (target_days[j] - 1)
#     lin_cost = cost_term * weight

#     # Local penalty: enforce that staffing in cluster c meets the local claim volume.
#     local_claim = cluster_claim_volume[c][j]
#     # Expanded penalty: A * (local_claim - target_days[j] * N_j)^2,
#     # where N_j is the effective number of handlers = sum(weight * prod_rate * x)
#     pen_lin = -2 * A * local_claim * target_days[j] * (prod_rate * weight)
#     pen_self = A * (target_days[j] ** 2) * ((prod_rate * weight) ** 2)
#     add_to_Q(i, i, lin_cost + pen_lin + pen_self)

# # Add quadratic (pairwise) penalty terms for variables within the same (cluster, severity) group.
# for c in clusters:
#     for j in severity_levels:
#         group = indices_by_cluster_severity[(c, j)]
#         for idx1 in range(len(group)):
#             i = group[idx1]
#             info_i = binary_vars[i]
#             for idx2 in range(idx1 + 1, len(group)):
#                 k = group[idx2]
#                 info_k = binary_vars[k]
#                 coeff = A * (target_days[j] ** 2) * (info_i['prod'] * info_i['weight']) * (info_k['prod'] * info_k['weight'])
#                 add_to_Q(i, k, coeff)

# # =============================================================================
# # STEP 7: SOLVE THE QUBO USING SIMULATED ANNEALING
# # =============================================================================
# def qubo_energy(solution, Q):
#     """Compute the QUBO energy for a binary solution vector."""
#     energy = 0
#     for (i, j), coeff in Q.items():
#         energy += coeff * solution[i] * solution[j]
#     return energy

# def simulated_annealing(Q, num_vars, num_steps=5000, init_temp=10.0, cooling=0.995):
#     # Start with a random binary solution.
#     solution = [random.choice([0, 1]) for _ in range(num_vars)]
#     current_energy = qubo_energy(solution, Q)
#     best_solution = solution[:]
#     best_energy = current_energy
#     T = init_temp

#     for step in range(num_steps):
#         # Flip a random bit.
#         i = random.randint(0, num_vars - 1)
#         new_solution = solution[:]
#         new_solution[i] = 1 - new_solution[i]
#         new_energy = qubo_energy(new_solution, Q)
#         delta = new_energy - current_energy
#         # Accept new solution probabilistically.
#         if delta < 0 or random.random() < math.exp(-delta / T):
#             solution = new_solution
#             current_energy = new_energy
#             if current_energy < best_energy:
#                 best_solution = solution[:]
#                 best_energy = current_energy
#         T *= cooling

#     return best_solution, best_energy

# print("Solving the extended QUBO with simulated annealing...")
# solution, energy = simulated_annealing(Q, num_vars)
# print(f"Best energy found: {energy}")

# # =============================================================================
# # STEP 8: DECODE THE SOLUTION
# # =============================================================================
# # For each (skill, severity, cluster), decode the number of handlers from the binary solution.
# staffing_solution = {}  # key: (s, j, cluster) -> handler count
# for s in range(1, 6):
#     for j in severity_levels:
#         for c in clusters:
#             staffing_solution[(s, j, c)] = 0

# for i in range(num_vars):
#     info = binary_vars[i]
#     if solution[i] == 1:
#         key = (info['s'], info['j'], info['cluster'])
#         staffing_solution[key] += info['weight']

# # =============================================================================
# # FUNCTION: CALCULATE CLUSTER RESOLUTION PERCENTAGES
# # =============================================================================
# def calculate_cluster_resolution():
#     """Calculate the estimated resolution percentage per cluster based on the staffing solution and claim volumes."""
#     resolution = {}
#     for c in clusters:
#         total_capacity = 0.0
#         total_claims = 0
#         for j in severity_levels:
#             capacity = 0.0
#             for s in range(1, 6):
#                 # Get the number of handlers for skill s, severity j, in cluster c
#                 count = staffing_solution.get((s, j, c), 0)
#                 # Calculate capacity: handlers * base productivity * target resolution days
#                 capacity += count * productivity[(s, j)] * target_days[j]
#             claim_vol = cluster_claim_volume[c][j]
#             # Sum effective capacity, capping at the actual claim volume
#             total_capacity += min(capacity, claim_vol)
#             total_claims += claim_vol
#         # Avoid division by zero; if no claims then resolution is 100%
#         if total_claims > 0:
#             resolution[c] = total_capacity / total_claims
#         else:
#             resolution[c] = 1.0
#     return resolution

# # =============================================================================
# # STEP 9: PRINT THE FINAL OPTIMAL STAFFING SOLUTION
# # =============================================================================
# print("\nOptimal staffing solution by skill, claim severity, and geographic cluster:")
# for c in clusters:
#     print(f"\nCluster {c}:")
#     for j in severity_levels:
#         print(f"  Severity {j} claims (Local volume: {cluster_claim_volume[c][j]}, Target days: {target_days[j]}):")
#         for s in range(1, 6):
#             count = staffing_solution[(s, j, c)]
#             if count > 0:
#                 print(f"    Skill {s} handlers: {count}")

# cluster_costs = {}
# cluster_total_handlers = {}
# cluster_avg_time = {}
# for c in clusters:
#     total_cost_cluster = 0
#     total_time = 0
#     total_count = 0
#     for j in severity_levels:
#         cost_per_handler = X + daily_cost * (target_days[j] - 1)
#         count_j = sum(staffing_solution[(s, j, c)] for s in range(1, 6))
#         total_cost_cluster += count_j * cost_per_handler
#         total_time += count_j * target_days[j]
#         total_count += count_j
#     cluster_costs[c] = total_cost_cluster
#     cluster_total_handlers[c] = total_count
#     avg_time = total_time / total_count if total_count > 0 else 0
#     cluster_avg_time[c] = avg_time

# # Compute resolution percentages per cluster
# cluster_resolution = calculate_cluster_resolution()

# # Print the resolution percentages for each cluster
# for c, res in cluster_resolution.items():
#     print(f"Cluster {c} resolution: {res * 100:.2f}%")

# # Create a DataFrame to hold existing metrics and resolution
# df_metrics = pd.DataFrame({
#     'Cluster': list(cluster_costs.keys()),
#     'Total Cost': list(cluster_costs.values()),
#     'Average Resolution Days': [cluster_avg_time[c] for c in clusters],
#     'Resolution Percentage': [cluster_resolution[c] * 100 for c in clusters]
# })
# df_metrics.sort_values('Cluster', inplace=True)

# # Create subplots: 1 row, 3 columns
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# # Subplot 1: Bar chart of Total Staffing Cost per Cluster.
# ax1.bar(df_metrics['Cluster'].astype(str), df_metrics['Total Cost'], color='orange')
# ax1.set_xlabel('Cluster')
# ax1.set_ylabel('Total Staffing Cost')
# ax1.set_title('Total Staffing Cost per Cluster')

# # Subplot 2: Bar chart of Average Resolution Days per Cluster.
# ax2.bar(df_metrics['Cluster'].astype(str), df_metrics['Average Resolution Days'], color='skyblue')
# ax2.set_xlabel('Cluster')
# ax2.set_ylabel('Average Resolution Days')
# ax2.set_title('Average Resolution Days per Cluster')

# # Subplot 3: Bar chart of Resolution Percentage per Cluster.
# bars = ax3.bar(df_metrics['Cluster'].astype(str), df_metrics['Resolution Percentage'], color='green')
# ax3.set_xlabel('Cluster')
# ax3.set_ylabel('Resolution Percentage (%)')
# ax3.set_title('Estimated Resolution Percentage per Cluster')

# # Annotate each bar with the exact percentage number
# for bar in bars:
#     height = bar.get_height()
#     ax3.annotate(f'{height:.2f}%',
#                  xy=(bar.get_x() + bar.get_width() / 2, height),
#                  xytext=(0, 3),  # 3 points vertical offset
#                  textcoords='offset points',
#                  ha='center', va='bottom')

# plt.tight_layout()
# plt.show()

# # =============================================================================
# # DYNAMIC STAFFING: REAL-TIME ADJUSTMENTS
# # =============================================================================

# def update_claims(new_claims_df):
#     """Update the claim data with new claims and re-run the optimization."""
#     global df, clusters, cluster_claim_volume, cluster_claim_volume_on_site, cluster_claim_volume_virtual, binary_vars, indices_by_cluster_severity, Q, num_vars, staffing_solution
    
#     # Append new claims to the existing DataFrame
#     df = df.append(new_claims_df, ignore_index=True)
    
#     # Recompute clustering if geographic patterns have significantly changed
#     coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     df['cluster'] = kmeans.fit_predict(coords)
#     clusters = sorted(df['cluster'].unique())
    
#     # Recompute claim volumes
#     cluster_claim_volume = {c: {} for c in clusters}
#     cluster_claim_volume_on_site = {c: {} for c in clusters}
#     cluster_claim_volume_virtual = {c: {} for c in clusters}
#     for c in clusters:
#         for s in severity_levels:
#             cluster_claim_volume[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s)].shape[0]
#             cluster_claim_volume_on_site[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'on_site')].shape[0]
#             cluster_claim_volume_virtual[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'virtual')].shape[0]
    
#     # (Optional) Update drive times if new geographic data suggests different travel times
#     # For simplicity, we assume drive times remain constant. In a real implementation, these could be re-estimated.
    
#     # Reconstruct the QUBO based on updated data
#     # NOTE: For brevity, re-run the binary variable mapping and QUBO construction sections here as needed.
#     print('Claims updated and optimization re-run based on new data.')
    
#     # Here, one would re-run the optimization process (e.g., re-calling simulated_annealing) and update staffing_solution accordingly.



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

# Verify required columns exists
required_cols = ['CAT Severity Code', 'ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV file must contain 'CAT Severity Code', 'ACC_STD_LAT_NBR', and 'ACC_STD_LON_NBR' columns.")

# Ensure each claim has a type: on_site or virtual (for simulation purposes)
if 'CLAIM_TYPE' not in df.columns:
    np.random.seed(42)  # for reproducibility
    df['CLAIM_TYPE'] = np.random.choice(['on_site', 'virtual'], size=len(df), p=[0.5, 0.5])

print("Data columns:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Make sure the severity column is integer typed.
df['CAT Severity Code'] = df['CAT Severity Code'].astype(int)

# =============================================================================
# STEP 2: CLUSTER THE GEOGRAPHIC DATA USING THE ELBOW METHOD
# =============================================================================
# Use the provided geographic columns for clustering.
coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]

# Determine optimal k using the Elbow Method
k_range = range(1, 11)  # Try between 1 and 10 clusters
inertias = []
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(coords)
    inertias.append(kmeans_temp.inertia_)

# Optionally, plot the elbow curve for visual inspection.
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Automatically select k based on relative improvement threshold.
optimal_k = 1
for i in range(1, len(inertias)):
    improvement = (inertias[i-1] - inertias[i]) / inertias[i-1]
    if improvement < 0.1:  # if improvement is less than 10%, select the previous k value.
        optimal_k = i
        break
if optimal_k < 2:
    optimal_k = 2

print(f"Optimal number of clusters determined by the elbow method: {optimal_k}")
n_clusters = optimal_k

# Perform final clustering with the determined optimal number of clusters.
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(coords)

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
# STEP 3b: COMPUTE CLAIM VOLUMES BY HANDLING TYPE
# =============================================================================
cluster_claim_volume_on_site = {c: {} for c in clusters}
cluster_claim_volume_virtual = {c: {} for c in clusters}

for c in clusters:
    for s in severity_levels:
        cluster_claim_volume_on_site[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'on_site')].shape[0]
        cluster_claim_volume_virtual[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'virtual')].shape[0]

print('On-site claim volumes per cluster:', cluster_claim_volume_on_site)
print('Virtual claim volumes per cluster:', cluster_claim_volume_virtual)

# =============================================================================
# STEP 3c: DEFINE DRIVE TIME IMPACTS FOR ON-SITE CLAIMS
# (20% productivity loss per 30 minutes of drive time)
if len(clusters) >= 5:
    cluster_drive_time = {0: 30, 1: 20, 2: 40, 3: 30, 4: 15}  
else:
    cluster_drive_time = {c: 30 for c in clusters}  

# =============================================================================
# STEP 4: PARAMETERS AND DUMMY DATA (For Cost, Productivity, etc.)
# =============================================================================
# Target resolution days for 90% of claims (example values).
target_days = {
    1: 7,
    2: 14,
    3: 21,
    4: 28,
    5: 28
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
# STEP 5: EXTEND THE BINARY VARIABLE MAPPING (Including Geographic Clustering and Dynamic Handling)
# =============================================================================
# Create binary variables for every combination (skill, severity, cluster, bit)
binary_vars = {}   # maps global index to a dict with keys: s, j, cluster, bit, weight, prod
# Create an index mapping for grouping by (cluster, severity)
indices_by_cluster_severity = {(c, s): [] for c in clusters for s in severity_levels}

var_idx = 0
for s in range(1, 6):  # skill levels from 1 to 5
    for j in severity_levels:  # claim severities (assumed 1 to 5)
        for c in clusters:  # geographic clusters from KMeans
            # Compute effective productivity multiplier for the (cluster, severity) group
            total_claims = cluster_claim_volume_on_site[c][j] + cluster_claim_volume_virtual[c][j]
            if total_claims > 0:
                # For on-site claims, reduce productivity by 20% per 30 minutes drive time
                drive_multiplier = 1 - 0.2 * (cluster_drive_time[c] / 30)
                # Weighted average multiplier based on claim type distribution
                multiplier = ((cluster_claim_volume_virtual[c][j] * 1.0) + (cluster_claim_volume_on_site[c][j] * drive_multiplier)) / total_claims
            else:
                multiplier = 1.0
            
            for b in range(num_bits):  # binary digits for count
                weight = 2 ** b
                # Adjust the base productivity using the computed multiplier
                effective_prod = productivity[(s, j)] * multiplier
                binary_vars[var_idx] = {
                    's': s,
                    'j': j,
                    'cluster': c,
                    'bit': b,
                    'weight': weight,
                    'prod': effective_prod
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
    # Expanded penalty: A * (local_claim - target_days[j] * N_j)^2,
    # where N_j is the effective number of handlers = sum(weight * prod_rate * x)
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
# FUNCTION: CALCULATE CLUSTER RESOLUTION PERCENTAGES
# =============================================================================
def calculate_cluster_resolution():
    """Calculate the estimated resolution percentage per cluster based on the staffing solution and claim volumes."""
    resolution = {}
    for c in clusters:
        total_capacity = 0.0
        total_claims = 0
        for j in severity_levels:
            capacity = 0.0
            for s in range(1, 6):
                # Get the number of handlers for skill s, severity j, in cluster c
                count = staffing_solution.get((s, j, c), 0)
                # Calculate capacity: handlers * base productivity * target resolution days
                capacity += count * productivity[(s, j)] * target_days[j]
            claim_vol = cluster_claim_volume[c][j]
            # Sum effective capacity, capping at the actual claim volume
            total_capacity += min(capacity, claim_vol)
            total_claims += claim_vol
        # Avoid division by zero; if no claims then resolution is 100%
        if total_claims > 0:
            resolution[c] = total_capacity / total_claims
        else:
            resolution[c] = 1.0
    return resolution

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

cluster_costs = {}
cluster_total_handlers = {}
cluster_avg_time = {}
for c in clusters:
    total_cost_cluster = 0
    total_time = 0
    total_count = 0
    for j in severity_levels:
        cost_per_handler = X + daily_cost * (target_days[j] - 1)
        count_j = sum(staffing_solution[(s, j, c)] for s in range(1, 6))
        total_cost_cluster += count_j * cost_per_handler
        total_time += count_j * target_days[j]
        total_count += count_j
    cluster_costs[c] = total_cost_cluster
    cluster_total_handlers[c] = total_count
    avg_time = total_time / total_count if total_count > 0 else 0
    cluster_avg_time[c] = avg_time

# Compute resolution percentages per cluster
cluster_resolution = calculate_cluster_resolution()

# Print the resolution percentages for each cluster
for c, res in cluster_resolution.items():
    print(f"Cluster {c} resolution: {res * 100:.2f}%")

# ------------------------------
# Additional Metrics Calculation:
# ------------------------------
cluster_on_site_percentage = {}
cluster_virtual_percentage = {}
cluster_total_claims = {}
cluster_target_met = {}

for c in clusters:
    on_site_total = sum(cluster_claim_volume_on_site[c].values())
    virtual_total = sum(cluster_claim_volume_virtual[c].values())
    total_claims = sum(cluster_claim_volume[c].values())
    cluster_total_claims[c] = total_claims
    if total_claims > 0:
        on_site_pct = (on_site_total / total_claims) * 100
        virtual_pct = (virtual_total / total_claims) * 100
    else:
        on_site_pct = 0
        virtual_pct = 0
    cluster_on_site_percentage[c] = on_site_pct
    cluster_virtual_percentage[c] = virtual_pct
    # Mark if the resolution target of 90% is met.
    cluster_target_met[c] = 'Yes' if cluster_resolution[c] >= 0.9 else 'No'

# Create a DataFrame to hold all metrics.
df_metrics = pd.DataFrame({
    'Cluster': list(cluster_costs.keys()),
    'Total Cost': list(cluster_costs.values()),
    'Average Resolution Days': [cluster_avg_time[c] for c in clusters],
    'Resolution Percentage': [cluster_resolution[c] * 100 for c in clusters],
    'On-site %': [cluster_on_site_percentage[c] for c in clusters],
    'Virtual %': [cluster_virtual_percentage[c] for c in clusters],
    'Drive Time (min)': [cluster_drive_time[c] for c in clusters],
    'Target Met': [cluster_target_met[c] for c in clusters]
})
df_metrics.sort_values('Cluster', inplace=True)

print("\nCluster Metrics:")
print(df_metrics)

# =============================================================================
# PLOTTING: 2x2 Subplots for Key Metrics
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Subplot 1 (Top Left): Total Staffing Cost per Cluster
ax1 = axes[0, 0]
ax1.bar(df_metrics['Cluster'].astype(str), df_metrics['Total Cost'], color='orange')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Total Staffing Cost')
ax1.set_title('Total Staffing Cost per Cluster')

# Subplot 2 (Top Right): Average Resolution Days per Cluster
ax2 = axes[0, 1]
ax2.bar(df_metrics['Cluster'].astype(str), df_metrics['Average Resolution Days'], color='skyblue')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Average Resolution Days')
ax2.set_title('Average Resolution Days per Cluster')

# Subplot 3 (Bottom Left): Resolution Percentage and Target Flag per Cluster
ax3 = axes[1, 0]
bars = ax3.bar(df_metrics['Cluster'].astype(str), df_metrics['Resolution Percentage'], color='green')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Resolution Percentage (%)')
ax3.set_title('Estimated Resolution Percentage per Cluster')
# Annotate each bar with the exact percentage and whether target is met.
for idx, bar in enumerate(bars):
    height = bar.get_height()
    target_flag = df_metrics['Target Met'].iloc[idx]
    ax3.annotate(f'{height:.2f}%\n(Target: {target_flag})',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # vertical offset in points
                 textcoords='offset points',
                 ha='center', va='bottom')

# Subplot 4 (Bottom Right): On-site vs. Virtual Claim Split per Cluster
ax4 = axes[1, 1]
width = 0.35  # width of the bars
indices = np.arange(len(df_metrics))
onsite_bars = ax4.bar(indices - width/2, df_metrics['On-site %'], width, color='dodgerblue', label='On-site %')
virtual_bars = ax4.bar(indices + width/2, df_metrics['Virtual %'], width, color='lightcoral', label='Virtual %')
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('On-site vs. Virtual Claim Handling per Cluster')
ax4.set_xticks(indices)
ax4.set_xticklabels(df_metrics['Cluster'].astype(str))
ax4.legend()

# Annotate each cluster with the drive time.
for idx, dt in enumerate(df_metrics['Drive Time (min)']):
    # Display drive time above the higher of on-site or virtual percentage for each cluster.
    max_pct = max(df_metrics['On-site %'].iloc[idx], df_metrics['Virtual %'].iloc[idx])
    ax4.text(idx, max_pct + 2,
             f'DT: {dt}m', ha='center', va='bottom', fontsize=9, color='gray')

plt.tight_layout()
plt.show()

# =============================================================================
# DYNAMIC STAFFING: REAL-TIME ADJUSTMENTS
# =============================================================================

def update_claims(new_claims_df):
    """Update the claim data with new claims and re-run the optimization."""
    global df, clusters, cluster_claim_volume, cluster_claim_volume_on_site, cluster_claim_volume_virtual, binary_vars, indices_by_cluster_severity, Q, num_vars, staffing_solution
    
    # Append new claims to the existing DataFrame
    df = df.append(new_claims_df, ignore_index=True)
    
    # Recompute clustering if geographic patterns have significantly changed
    coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    clusters = sorted(df['cluster'].unique())
    
    # Recompute claim volumes
    cluster_claim_volume = {c: {} for c in clusters}
    cluster_claim_volume_on_site = {c: {} for c in clusters}
    cluster_claim_volume_virtual = {c: {} for c in clusters}
    for c in clusters:
        for s in severity_levels:
            cluster_claim_volume[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s)].shape[0]
            cluster_claim_volume_on_site[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'on_site')].shape[0]
            cluster_claim_volume_virtual[c][s] = df[(df['cluster'] == c) & (df['CAT Severity Code'] == s) & (df['CLAIM_TYPE'] == 'virtual')].shape[0]
    
    # (Optional) Update drive times if new geographic data suggests different travel times
    # For simplicity, we assume drive times remain constant. In a real implementation, these could be re-estimated.
    
    # Reconstruct the QUBO based on updated data
    # NOTE: For brevity, re-run the binary variable mapping and QUBO construction sections here as needed.
    print('Claims updated and optimization re-run based on new data.')
    
    # Here, one would re-run the optimization process (e.g., re-calling simulated_annealing) and update staffing_solution accordingly.