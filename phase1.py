
import pandas as pd
import numpy as np
import math
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
# STEP 2: DOMAIN-DRIVEN APPROACH FOR CLUSTERING
# =============================================================================
coords = df[['ACC_STD_LAT_NBR', 'ACC_STD_LON_NBR']]

# We'll try k from 2..10 (or fewer/more depending on your domain knowledge)
k_range = range(2, 11)

# This alpha determines how much we penalize cluster diameters.
alpha = 50.0

penalized_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels_temp = kmeans_temp.fit_predict(coords)
    inertia_temp = kmeans_temp.inertia_

    # Compute sum of diameters for each cluster
    sum_of_diameters = 0.0
    for cluster_id in range(k):
        # Extract points that belong to this cluster
        cluster_points = coords[labels_temp == cluster_id].values
        # Determine the cluster's centroid
        cluster_center = kmeans_temp.cluster_centers_[cluster_id]
        # Compute max distance from center
        max_dist = 0.0
        for p in cluster_points:
            dist = np.linalg.norm(p - cluster_center)
            if dist > max_dist:
                max_dist = dist
        # Add to the running total of diameters
        sum_of_diameters += max_dist

    # Combine the inertia with the diameter penalty
    penalized_score = inertia_temp + alpha * sum_of_diameters
    penalized_scores.append((k, penalized_score))

# Choose the k that yields the smallest penalized score
optimal_k, best_score = min(penalized_scores, key=lambda x: x[1])
print(f"Penalized cluster selection chose k={optimal_k} with a combined score of {best_score:.2f}")

# Perform the final clustering with the chosen k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(coords)
df['cluster'] = labels

# =============================================================================
# STEP 3: COMPUTE CLUSTER-SPECIFIC CLAIM VOLUMES
# =============================================================================
clusters = sorted(df['cluster'].unique())
severity_levels = sorted(df['CAT Severity Code'].unique())

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
        cluster_claim_volume_on_site[c][s] = df[
            (df['cluster'] == c) & 
            (df['CAT Severity Code'] == s) & 
            (df['CLAIM_TYPE'] == 'on_site')
        ].shape[0]
        
        cluster_claim_volume_virtual[c][s] = df[
            (df['cluster'] == c) & 
            (df['CAT Severity Code'] == s) & 
            (df['CLAIM_TYPE'] == 'virtual')
        ].shape[0]

print('On-site claim volumes per cluster:', cluster_claim_volume_on_site)
print('Virtual claim volumes per cluster:', cluster_claim_volume_virtual)

# For simplicity, assign 30 minutes drive time to every cluster, or adjust as desired.
cluster_drive_time = {c: 30 for c in clusters}

# =============================================================================
# STEP 4: PARAMETERS AND DUMMY DATA (For Cost, Productivity, etc.)
# =============================================================================
target_days = {
    1: 7,
    2: 14,
    3: 21,
    4: 28,
    5: 28
}

X = 1000           # fixed cost per handler
daily_cost = 200   # cost for every additional day (n-1)

productivity = {
    (1, 1): 0.8, (1, 2): 0.7, (1, 3): 0.5, (1, 4): 0.3, (1, 5): 0.2,
    (2, 1): 1.0, (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.4, (2, 5): 0.3,
    (3, 1): 1.2, (3, 2): 1.1, (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.5,
    (4, 1): 1.5, (4, 2): 1.3, (4, 3): 1.1, (4, 4): 0.8, (4, 5): 0.7,
    (5, 1): 1.8, (5, 2): 1.6, (5, 3): 1.4, (5, 4): 1.2, (5, 5): 1.0,
}

A = 10000  # large penalty multiplier for under-capacity
max_count = 10
num_bits = 4  # each count is represented with 4 bits (0–15)

# =============================================================================
# STEP 5: EXTEND THE BINARY VARIABLE MAPPING
# =============================================================================
binary_vars = {}
indices_by_cluster_severity = {(c, s): [] for c in clusters for s in severity_levels}

var_idx = 0
for s in range(1, 6):  # skill levels 1..5
    for j in severity_levels:  
        for c in clusters:
            total_claims = cluster_claim_volume_on_site[c][j] + cluster_claim_volume_virtual[c][j]
            if total_claims > 0:
                drive_multiplier = 1 - 0.2 * (cluster_drive_time[c] / 30)
                # Weighted average of on-site (drive_multiplier) vs. virtual (1.0)
                multiplier = (
                    cluster_claim_volume_virtual[c][j] * 1.0 + 
                    cluster_claim_volume_on_site[c][j] * drive_multiplier
                ) / total_claims
            else:
                multiplier = 1.0
            
            for b in range(num_bits):
                weight = 2 ** b
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
Q = {}

def add_to_Q(i, j, value):
    key = (i, j) if i <= j else (j, i)
    Q[key] = Q.get(key, 0) + value

# Original QUBO terms for each binary variable based on cost and penalty terms.
for i in range(num_vars):
    info_i = binary_vars[i]
    s = info_i['s']
    j = info_i['j']
    c = info_i['cluster']
    weight = info_i['weight']
    prod_rate = info_i['prod']
    
    cost_term = X + daily_cost * (target_days[j] - 1)
    lin_cost = cost_term * weight
    
    local_claim = cluster_claim_volume[c][j]
    # Penalty: -2*A*(claim_count)*(target_days)*(prod*weight) + A*(target_days^2)*(prod^2*weight^2)
    pen_lin = -2 * A * local_claim * target_days[j] * (prod_rate * weight)
    pen_self = A * (target_days[j] ** 2) * ((prod_rate * weight) ** 2)
    
    add_to_Q(i, i, lin_cost + pen_lin + pen_self)

# Cross terms for variables in the same cluster and claim severity.
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
# STEP 6b: ADD EVENNESS PENALTY TO ENCOURAGE SIMILAR STAFFING ACROSS CLUSTERS
# =============================================================================
# This extra term will penalize the squared difference in total staffing between clusters.
even_penalty = 500  # Adjust as needed based on your problem scale

# Build a mapping from each cluster to its associated variable indices.
cluster_to_vars = {}
for i, info in binary_vars.items():
    c = info['cluster']
    if c not in cluster_to_vars:
        cluster_to_vars[c] = []
    cluster_to_vars[c].append(i)

num_clusters = len(cluster_to_vars)
clusters_sorted = sorted(cluster_to_vars.keys())

# For each cluster, add self-terms. Each cluster appears (num_clusters - 1) times in the difference penalty.
for c in clusters_sorted:
    for i in cluster_to_vars[c]:
        w_i = binary_vars[i]['weight']
        add_to_Q(i, i, even_penalty * (num_clusters - 1) * (w_i ** 2))

# For every distinct pair of clusters, add cross terms to penalize differences.
for idx, c in enumerate(clusters_sorted):
    for d in clusters_sorted[idx+1:]:
        for i in cluster_to_vars[c]:
            for j in cluster_to_vars[d]:
                w_i = binary_vars[i]['weight']
                w_j = binary_vars[j]['weight']
                add_to_Q(i, j, -2 * even_penalty * w_i * w_j)

# STEP 7: SOLVE THE QUBO USING QAOA (Quantum Algorithm)
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Build a QuadraticProgram from Q
qp = QuadraticProgram()
for i in range(num_vars):
    qp.binary_var(name=f'x_{i}')

linear = {}
quadratic = {}
for (i, j), coeff in Q.items():
    if i == j:
        linear[f'x_{i}'] = linear.get(f'x_{i}', 0) + coeff
    else:
        quadratic[(f'x_{i}', f'x_{j}')] = quadratic.get((f'x_{i}', f'x_{j}'), 0) + coeff

qp.minimize(linear=linear, quadratic=quadratic)

optimizer = COBYLA(maxiter=100)
quantum_instance = Aer.get_backend('aer_simulator')
qaoa = QAOA(optimizer=optimizer, quantum_instance=quantum_instance)
quantum_optimizer = MinimumEigenOptimizer(qaoa)

print("Solving the extended QUBO with QAOA (Quantum Algorithm)...")
result = quantum_optimizer.solve(qp)

solution = [int(result.x[i]) for i in range(num_vars)]
energy = result.fval
print(f"Best energy found (QAOA): {energy}")

# =============================================================================
# STEP 8: DECODE THE SOLUTION
# =============================================================================
staffing_solution = {}
for s in range(1, 6):
    for j in severity_levels:
        for c in clusters:
            staffing_solution[(s, j, c)] = 0

for i in range(num_vars):
    info = binary_vars[i]
    if solution[i] == 1:
        key = (info['s'], info['j'], info['cluster'])
        staffing_solution[key] += info['weight']

def calculate_cluster_resolution():
    resolution = {}
    for c in clusters:
        total_capacity = 0.0
        total_claims = 0
        for j in severity_levels:
            capacity = 0.0
            for s in range(1, 6):
                count = staffing_solution.get((s, j, c), 0)
                capacity += count * productivity[(s, j)] * target_days[j]
            claim_vol = cluster_claim_volume[c][j]
            total_capacity += min(capacity, claim_vol)
            total_claims += claim_vol
        resolution[c] = total_capacity / total_claims if total_claims > 0 else 1.0
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

cluster_resolution = calculate_cluster_resolution()

for c, res in cluster_resolution.items():
    print(f"Cluster {c} resolution: {res * 100:.2f}%")

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
    cluster_target_met[c] = 'Yes' if cluster_resolution[c] >= 0.9 else 'No'

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
# COMBINED VISUALIZATION: ELBOW PLOT, METRICS, AND HANDLER DISTRIBUTION
# =============================================================================
fig = plt.figure(figsize=(18, 20))  # Increase height for an extra row
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

# -- Subplot 1: Penalized Score Plot --
ax0 = fig.add_subplot(gs[0, :])
# Extract k values and their corresponding penalized scores
k_vals = [score[0] for score in penalized_scores]
pen_scores = [score[1] for score in penalized_scores]
ax0.plot(k_vals, pen_scores, marker='o')
ax0.set_xlabel("Number of Clusters (k)")
ax0.set_ylabel("Penalized Score")
ax0.set_title("Penalized Score for Different k Values")

# -- Subplot 2: Total Staffing Cost per Cluster --
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(df_metrics['Cluster'].astype(str), df_metrics['Total Cost'])
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Total Staffing Cost')
ax1.set_title('Total Staffing Cost per Cluster')

# -- Subplot 3: Average Resolution Days per Cluster --
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(df_metrics['Cluster'].astype(str), df_metrics['Average Resolution Days'])
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Average Resolution Days')
ax2.set_title('Average Resolution Days per Cluster')

# -- Subplot 4: Resolution Percentage (+ annotation for target) --
ax3 = fig.add_subplot(gs[2, 0])
bars = ax3.bar(df_metrics['Cluster'].astype(str), df_metrics['Resolution Percentage'])
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Resolution Percentage (%)')
ax3.set_title('Estimated Resolution Percentage per Cluster')
for idx, bar in enumerate(bars):
    height = bar.get_height()
    target_flag = df_metrics['Target Met'].iloc[idx]
    ax3.annotate(f'{height:.2f}%\n(Target: {target_flag})',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords='offset points',
                 ha='center', va='bottom')

# -- Subplot 5: On-site vs. Virtual Claim Handling per Cluster --
ax4 = fig.add_subplot(gs[2, 1])
width = 0.35
indices = np.arange(len(df_metrics))
onsite_bars = ax4.bar(indices - width/2, df_metrics['On-site %'], width, label='On-site %')
virtual_bars = ax4.bar(indices + width/2, df_metrics['Virtual %'], width, label='Virtual %')
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('On-site vs. Virtual Claim Handling per Cluster')
ax4.set_xticks(indices)
ax4.set_xticklabels(df_metrics['Cluster'].astype(str))
ax4.legend()

# Annotate each cluster with the drive time
for idx, dt in enumerate(df_metrics['Drive Time (min)']):
    max_pct = max(df_metrics['On-site %'].iloc[idx], df_metrics['Virtual %'].iloc[idx])
    ax4.text(idx, max_pct + 2, f'DT: {dt}m', ha='center', va='bottom', fontsize=9)

# =============================================================================
# NEW Subplot 6: Distribution of Handler Skills per Cluster
# Aggregate the staffing_solution for each cluster by skill
cluster_skill_counts = {}
for c in clusters:
    cluster_skill_counts[c] = {s: 0 for s in range(1, 6)}

for (s, j, c), val in staffing_solution.items():
    cluster_skill_counts[c][s] += val

ax5 = fig.add_subplot(gs[3, :])
clusters_sorted = sorted(cluster_skill_counts.keys())
skill_levels = [1, 2, 3, 4, 5]
x_indices = np.arange(len(clusters_sorted))
bottom_vals = np.zeros(len(clusters_sorted))
for skill in skill_levels:
    counts = [cluster_skill_counts[c][skill] for c in clusters_sorted]
    ax5.bar(x_indices, counts, bottom=bottom_vals, label=f'Skill {skill}')
    bottom_vals += np.array(counts)

ax5.set_title("Distribution of Handler Skills per Cluster")
ax5.set_xlabel("Cluster")
ax5.set_ylabel("Number of Handlers")
ax5.set_xticks(x_indices)
ax5.set_xticklabels([str(c) for c in clusters_sorted])
ax5.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()

# =============================================================================
# DYNAMIC STAFFING: REAL-TIME ADJUSTMENTS
# =============================================================================
def update_claims(new_claims_df):
    """
    Update the claim data with new claims and re-run the optimization.
    (Truncated for brevity; replicate your QUBO building, solving, etc.)
    """
    print('Claims updated and optimization re-run based on new data.')


