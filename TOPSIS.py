import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


data = {
    'ID': ['A1', 'A2', 'A3', 'A7', 'A9', 'A11', 'A12'],
    'Entreprise': ['Ubisoft', 'Capgemini', 'MedTech', 'GreenData', 'CGI', 'Orange', 'Crédit Agri'],
    'Intérêt':      [1.00, 0.00, 0.00, 0.75, 0.25, 0.75, 1.00],
    'Encadrement':  [1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00],
    'Pré-embauche': [0.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00],
    'Gratification':[0.13, 0.63, 0.13, 0.00, 0.50, 0.75, 1.00],
    'Mobilité':     [0.00, 1.00, 1.00, 1.00, 1.00, 0.00, 1.00],
    'Stack Tech':   [1.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00],
    'Réputation':   [1.00, 0.50, 0.25, 0.00, 0.50, 0.75, 0.75],    
    'Type':         [1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
    'Avis':         [1.00, 0.33, 0.00, 0.00, 0.13, 0.73, 0.60],
    'Télétravail':  [0.50, 1.00, 1.00, 1.00, 0.00, 0.50, 0.50]
}
df = pd.DataFrame(data).set_index('ID')

# Poids de base
weights = {
    'Intérêt': 0.20, 'Encadrement': 0.20, 'Pré-embauche': 0.15,
    'Gratification': 0.10, 'Mobilité': 0.10, 'Stack Tech': 0.10,
    'Réputation': 0.05, 'Type': 0.04, 'Avis': 0.03, 'Télétravail': 0.03
}

def run_topsis_detailed(dataframe, weights_dict):
    criteria = list(weights_dict.keys())
    w_values = np.array([weights_dict[c] for c in criteria])
    
    # A. Normalisation Vectorielle
    raw_matrix = dataframe[criteria].values.astype(float)
    denom = np.sqrt((raw_matrix**2).sum(axis=0))
    # Protection division par zéro
    denom[denom == 0] = 1 
    norm_matrix = raw_matrix / denom
    
    # B. Pondération
    weighted_matrix = norm_matrix * w_values
    
    # C. Solutions Idéales (Best/Worst)
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    
    # D. Distances
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    # E. Score
    total_dist = dist_best + dist_worst
    scores = np.divide(dist_worst, total_dist, out=np.zeros_like(dist_worst), where=total_dist!=0)
    
    # DataFrames pour affichage
    df_norm = pd.DataFrame(norm_matrix, index=dataframe.index, columns=criteria)
    df_weighted = pd.DataFrame(weighted_matrix, index=dataframe.index, columns=criteria)
    
    return df_norm, df_weighted, ideal_best, ideal_worst, dist_best, dist_worst, scores

# EXÉCUTION BASE
df_norm, df_w, vect_best, vect_worst, d_plus, d_minus, scores = run_topsis_detailed(df, weights)


print("\n" + "="*60)
print(" 1. MATRICE NORMALISÉE (Vectorielle)")
print("="*60)
print(df_norm.round(3))

print("\n" + "="*60)
print(" 2. MATRICE PONDÉRÉE (Normalisée * Poids)")
print("="*60)
print(df_w.round(3))

print("\n" + "="*60)
print(" 3. SOLUTIONS DE RÉFÉRENCE")
print("="*60)
df_ideals = pd.DataFrame({
    'Critère': weights.keys(),
    'A* (Idéal)': vect_best,
    'A- (Anti-Idéal)': vect_worst
}).set_index('Critère')
print(df_ideals.round(4))

df_res = pd.DataFrame({
    'Entreprise': df['Entreprise'],
    'D+ (Idéal)': d_plus,
    'D- (Anti-Idéal)': d_minus,
    'Score': scores
}, index=df.index).sort_values(by='Score', ascending=False)
df_res['Rang'] = range(1, len(df_res) + 1)

print("\n" + "="*60)
print(" 4. CLASSEMENT FINAL TOPSIS (SCÉNARIO BASE)")
print("="*60)
print(df_res)

# ==========================================
# GÉNÉRATION DES 3 GRAPHIQUES
# ==========================================

# --- SCATTER PLOT ---
plt.figure(figsize=(9, 6))
x = df_res['D- (Anti-Idéal)']
y = df_res['D+ (Idéal)']
labels = df_res.index

plt.scatter(x, y, color='#004b8c', s=120, zorder=2)
for i, txt in enumerate(labels):
    plt.annotate(f"{txt} ({df_res.iloc[i]['Entreprise']})", 
                 (x[i], y[i]), xytext=(7, 0), textcoords='offset points', fontsize=10)

plt.title("TOPSIS : Positionnement Géométrique (Base)")
plt.xlabel("Distance au Pire (D-)")
plt.ylabel("Distance au Meilleur (D+)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("topsis_scatter.png", dpi=150)
plt.show()

# --- BAR PLOT (Scores) ---
plt.figure(figsize=(9, 6))
colors = ['#28a745' if s == df_res['Score'].max() else '#004b8c' for s in df_res['Score']]
plt.bar(df_res['Entreprise'], df_res['Score'], color=colors, alpha=0.9)

for i, v in enumerate(df_res['Score']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

plt.title("Classement TOPSIS - Base (Scores)")
plt.ylabel("Score Ci")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("topsis_bar.png", dpi=150)
plt.show()

print("\n[INFO] Images 'topsis_scatter.png' et 'topsis_bar.png' sauvegardées.")


# ==========================================
# 5. ÉTUDE DE ROBUSTESSE
# ==========================================
print("\n" + "="*60)
print(" 5. ANALYSE DE ROBUSTESSE (SCÉNARIOS)")
print("="*60)

scenarios = {
    "1. BASE": weights,
    "2. TECH": {
        'Intérêt': 0.25, 'Encadrement': 0.05, 'Pré-embauche': 0.05,
        'Gratification': 0.05, 'Mobilité': 0.05, 'Stack Tech': 0.35,
        'Réputation': 0.05, 'Type': 0.05, 'Avis': 0.00, 'Télétravail': 0.10
    },
    "3. SÉCURITÉ": {
        'Intérêt': 0.05, 'Encadrement': 0.15, 'Pré-embauche': 0.25,
        'Gratification': 0.25, 'Mobilité': 0.10, 'Stack Tech': 0.05,
        'Réputation': 0.10, 'Type': 0.05, 'Avis': 0.00, 'Télétravail': 0.00
    },
    "4. SOCIAL": {
        'Intérêt': 0.10, 'Encadrement': 0.30, 'Pré-embauche': 0.05,
        'Gratification': 0.10, 'Mobilité': 0.05, 'Stack Tech': 0.05,
        'Réputation': 0.05, 'Type': 0.05, 'Avis': 0.15, 'Télétravail': 0.10
    }
}

results_ranks = {}

for name, scen_weights in scenarios.items():
    _, _, _, _, _, _, sc_scores = run_topsis_detailed(df, scen_weights)
    temp_df = pd.DataFrame({'Score': sc_scores}, index=df.index)
    temp_df = temp_df.sort_values(by='Score', ascending=False)
    temp_df['Rang'] = range(1, len(temp_df) + 1)
    results_ranks[name] = temp_df['Rang'].sort_index()

df_robustness = pd.DataFrame(results_ranks)
df_robustness.insert(0, 'Entreprise', df['Entreprise'])
df_robustness = df_robustness.sort_values(by="1. BASE")

print("\n--- TABLEAU DES RANGS PAR SCÉNARIO ---")
print(df_robustness)

# --- GRAPHIQUE ROBUSTESSE ---
plt.figure(figsize=(10, 6))

scenarios_list = list(scenarios.keys())
candidates = df.index.tolist()
colors_map = {
    'A1': 'red', 'A2': 'purple', 'A3': 'grey', 
    'A7': 'green', 'A9': 'orange', 'A11': 'cyan', 'A12': 'blue'
}

for cand in candidates:
    y_values = [results_ranks[sc][cand] for sc in scenarios_list]
    plt.plot(scenarios_list, y_values, marker='o', linewidth=2, label=f"{cand} ({df.loc[cand, 'Entreprise']})", color=colors_map.get(cand, 'black'))
    
    for i, rank in enumerate(y_values):
        plt.text(i, rank - 0.1, str(rank), ha='center', fontsize=8, fontweight='bold')

plt.gca().invert_yaxis()
plt.title("TOPSIS : Sensibilité du Classement aux Poids", fontweight='bold')
plt.ylabel("Rang (1 = Premier)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("topsis_robustness.png", dpi=150)
plt.show()