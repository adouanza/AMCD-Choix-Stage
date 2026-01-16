import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

scenarios_weights = {
    "1. BASE (Equilibré)": {
        'Intérêt': 0.20, 'Encadrement': 0.20, 'Pré-embauche': 0.15,
        'Gratification': 0.10, 'Mobilité': 0.10, 'Stack Tech': 0.10,
        'Réputation': 0.05, 'Type': 0.04, 'Avis': 0.03, 'Télétravail': 0.03
    },
    "2. TECH (Passion)": {
        'Intérêt': 0.25, 'Encadrement': 0.05, 'Pré-embauche': 0.05,
        'Gratification': 0.05, 'Mobilité': 0.05, 'Stack Tech': 0.35,
        'Réputation': 0.05, 'Type': 0.05, 'Avis': 0.00, 'Télétravail': 0.10
    },
    "3. SÉCURITÉ (Raison)": {
        'Intérêt': 0.05, 'Encadrement': 0.15, 'Pré-embauche': 0.25,
        'Gratification': 0.25, 'Mobilité': 0.10, 'Stack Tech': 0.05,
        'Réputation': 0.10, 'Type': 0.05, 'Avis': 0.00, 'Télétravail': 0.00
    },
    "4. SOCIAL (Bien-être)": {
        'Intérêt': 0.10, 'Encadrement': 0.30, 'Pré-embauche': 0.05,
        'Gratification': 0.10, 'Mobilité': 0.05, 'Stack Tech': 0.05,
        'Réputation': 0.05, 'Type': 0.05, 'Avis': 0.15, 'Télétravail': 0.10
    }
}

scenarios_thresholds = {
    "1. STANDARD (Base)":  (0.95, 0.75, 0.55, 0.80, 0.30), 
    "2. STRICT (Exigeant)":(0.98, 0.85, 0.65, 0.70, 0.20), 
    "3. LAXISTE (Souple)": (0.90, 0.65, 0.50, 0.90, 0.40), 
    "4. VETO SÉVÈRE":      (0.95, 0.75, 0.55, 0.80, 0.10), 
    "5. VETO SOUPLE":      (0.95, 0.75, 0.55, 1.00, 0.50)  
}

def get_matrices_robust(dataframe, weights_dict):
    criteria = list(weights_dict.keys())
    w_values = np.array([weights_dict[c] for c in criteria])
    candidates = dataframe.index.tolist()
    
    ranges = (dataframe[criteria].max() - dataframe[criteria].min()).replace(0, 1).values
    
    c_mat = pd.DataFrame(index=candidates, columns=candidates, dtype=float)
    d_mat = pd.DataFrame(index=candidates, columns=candidates, dtype=float)
    
    for row in candidates:
        for col in candidates:
            if row == col: continue
            vals_a = dataframe.loc[row, criteria].values.astype(float)
            vals_b = dataframe.loc[col, criteria].values.astype(float)
            
            c_score = np.sum(w_values[np.where(vals_a >= vals_b)[0]])
            c_mat.loc[row, col] = c_score
            
            diffs = (vals_b - vals_a) / ranges
            d_score = np.max(diffs[np.where(diffs > 0)[0]]) if np.any(diffs > 0) else 0.0
            d_mat.loc[row, col] = d_score
            
    return c_mat, d_mat

def electre_ii_ranking_dynamic(c_mat, d_mat, thresholds):
    C_PLUS, C_ZERO, C_MINUS, D_1, D_2 = thresholds
    candidates = c_mat.index.tolist()
    n = len(candidates)
    
    mat_strong = np.zeros((n, n))
    mat_weak = np.zeros((n, n))
    idx_map = {name: i for i, name in enumerate(candidates)}
    
    for a in candidates:
        for b in candidates:
            if a == b: continue
            i, j = idx_map[a], idx_map[b]
            
            c_val = c_mat.loc[a, b]
            d_val = d_mat.loc[a, b]
            c_inv = c_mat.loc[b, a]
            
            if c_val < c_inv: continue 

            if ((c_val >= C_PLUS) and (d_val <= D_1)) or ((c_val >= C_ZERO) and (d_val <= D_2)):
                mat_strong[i, j] = 1
            
            if (c_val >= C_MINUS) and (d_val <= D_1):
                mat_weak[i, j] = 1
                
    scores = {}
    for name in candidates:
        i = idx_map[name]
        net_strong = np.sum(mat_strong[i, :]) - np.sum(mat_strong[:, i])
        net_weak = np.sum(mat_weak[i, :]) - np.sum(mat_weak[:, i])
        scores[name] = {'strong': net_strong, 'weak': net_weak}

    sorted_candidates = sorted(scores.items(), key=lambda x: (x[1]['strong'], x[1]['weak']), reverse=True)
    return sorted_candidates, mat_strong, mat_weak

w_base = scenarios_weights["1. BASE (Equilibré)"]
t_base = scenarios_thresholds["1. STANDARD (Base)"]

print("="*60)
print(" RÉSULTATS : SCÉNARIO DE BASE")
print("="*60)
print(f"SEUILS UTILISÉS (C+, C0, C-, D1, D2) : {t_base}")
print("POIDS UTILISÉS :")
for k, v in w_base.items():
    print(f"  - {k:<15} : {v}")

c_mat_base, d_mat_base = get_matrices_robust(df, w_base)
class_base, m_strong_base, m_weak_base = electre_ii_ranking_dynamic(c_mat_base, d_mat_base, t_base)

print("\n--- MATRICE DE CONCORDANCE ---")
print(c_mat_base.round(2).fillna("-"))
print("\n--- MATRICE DE DISCORDANCE ---")
print(d_mat_base.round(2).fillna("-"))

print(f"\n--- CLASSEMENT FINAL ---")
print(f"{'Rang':<5} {'ID':<5} {'Entreprise':<20} {'Score Fort':<12} {'Score Faible'}")
print("-" * 60)

current_rank = 1
for i in range(len(class_base)):
    cand, scores = class_base[i]
    if i > 0:
        prev = class_base[i-1][1]
        if scores['strong'] < prev['strong'] or scores['weak'] < prev['weak']:
            current_rank = i + 1
    nom = df.loc[cand, 'Entreprise']
    print(f"{current_rank:<5} {cand:<5} {nom:<20} {scores['strong']:<12.1f} {scores['weak']:.1f}")

print("\n" + "="*60)
print(" ANALYSE DE ROBUSTESSE (POIDS VARIABLES)")
print("="*60)
print("SCÉNARIOS TESTÉS :")
for name, w in scenarios_weights.items():
    top3 = sorted(w.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_str = ", ".join([f"{k}={v}" for k, v in top3])
    print(f"  > {name:<20} : {top3_str}...")

res_w = {}
for name, w in scenarios_weights.items():
    c, d = get_matrices_robust(df, w)
    classement, _, _ = electre_ii_ranking_dynamic(c, d, t_base)
    
    ranks = {}
    curr = 1
    for i in range(len(classement)):
        cand, sc = classement[i]
        if i > 0:
            prev = classement[i-1][1]
            if sc['strong'] < prev['strong'] or sc['weak'] < prev['weak']:
                curr = i + 1
        ranks[cand] = curr
    res_w[name] = ranks

df_rob_w = pd.DataFrame(res_w)
df_rob_w['Entreprise'] = df['Entreprise']
cols = ['Entreprise'] + list(scenarios_weights.keys())
print("\nTABLEAU DES RANGS (POIDS) :")
print(df_rob_w[cols].sort_values("1. BASE (Equilibré)"))

print("\n" + "="*60)
print(" ANALYSE DE ROBUSTESSE ")
print("="*60)
print("SCÉNARIOS TESTÉS :")
print(f"{'Scénario':<25} {'C+':<6} {'C0':<6} {'C-':<6} {'D1':<6} {'D2':<6}")
print("-" * 65)
for name, t in scenarios_thresholds.items():
    print(f"{name:<25} {t[0]:<6} {t[1]:<6} {t[2]:<6} {t[3]:<6} {t[4]:<6}")

res_t = {}
for name, t in scenarios_thresholds.items():
    classement, _, _ = electre_ii_ranking_dynamic(c_mat_base, d_mat_base, t)
    
    ranks = {}
    curr = 1
    for i in range(len(classement)):
        cand, sc = classement[i]
        if i > 0:
            prev = classement[i-1][1]
            if sc['strong'] < prev['strong'] or sc['weak'] < prev['weak']:
                curr = i + 1
        ranks[cand] = curr
    res_t[name] = ranks

df_rob_t = pd.DataFrame(res_t)
df_rob_t['Entreprise'] = df['Entreprise']
cols_t = ['Entreprise'] + list(scenarios_thresholds.keys())
print("\nTABLEAU DES RANGS (SEUILS) :")
print(df_rob_t[cols_t].sort_values("1. STANDARD (Base)"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
candidates = df.index.tolist()
NODE_SIZE = 2200

def draw_graph(ax, title, matrix_binary, edge_color, note):
    G = nx.DiGraph()
    for cand in candidates:
        G.add_node(cand, label=cand)
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j: continue
            if matrix_binary[i, j] == 1:
                G.add_edge(candidates[i], candidates[j])
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#ffdfba', node_size=NODE_SIZE, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=NODE_SIZE, arrowstyle='-|>', arrowsize=20, 
                           edge_color=edge_color, width=2, connectionstyle='arc3,rad=0.1')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.text(0.5, -0.05, note, transform=ax.transAxes, ha='center', style='italic', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.axis('off')

C_P, C_0, C_M, D1, D2 = t_base
draw_graph(ax1, "1. Graphe FORT (Noyau Dur)", m_strong_base, 'green', 
           f"Base : C+={C_P}, D1={D1} OU C0={C_0}, D2={D2}")
draw_graph(ax2, "2. Graphe GLOBAL (Réalité)", m_weak_base, 'blue', 
           f"Base : Inclut relations tolérantes C-={C_M}, D1={D1}")

plt.tight_layout()
plt.savefig("electre_ii_final_readable.png", dpi=300)
plt.show()