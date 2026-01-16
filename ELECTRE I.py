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

weights = {
    'Intérêt': 0.20, 'Encadrement': 0.20, 'Pré-embauche': 0.15,
    'Gratification': 0.10, 'Mobilité': 0.10, 'Stack Tech': 0.10,
    'Réputation': 0.05, 'Type': 0.04, 'Avis': 0.03, 'Télétravail': 0.03
}

# Scénarios de robustesse
scenarios = {
    "1_Base":      (0.60, 0.80), 
    "2_Exigeant":  (0.70, 0.50), 
    "3_Souple":    (0.50, 0.90), 
    "4_Veto_Strict":(0.60, 0.30)  
}

def calculate_matrices(dataframe, weights_dict):
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

def get_kernel_and_graph(c_mat, d_mat, s, v, title, filename):
    outranking = (c_mat >= s) & (d_mat <= v)
    
    dominated = outranking.any(axis=0) 
    kernel = dominated[~dominated].index.tolist()
    
    G = nx.DiGraph()
    candidates = c_mat.index.tolist()
    
    for cand in candidates:
        color = '#90EE90' if cand in kernel else '#ffdfba'
        G.add_node(cand, label=cand, color=color)
    
    for a in candidates:
        for b in candidates:
            if a != b and outranking.loc[a, b]:
                G.add_edge(a, b)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=25, node_size=2000, edge_color='#555555')
    
    plt.title(f"{title}\n(s={s}, v={v}) | Noyau: {kernel}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    
    return kernel, outranking

mat_c, mat_d = calculate_matrices(df, weights)

print("=== MATRICES ELECTRE I ===")
print("\n--- MATRICE DE CONCORDANCE ---")
print(mat_c.fillna("-").round(2))
print("\n--- MATRICE DE DISCORDANCE ---")
print(mat_d.fillna("-").round(2))
print("-" * 60)

print("\n=== ÉTUDE DE ROBUSTESSE ===")
print(f"{'Scénario':<15} {'Seuils (s, v)':<15} {'Taille Noyau':<15} {'Composition du Noyau'}")
print("-" * 80)

for name, (s, v) in scenarios.items():
    title = f"Graphe ELECTRE I - {name}"
    fname = f"electre1_{name}.png"
    
    noyau, relations = get_kernel_and_graph(mat_c, mat_d, s, v, title, fname)
    
    noyau_str = ", ".join(noyau)
    print(f"{name:<15} {f's={s}, v={v}':<15} {len(noyau):<15} {noyau_str}")

    if name == "1_Base":
        print(f"\n[Détail Base] Relations de surclassement validées :")
        count = 0
        for i in relations.index:
            for j in relations.columns:
                if relations.loc[i, j]:
                    print(f"  - {i} surclasse {j}")
                    count += 1
        if count == 0: print("  - Aucune relation de surclassement stricte.")
        print("-" * 80)