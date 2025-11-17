#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:21:03 2024

@author: thomas
"""

import networkx as nx
import matplotlib.pyplot as plt

# Création du graphe pour le treillis
G = nx.DiGraph()

# Définition des états
states = ["00", "01", "10", "11"]
levels = 5  # Nombre de sections dans le treillis

# Ajouter les nœuds pour chaque niveau du treillis
nodes = []
for i in range(levels):
    for state in states:
        node = f"{state}_{i}"
        nodes.append(node)
        G.add_node(node, level=i)

# Ajouter les arêtes pour chaque transition
outputs = {
    ("00", "00"): "00",
    ("00", "10"): "11",
    ("01", "01"): "00",
    ("01", "11"): "11",
    ("10", "00"): "10",
    ("10", "10"): "01",
    ("11", "01"): "10",
    ("11", "11"): "01",
}

transitions = {
    "00": [("00", 0), ("10", 1)],
    "01": [("01", 0), ("11", 1)],
    "10": [("00", 0), ("10", 1)],
    "11": [("01", 0), ("11", 1)],
}

for i in range(levels - 1):
    for state, next_states in transitions.items():
        for next_state, input_bit in next_states:
            source = f"{state}_{i}"
            target = f"{next_state}_{i + 1}"
            output = outputs[(state, next_state)]
            G.add_edge(source, target, label=f"{input_bit}/{output}")

# Positionnement des nœuds
pos = {}
x_offset = 2
for i in range(levels):
    for j, state in enumerate(states):
        pos[f"{state}_{i}"] = (i * x_offset, -j)

# Dessiner le graphe
plt.figure(figsize=(12, 8))
nx.draw(
    G,
    pos,
    with_labels=False,
    node_color="red",
    node_size=800,
    edge_color="blue",
    font_weight="bold",
    font_color="green",
)

# Ajouter les étiquettes des nœuds
for node, (x, y) in pos.items():
    plt.text(x, y, node.split("_")[0], fontsize=10, ha="center", va="center", color="white")

# Ajouter les étiquettes des arêtes
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="green")

# Ajouter des annotations pour "Ouverture", "Section", et "Fermeture"
plt.text(-1, 1, "Ouverture", fontsize=12, color="black", ha="center")
plt.text(levels * x_offset / 2, 1, "Section", fontsize=12, color="black", ha="center")
plt.text((levels - 1) * x_offset, 1, "Fermeture", fontsize=12, color="black", ha="center")

plt.title("Treillis adapté à votre sujet")
plt.axis("off")

# Sauvegarde
plt.savefig("Treillis_Adapted_Local.png", dpi=300)
plt.show()