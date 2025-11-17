#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:00:21 2024

@author: art
"""

import os
import matplotlib.pyplot as plt
from skimage import io

# Chemin vers le répertoire tests
tests_dir = "tests"

# Récupérer tous les fichiers jpg du répertoire tests
image_files = [f for f in os.listdir(tests_dir) if f.lower().endswith('.jpg')]

# Trier les fichiers par ordre numérique,
# en supposant qu'ils sont nommés "1.jpg", "2.jpg", ...
image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

# Nombre d'images à tester
nombre_images = len(image_files)

# Créer une liste possédant le nom de tous les fichiers jpg à tester
liste_fichiers = image_files

# Lire les barycentres
barycentres = []
with open(os.path.join(tests_dir, "barycentres.txt"), "r") as bf:
    for line in bf:
        line = line.strip()
        if line:
            # Chaque ligne est du type "x,y"
            x_str, y_str = line.split(',')
            x, y = float(x_str), float(y_str)
            barycentres.append((x, y))

# Lire les vecteurs
vecteurs = []
with open(os.path.join(tests_dir, "vecteurs.txt"), "r") as vf:
    for line in vf:
        line = line.strip()
        if line:
            # Chaque ligne est du type "u_x,u_y"
            ux_str, uy_str = line.split(',')
            ux, uy = float(ux_str), float(uy_str)
            vecteurs.append((ux, uy))

# Vérification
print("Nombre d'images :", nombre_images)
print("Liste des fichiers :", liste_fichiers)
print("Barycentres :", barycentres)
print("Vecteurs :", vecteurs)

# Affichage des images
for i, img_name in enumerate(liste_fichiers, start=1):
    img_path = os.path.join(tests_dir, img_name)
    image = io.imread(img_path)

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {i}: {img_name}\nBarycentre: {barycentres[i-1]}, Vecteur: {vecteurs[i-1]}")
    plt.axis('off')
    plt.show()
