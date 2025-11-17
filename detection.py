import os
import shutil
import numpy as np
from scipy import ndimage
from skimage import io, color, morphology, measure
import matplotlib.pyplot as plt

# Création et nettoyage du répertoire tests
tests_dir = "tests"
if os.path.exists(tests_dir):
    # Supprimer tout le contenu du répertoire tests
    for f in os.listdir(tests_dir):
        file_path = os.path.join(tests_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    os.mkdir(tests_dir)

# Ouverture des fichiers barycentres et vecteurs
barycentres_file = open(os.path.join(tests_dir, "barycentres.txt"), "w")
vecteurs_file = open(os.path.join(tests_dir, "vecteurs.txt"), "w")

# Fonction de filtrage gaussien
def gaussian_filter(image, sigma, order):
    def gaussian_kernel1d(sigma, order, radius):
        if order == 0:
            kernel = np.exp(-0.5 * (np.arange(-radius, radius+1) / sigma)**2)
        elif order == 1:
            kernel = -np.arange(-radius, radius+1) / sigma**2 * np.exp(-0.5 * (np.arange(-radius, radius+1) / sigma)**2)
        else:
            raise ValueError("Order not supported")
        return kernel / np.sum(np.abs(kernel))

    radius = int(3 * sigma + 0.5)
    kernel_x = gaussian_kernel1d(sigma, order[0], radius)
    kernel_y = gaussian_kernel1d(sigma, order[1], radius)

    if order[0] == 1:
        image = ndimage.convolve(image, kernel_x[:, np.newaxis])
    if order[1] == 1:
        image = ndimage.convolve(image, kernel_y[np.newaxis, :])

    return image

# Paramètres
sigma_G = 0.2
sigma_T = 8
seuil = 0.3

# Charger l'image en niveaux de gris
image = io.imread('assets/exemple3.jpg')
if image.ndim == 3:
    image = color.rgb2gray(image)
image = image.astype(np.float64)

# Utilisation de la fonction gaussian_filter
Ix = gaussian_filter(image, sigma=sigma_G, order=[0, 1])
Iy = gaussian_filter(image, sigma=sigma_G, order=[1, 0])

# Calcul de la magnitude du gradient pour la normalisation
magnitude = np.sqrt(Ix**2 + Iy**2)
magnitude[magnitude == 0] = 1e-10

# Normalisation des gradients
Ix_normalized = Ix / magnitude
Iy_normalized = Iy / magnitude

# Calcul des éléments du tenseur de structure avec les gradients normalisés
Txx = Ix_normalized * Ix_normalized
Txy = Ix_normalized * Iy_normalized
Tyy = Iy_normalized * Iy_normalized

# Application de la pondération gaussienne
Wxx = ndimage.gaussian_filter(Txx, sigma=sigma_T)
Wxy = ndimage.gaussian_filter(Txy, sigma=sigma_T)
Wyy = ndimage.gaussian_filter(Tyy, sigma=sigma_T)

# Calcul de la mesure de dispersion D(x, y)
numerateur = np.sqrt((Wxx - Wyy)**2 + 4 * Wxy**2)
denominateur = Wxx + Wyy
denominateur[denominateur == 0] = 1e-10

D = (denominateur - numerateur) / denominateur

# Binarisation de D(x, y) avec un seuil
masque = D > seuil

# Affichage des résultats
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(D, cmap='gray')
plt.title('Mesure de dispersion D(x, y)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masque, cmap='gray', alpha=0.5)
plt.title('Zones probables du code-barres')
plt.axis('off')

plt.tight_layout()
plt.show()

# Inversion du masque pour obtenir les tâches noires
masque_inverse = ~masque
labels = measure.label(masque_inverse, connectivity=2)

plt.figure(figsize=(8, 8))
plt.imshow(masque_inverse, cmap='gray')
plt.title('Tâches détectées numérotées avec vecteurs propres')
plt.axis('off')

props = measure.regionprops(labels)

# Paramètres d'ajustement (marges sur les images exportées)
margin_x = 10  # marge en colonnes
margin_y = 10  # marge en lignes

region_count = 1

for prop in props:
    centroid = prop.centroid
    plt.text(centroid[1], centroid[0], str(prop.label), color='red', fontsize=12, ha='center', va='center')

    coords = prop.coords
    centroid_array = np.array(centroid)
    centered_coords = coords - centroid_array

    y = centered_coords[:, 0]
    x = centered_coords[:, 1]
    coord_array = np.vstack((x, y))

    cov_matrix = np.cov(coord_array, bias=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx_max = np.argmax(eigenvalues)
    lambda_max = eigenvalues[idx_max]
    u_max = eigenvectors[:, idx_max]

    # Informations
    print(f"Région {prop.label}:")
    print(f"  Barycentre (centroïde): (x={centroid[1]:.2f}, y={centroid[0]:.2f})")
    print(f"  Matrice de covariance :\n{cov_matrix}")
    print(f"  Valeurs propres : {eigenvalues}")
    print(f"  Vecteurs propres associés :\n{eigenvectors}")
    print(f"  Plus grande valeur propre : {lambda_max}")
    print(f"  Vecteur propre associé : {u_max}\n")

    # Ecriture du barycentre et du vecteur dans les fichiers
    barycentres_file.write(f"{centroid[1]:.4f},{centroid[0]:.4f}\n")
    vecteurs_file.write(f"{u_max[0]:.4f},{u_max[1]:.4f}\n")

    # Déterminer la bounding box de la région
    min_row, min_col, max_row, max_col = prop.bbox

    # Application des marges en tenant compte des limites de l'image
    min_row_expanded = max(min_row - margin_y, 0)
    max_row_expanded = min(max_row + margin_y, image.shape[0])
    min_col_expanded = max(min_col - margin_x, 0)
    max_col_expanded = min(max_col + margin_x, image.shape[1])

    sub_image = image[min_row_expanded:max_row_expanded, min_col_expanded:max_col_expanded]

    io.imsave(os.path.join(tests_dir, f"{region_count}.jpg"), (sub_image * 255).astype(np.uint8))

    # Tracer le vecteur propre sur la tâche
    length = 2 * np.sqrt(lambda_max)
    x0, y0 = centroid[1], centroid[0]
    u_x, u_y = u_max[0], -u_max[1]

    x_vals = [x0 - length * u_x, x0 + length * u_x]
    y_vals = [y0 - length * u_y, y0 + length * u_y]

    plt.plot(x_vals, y_vals, color='green', linewidth=2)
    plt.plot(x0, y0, 'bo', markersize=5)

    region_count += 1

plt.show()

barycentres_file.close()
vecteurs_file.close()
