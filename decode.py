import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d

# Charger l'image
image_path = "assets/exemple3.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_height, image_width = image.shape

# Lire les barycentres et vecteurs propres
barycentres = np.loadtxt("tests/barycentres.txt", delimiter=",")
vecteurs = np.loadtxt("tests/vecteurs.txt", delimiter=",")

# Tracer les rayons sur l'image avec ajustement de l'affichage
plt.figure(figsize=(10, 10))  # Ajuster la taille de la figure
plt.imshow(image, cmap="gray")
plt.title("Rayons tracés avec limites ajustées")

signatures = []

# Table des configurations
configurations = {
    "A": {
        "0": "BBBNNBN", "1": "BBNNBBN", "2": "BBNBBNN", "3": "BNNNNBN", 
        "4": "BNBBBNN", "5": "BNNBBBN", "6": "BNBNNNN", "7": "BNNNBNN", 
        "8": "BNNBNNN", "9": "BBBNBNN"
    },
    "B": {
        "0": "BNBBNNN", "1": "BNNBBNN", "2": "BBNNBNN", "3": "BNBBBBN", 
        "4": "BBNNNBN", "5": "BNNNBBN", "6": "BBBBNBN", "7": "BBNBBBN", 
        "8": "BBNBNBB", "9": "BBNBNNN"
    },
    "C": {
        "0": "NNNBBNB", "1": "NNBBNNB", "2": "NNBNNBB", "3": "NBBBBNB", 
        "4": "NBNNNBB", "5": "NBBNNNB", "6": "NBNBBBB", "7": "NBBBNBB", 
        "8": "NBBNBBB", "9": "NNNBNBB"
    }
}

for barycentre, vecteur in zip(barycentres, vecteurs):
    # Barycentre (point de départ)
    x0, y0 = barycentre

    # Étendre le vecteur pour tracer un rayon
    u_x, u_y = vecteur
    length = max(image_width, image_height)  # Ajuster la longueur dynamiquement
    x_start = x0 - length * u_x
    y_start = y0 - length * u_y
    x_end = x0 + length * u_x
    y_end = y0 + length * u_y

    x_start = np.clip(x_start, 0, image_width - 1)
    y_start = np.clip(y_start, 0, image_height - 1)
    x_end = np.clip(x_end, 0, image_width - 1)
    y_end = np.clip(y_end, 0, image_height - 1)

    # Tracer le rayon
    plt.plot([x_start, x_end], [y_start, y_end], color="red", linewidth=2)
    plt.plot(x0, y0, "bo", markersize=5)  # Barycentre

    # Échantillonnage de l'intensité de l'image le long du rayon initial
    num_points = int(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))
    num_points = max(num_points, 95)  # Toujours avoir au moins 95 points pour un EAN13

    x = np.linspace(x_start, x_end, num_points)
    y = np.linspace(y_start, y_end, num_points)
    signature = image[y.astype(int), x.astype(int)]
    signatures.append(signature)

plt.axis("off")
plt.tight_layout()
plt.show()

def calculer_seuil_otsu(signature, nb_classes=256):
    # Calcul de l'histogramme
    hist, bins = np.histogram(signature, bins=nb_classes, range=(0, 256))
    
    # Normalisation de l'histogramme (pour obtenir des probabilités)
    hist_normalized = hist / hist.sum()

    # Calcul des probabilités cumulées et des moyennes cumulées
    P1 = np.cumsum(hist_normalized)  # Probabilités cumulées pour les classes 1
    P2 = 1 - P1                      # Probabilités cumulées pour les classes 2
    m = np.cumsum(hist_normalized * np.arange(nb_classes))  # Moyenne cumulée
    m_total = m[-1]                 # Moyenne totale de toutes les intensités

    # Calcul du critère d'Otsu
    sigma_b_squared = (m_total * P1 - m) ** 2 / (P1 * P2 + 1e-6)  # Variance inter-classes
    sigma_b_squared[P1 == 0] = 0  # Éviter les divisions par zéro
    sigma_b_squared[P2 == 0] = 0

    # Trouver le seuil qui maximise sigma_b_squared
    seuil_otsu = np.argmax(sigma_b_squared)
    return seuil_otsu

def binariser_signature(signature, seuil):

    return (signature > seuil).astype(int)


# Fonction de ré-échantillonnage
def reechantillonner(signature, new_num_points=1000):
    x_old = np.linspace(0, 1, len(signature))
    x_new = np.linspace(0, 1, new_num_points)
    interpolation = interp1d(x_old, signature, kind='linear')
    return interpolation(x_new)

# Traiter une signature (par exemple la première)
signature = signatures[0]

# Ré-échantillonnage de la signature initiale
signature_reechantillonnee = reechantillonner(signature)

# Estimation du seuil de binarisation
seuil = calculer_seuil_otsu(signature_reechantillonnee)
signature_binaire = binariser_signature(signature_reechantillonnee, seuil)

# Déterminer les limites gauche et droite
indices = np.where(np.diff(signature_binaire) != 0)[0]
if len(indices) > 1:
    debut, fin = indices[0], indices[-1]
else:
    debut, fin = 0, len(signature_binaire)
    
# Extraction d’une seconde signature
x_start_util = debut / len(signature_binaire) * (x[-1] - x[0]) + x[0]
y_start_util = debut / len(signature_binaire) * (y[-1] - y[0]) + y[0]
x_end_util = fin / len(signature_binaire) * (x[-1] - x[0]) + x[0]
y_end_util = fin / len(signature_binaire) * (y[-1] - y[0]) + y[0]

x_util = np.linspace(x_start_util, x_end_util, num_points)
y_util = np.linspace(y_start_util, y_end_util, num_points)
seconde_signature = image[y_util.astype(int), x_util.astype(int)]

# Ré-échantillonnage de la seconde signature
seconde_signature_reechantillonnee = reechantillonner(seconde_signature)

# Binarisation de la seconde signature
seconde_signature_binaire = (seconde_signature_reechantillonnee > seuil).astype(int)

# Fonction pour découper et normaliser la signature complète
def decouper_signature(signature_binaire, longueur_totale=None, nb_chiffres=12):
    if longueur_totale is None:
        longueur_totale = len(signature_binaire)
    u = longueur_totale // nb_chiffres
    segments = [signature_binaire[i * u:(i + 1) * u] for i in range(nb_chiffres)]
    # Normaliser chaque segment à 7 bits
    normalized_segments = [normaliser_segment(segment, expected_length=7) for segment in segments]
    for i, segment in enumerate(normalized_segments):
        segment_str = ''.join(['B' if bit == 1 else 'N' for bit in segment])
        print(f"Segment {i} normalisé (longueur {len(segment)}): {segment_str}")
    return normalized_segments



def decoder_segment(segment, famille, configurations):
    segment_str = ''.join(['B' if bit == 1 else 'N' for bit in segment])
    print(f"Tentative de décodage du segment : {segment_str} (Famille : {famille})")
    for chiffre, code in configurations[famille].items():
        print(f"  Comparaison avec : {chiffre} -> {code}")
        if segment_str == code:
            return chiffre
    print(f"  Aucun modèle trouvé pour le segment : {segment_str}")
    return None  # Si aucune correspondance


def decoder_code_barres(segments, familles, configurations):
    chiffres = []
    for i, (segment, famille) in enumerate(zip(segments, familles)):
        segment_str = ''.join(['B' if bit == 1 else 'N' for bit in segment])
        print(f"Segment {i}: {segment_str} (Famille: {famille})")
        if famille in configurations:
            for chiffre, code in configurations[famille].items():
                if segment_str == code:
                    print(f"  Correspond au chiffre : {chiffre}")
                    chiffres.append(chiffre)
                    break
            else:
                print(f"  Aucun modèle trouvé pour ce segment {i}.")
                chiffres.append('?')  # Ajouter un caractère inconnu
    return chiffres



# Fonction pour normaliser les segments à une longueur cible
def normaliser_segment(segment, expected_length=7):
    if len(segment) != expected_length:
        x_old = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, expected_length)
        interp = interp1d(x_old, segment, kind="nearest", fill_value="extrapolate")
        return (interp(x_new) > 0.5).astype(int)
    return segment

def determiner_premier_chiffre(familles):
    """
    Détermine le premier chiffre du code-barres EAN13 selon les familles des segments.
    """
    table_premier_chiffre = {
        "AAAAAA": "0", "AABABB": "1", "AABBAB": "2", "AABBBA": "3",
        "ABAABB": "4", "ABBAAB": "5", "ABBBAA": "6", "ABABAB": "7",
        "ABABBA": "8", "ABBABA": "9"
    }
    famille_str = ''.join(familles[:6])
    return table_premier_chiffre.get(famille_str, None)

def calculer_cle_controle(chiffres):
    """
    Calcule la clé de contrôle pour un code-barres EAN13.
    """
    somme_impair = sum(int(chiffres[i]) for i in range(0, 12, 2))  # Chiffres en positions impaires
    somme_pair = sum(int(chiffres[i]) for i in range(1, 12, 2))    # Chiffres en positions paires
    total = somme_impair + 3 * somme_pair
    cle = (10 - (total % 10)) % 10
    return str(cle)

# Exemple d'utilisation avec validation et débogage
try:
    # Hypothèse des familles (par défaut pour EAN13)
    familles = ["A", "A", "A", "A", "A", "A", "C", "C", "C", "C", "C", "C"]

    # Découper la signature en segments
    segments = decouper_signature(seconde_signature_binaire, longueur_totale=len(seconde_signature_binaire), nb_chiffres=12)

    # Décoder les segments en chiffres
    chiffres = decoder_code_barres(segments, familles, configurations)

    # Déterminer le premier chiffre
    premier_chiffre = determiner_premier_chiffre(familles)
    if premier_chiffre is None:
        raise ValueError("Impossible de déterminer le premier chiffre du code-barres.")

    # Ajouter le premier chiffre au début du code-barres
    chiffres.insert(0, premier_chiffre)

    # Calculer la clé de contrôle
    cle_controle = calculer_cle_controle(chiffres)
    
    # Vérifier la clé de contrôle
    code_complet = ''.join(chiffres)
    if cle_controle == chiffres[-1]:
        print("Code-barres valide :", code_complet)
    else:
        print("Code-barres invalide : clé de contrôle incorrecte.")
        print(f"Code partiel : {''.join(chiffres[:-1])}, clé attendue : {cle_controle}")
except Exception as e:
    print("Erreur :", e)


# Visualisation des résultats
plt.figure(figsize=(10, 4))
plt.plot(signature_reechantillonnee, label="Signature Ré-échantillonnée", color="blue")
plt.title("Signature ré-échantillonnée le long du rayon")
plt.xlabel("Position")
plt.ylabel("Intensité")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(signature_binaire, label="Signature Binaire", color="orange")
plt.title("Signature binaire")
plt.xlabel("Position")
plt.ylabel("Valeur (0 ou 1)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(seconde_signature_reechantillonnee, label="Seconde Signature Ré-échantillonnée", color="green")
plt.title("Seconde signature ré-échantillonnée")
plt.xlabel("Position")
plt.ylabel("Intensité")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(seconde_signature_binaire, label="Seconde Signature Binaire", color="red")
plt.title("Seconde signature binaire")
plt.xlabel("Position")
plt.ylabel("Valeur (0 ou 1)")
plt.legend()
plt.show()


    
    