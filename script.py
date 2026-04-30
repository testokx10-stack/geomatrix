import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================
# 1. Définition des paramètres phonétiques
# ============================================

# Dictionnaire : pour chaque lettre on associe (freq_Hz, duree, type_angle)
#   - freq_Hz : fréquence fondamentale (pour la synthèse sonore)
#   - duree   : durée relative (entre 0.5 et 2.0)
#   - type_angle : 'voyelle' ou 'consonne' (pour le style de tracé)
#
# Pour simplifier, nous utilisons l'alphabet latin.
# Pour les voyelles : angle = fonction linéaire de la fréquence (grave→0°, aigu→360°)
# Pour les consonnes : angle = valeur fixe + petite variation aléatoire (pour simuler le bruit)

params = {}

# Voyelles (a, e, i, o, u, y) - fréquences typiques (environ)
# on choisit des fréquences dans la gamme audible (220 Hz - 880 Hz)
voyelles = {
    'a': (220, 1.0, 'voyelle'),   # la grave
    'e': (330, 1.0, 'voyelle'),
    'i': (440, 1.0, 'voyelle'),
    'o': (550, 1.0, 'voyelle'),
    'u': (660, 1.0, 'voyelle'),
    'y': (770, 1.0, 'voyelle'),
}
# Consonnes (valeurs approximatives, on prend une fréquence constante + bruit)
consonnes_base = {
    'b': (200, 0.8, 'consonne'), 'c': (250, 0.8, 'consonne'), 'd': (300, 0.8, 'consonne'),
    'f': (350, 0.8, 'consonne'), 'g': (400, 0.8, 'consonne'), 'h': (450, 0.8, 'consonne'),
    'j': (500, 0.8, 'consonne'), 'k': (550, 0.8, 'consonne'), 'l': (600, 0.8, 'consonne'),
    'm': (650, 0.8, 'consonne'), 'n': (700, 0.8, 'consonne'), 'p': (750, 0.8, 'consonne'),
    'q': (800, 0.8, 'consonne'), 'r': (850, 0.8, 'consonne'), 's': (150, 0.8, 'consonne'),
    't': (200, 0.8, 'consonne'), 'v': (300, 0.8, 'consonne'), 'w': (400, 0.8, 'consonne'),
    'x': (500, 0.8, 'consonne'), 'z': (600, 0.8, 'consonne'),
}

params.update(voyelles)
params.update(consonnes_base)

# Pour les lettres non trouvées (accentuées, chiffres, ponctuation), on utilise un défaut
default_param = (440, 0.5, 'silence')

# ============================================
# 2. Fonction pour convertir un texte en séquence de points
# ============================================

def text_to_path(text, step_length=1.0, angle_base=0.0):
    """
    Parcourt le texte et construit une liste de points (x, y).
    step_length : longueur de chaque segment (unitaire).
    angle_base   : angle de départ (par défaut 0).
    Chaque lettre détermine un angle (direction) et une longueur (durée relative).
    """
    # On commence à l'origine
    points = [(0, 0)]
    current_angle = angle_base
    
    for ch in text.lower():
        # Récupération des paramètres
        freq, duree, typ = params.get(ch, default_param)
        # Longueur du segment : step_length * duree
        length = step_length * duree
        # Calcul de l'angle : pour les voyelles, proportionnel à la fréquence (mod 360)
        if typ == 'voyelle':
            # On normalise la fréquence entre 220 et 880 Hz -> angle entre 0 et 360
            ang = (freq - 220) / (880 - 220) * 360
        else:
            # Pour les consonnes, on ajoute une petite variation pour simuler le bruit
            ang = current_angle + np.random.uniform(-20, 20)
            # on garde l'angle dans [0,360)
            ang = ang % 360
        current_angle = ang
        rad = np.radians(ang)
        last_x, last_y = points[-1]
        new_x = last_x + length * np.cos(rad)
        new_y = last_y + length * np.sin(rad)
        points.append((new_x, new_y))
    
    return np.array(points)

# ============================================
# 3. Animation du tracé
# ============================================

def animate_path(points, interval=50, save_as='trajectoire.mp4'):
    """
    Anime l'affichage du chemin point par point.
    points : array Nx2.
    interval : temps entre deux points en millisecondes.
    save_as : nom du fichier de sortie (None pour ne pas sauvegarder).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(points[:,0].min() - 1, points[:,0].max() + 1)
    ax.set_ylim(points[:,1].min() - 1, points[:,1].max() + 1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro', markersize=4)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def update(frame):
        # frame correspond au nombre de points à afficher
        x = points[:frame+1, 0]
        y = points[:frame+1, 1]
        line.set_data(x, y)
        point.set_data([x[-1]], [y[-1]])
        return line, point
    
    ani = FuncAnimation(fig, update, frames=len(points), init_func=init, interval=interval, blit=True)
    if save_as:
        ani.save(save_as, writer='ffmpeg', fps=20)
    plt.show()
    return ani

# ============================================
# 4. Génération sonore (export WAV)
# ============================================

import wave
import struct

def generate_audio_from_text(text, sample_rate=44100, duration_per_letter=0.2, volume=0.5):
    """
    Crée un signal audio en concaténant une sinusoïde pour chaque lettre.
    La fréquence est donnée par le dictionnaire.
    Chaque lettre dure duration_per_letter secondes (malheureusement indépendante
    de la durée relative, mais on pourrait l'utiliser pour moduler la longueur).
    On exporte un fichier WAV.
    """
    audio = []
    for ch in text.lower():
        freq, duree, typ = params.get(ch, default_param)
        # Nombre d'échantillons pour cette lettre
        n_samples = int(sample_rate * duration_per_letter * duree)
        t = np.linspace(0, duration_per_letter * duree, n_samples, endpoint=False)
        # Sinusoïde
        wave_data = volume * np.sin(2 * np.pi * freq * t)
        # Pour les consonnes, on peut ajouter un léger bruit (facultatif)
        if typ == 'consonne':
            noise = np.random.normal(0, 0.1, len(wave_data))
            wave_data += noise
        audio.extend(wave_data)
    
    audio = np.array(audio, dtype=np.float32)
    # Normalisation
    audio = audio / np.max(np.abs(audio) + 1e-6)
    # Conversion en int16 pour fichier WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open('output.wav', 'wb') as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(sample_rate)
        wavfile.writeframes(audio_int16.tobytes())
    print("Fichier audio généré : output.wav")

# ============================================
# 5. Exemple d'utilisation
# ============================================

if __name__ == "__main__":
    # Un petit texte (par exemple, début de "L'Albatros")
    texte = "Leurs grandes ailes blanches"
    print("Texte :", texte)
    
    # Génération du chemin
    pts = text_to_path(texte, step_length=0.5)
    print("Nombre de points :", len(pts))
    
    # Visualisation statique
    plt.figure()
    plt.plot(pts[:,0], pts[:,1], 'b-o', markersize=2)
    plt.title("Trajectoire du texte")
    plt.axis('equal')
    plt.savefig('trajectoire.png')  # Save instead of show for CLI
    plt.close()  # Close to avoid display issues
    
    # Animation (nécessite ffmpeg pour sauvegarder, sinon commenter save_as)
    # ani = animate_path(pts, interval=100, save_as='albatros.mp4')
    
    # Génération audio (optionnel)
    generate_audio_from_text(texte)