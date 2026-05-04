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

# Dictionnaire de ponctuation : (durée_silence, modificateur_amplitude, mod_freq, delta_psi, delta_phi)
PUNCTUATION = {
    ',': (0.2, 1.0, 1.0, 0.0, 0.0),
    '.': (0.4, 1.0, 1.0, 0.0, 0.0),
    ';': (0.3, 1.0, 1.0, 0.0, 0.0),
    ':': (0.25, 1.0, 1.0, 0.0, 0.0),
    '!': (0.1, 1.5, 1.0, 0.0, 0.0),  # amplifie le prochain symbole
    '?': (0.1, 1.0, 1.1, 0.0, 0.1),  # monte la fréquence du précédent (non trivial ici)
    '…': (0.5, 1.0, 1.0, 0.0, 0.0),
    '«': (0.1, 1.0, 1.0, -0.3, 0.0), # léger virage à gauche
    '»': (0.1, 1.0, 1.0, 0.3, 0.0),  # virage à droite
}

# ============================================
# 2. Fonction auxiliaire pour symbol_to_triplet
# ============================================

def symbol_to_triplet(sym, duration):
    if sym in params:
        freq, duree, typ = params[sym]
        if typ == 'voyelle':
            dpsi = (freq - 220) / (880 - 220) * 2 * np.pi
            dphi = 0.0
        else:
            dpsi = np.random.uniform(-0.5, 0.5)  # small random change
            dphi = np.random.uniform(-0.1, 0.1)
        dur = duration
        return freq, dpsi, dphi, dur
    else:
        return 440, 0.0, 0.0, duration

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
# 5. Nouvelle fonction avec ponctuation
# ============================================

def geomatrix_chain_with_punctuation(api_string, step_length=1.0, base_duration=0.1):
    symbols = list(api_string)
    n = len(symbols)
    # Compter les symboles non-punctuation pour les positions
    non_punct_count = sum(1 for s in symbols if s not in PUNCTUATION)
    positions = np.zeros((non_punct_count + 1, 3))
    pos_idx = 0
    positions[pos_idx] = [0.0, 0.0, 0.0]
    psi_cum = 0.0
    phi_cum = 0.0
    sample_rate = 44100
    total_duration = 0.0
    # Première passe pour calculer la durée totale (à cause des silences variables)
    durations = []
    for sym in symbols:
        if sym in PUNCTUATION:
            d = PUNCTUATION[sym][0]
        elif sym.lower() in params or sym in params:
            d = base_duration
        else:
            d = base_duration
        durations.append(d)
        total_duration += d
    
    t = np.linspace(0, total_duration, int(sample_rate * total_duration), endpoint=False)
    signal = np.zeros_like(t)
    
    current_sample = 0
    next_amplitude_mod = 1.0
    last_freq = 440.0  # pour gérer '?'
    
    for idx, sym in enumerate(symbols):
        if sym in PUNCTUATION:
            silence_dur, amp_mod, freq_mod, dpsi, dphi = PUNCTUATION[sym]
            # Appliquer le silence
            end_silence = current_sample + int(silence_dur * sample_rate)
            # Le signal reste à zéro pendant le silence
            current_sample = end_silence
            # Stocker un modificateur pour le prochain symbole
            if sym == '!':
                next_amplitude_mod = amp_mod
            elif sym == '?':
                # Modifier la fréquence du précédent (simplifié : ajuster le dernier freq)
                last_freq *= freq_mod
                # Mais comme le signal est déjà généré, on ne peut pas modifier rétrospectivement
                # Ici, on ignore ou on applique à next, simplifions
                pass
            # Mise à jour angles
            psi_cum += dpsi
            phi_cum += dphi
            continue
        
        # Symbole normal (voyelle/consonne)
        freq, dpsi, dphi, dur = symbol_to_triplet(sym.lower(), durations[idx])
        # Appliquer modificateur d'amplitude
        amp = 0.3 * next_amplitude_mod
        next_amplitude_mod = 1.0  # reset
        
        psi_cum += dpsi
        phi_cum += dphi
        psi_mod = psi_cum % (2*np.pi)
        phi_mod = phi_cum % np.pi
        dx = np.cos(psi_mod) * np.cos(phi_mod)
        dy = np.sin(psi_mod) * np.cos(phi_mod)
        dz = np.sin(phi_mod)
        
        # Mise à jour positions
        last_pos = positions[pos_idx]
        new_pos = last_pos + step_length * np.array([dx, dy, dz])
        pos_idx += 1
        positions[pos_idx] = new_pos
        
        # Génération du signal pour ce symbole
        n_samples = int(dur * sample_rate)
        t_local = np.linspace(0, dur, n_samples, endpoint=False)
        wave_data = amp * np.sin(2 * np.pi * freq * t_local)
        end_sample = current_sample + n_samples
        if end_sample > len(signal):
            # Étendre le signal si nécessaire, mais normalement calculé
            pass
        signal[current_sample:end_sample] += wave_data
        current_sample = end_sample
        last_freq = freq
    
    return positions, signal, t

# ============================================
# 6. Exemple d'utilisation
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
    
    # Nouvelle fonction avec ponctuation
    texte_avec_ponct = "Leurs grandes ailes blanches!"
    positions_3d, signal_audio, t_audio = geomatrix_chain_with_punctuation(texte_avec_ponct)
    print("Positions 3D générées :", positions_3d.shape)
    # Sauvegarder l'audio
    signal_int16 = (signal_audio * 32767).astype(np.int16)
    with wave.open('output_ponct.wav', 'wb') as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(44100)
        wavfile.writeframes(signal_int16.tobytes())
    print("Fichier audio avec ponctuation généré : output_ponct.wav")