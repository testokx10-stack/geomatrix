import numpy as np
import matplotlib.pyplot as plt
import sys

def mot_en_vecteurs(mot, valeurs_lettres, longueur=1.0):
    """
    Trace une ligne brisée pour un mot.
    valeurs_lettres : dict { 'a':1, 'b':2, ... }
    """
    angles = [valeurs_lettres[ch] % 360 for ch in mot if ch in valeurs_lettres]
    points = [(0,0)]
    for ang in angles:
        rad = np.radians(ang)
        x = points[-1][0] + longueur * np.cos(rad)
        y = points[-1][1] + longueur * np.sin(rad)
        points.append((x,y))
    return np.array(points)

# Valeurs simples (a=1,b=2,...)
valeurs = {chr(97+i): i+1 for i in range(26)}  # a->1, b->2, ..., z->26

# Get phrase from command line argument, interactive input, or default
if len(sys.argv) > 1:
    # Use command line argument
    phrase = ' '.join(sys.argv[1:])
elif not sys.stdin.isatty():
    # Stdin is redirected/piped, read from it
    try:
        phrase = sys.stdin.read().strip()
        if not phrase:
            phrase = "le silence est d or"
    except:
        phrase = "le silence est d or"
else:
    # Interactive mode
    try:
        print("Enter a word or phrase to visualize:")
        phrase = input().strip()
        if not phrase:
            phrase = "le silence est d or"
            print(f"Using default phrase: '{phrase}'")
    except EOFError:
        phrase = "le silence est d or"
        print(f"Using default phrase: '{phrase}'")

mots = phrase.split()

plt.figure(figsize=(10,8))
colors = plt.cm.viridis(np.linspace(0, 1, len(mots)))

for i, mot in enumerate(mots):
    pts = mot_en_vecteurs(mot.lower(), valeurs)
    plt.plot(pts[:,0], pts[:,1], color=colors[i], marker='o', label=mot)
    # Silence (espace): on ajoute une ligne pointillée de séparation
    if i < len(mots)-1:
        plt.plot([pts[-1,0], pts[-1,0]+0.5], [pts[-1,1], pts[-1,1]-0.5],
                 'k--', alpha=0.5)

plt.legend()
plt.axis('equal')
plt.title(f"Traduction géométrique de: '{phrase}' (chaque mot = couleur)")
plt.savefig('output.png', dpi=150, bbox_inches='tight')
print("Plot saved as output.png")