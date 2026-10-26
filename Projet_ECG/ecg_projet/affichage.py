import collections
import argparse
import re
import time
import numpy as np
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PREFIXE = "ecg-doua"
parser = argparse.ArgumentParser(description="Affiche le signal ECG d'un patient.")
parser.add_argument("patient_id", nargs="?", default="patient1",
                    help="identifiant MQTT du patient (défaut : patient1)")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.patient_id):
    parser.error("l'identifiant doit contenir 1 à 64 lettres, chiffres, '_' ou '-'")

PATIENT_ID = args.patient_id
TOPIC_RAW = f"{PREFIXE}/{PATIENT_ID}/raw"
TOPIC_BPM = f"{PREFIXE}/{PATIENT_ID}/bpm"

FS = 100                       # identique à l'ESP32
SECONDES = 6                   # durée affichée à l'écran
N = FS * SECONDES
GAP = int(FS * 0.4)            # espace effacé devant le point d'écriture
TARGET = int(FS * 0.6)         # tampon visé (latence ~0,6 s, gage de fluidité)

file = collections.deque(maxlen=FS * 10)
stats = {"recus": 0, "bpm": None}


# ---------------- MQTT ----------------
def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connecté au broker, code :", reason_code)
    client.subscribe(f"{PREFIXE}/{PATIENT_ID}/#")


def on_message(client, userdata, msg):
    texte = msg.payload.decode(errors="ignore").strip()
    if not texte:
        return
    try:
        if msg.topic == TOPIC_RAW:
            valeurs = [int(v) / 1000 for v in texte.split(",")]
            file.extend(valeurs)
            stats["recus"] += len(valeurs)
        elif msg.topic == TOPIC_BPM:
            stats["bpm"] = float(texte)
    except ValueError:
        print("Message ignoré :", texte[:40])


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

# ---------------- Interface ----------------
FOND = "#0b0f14"
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=FOND)
ax.set_facecolor(FOND)
ax.set_xlim(0, N)
ax.set_ylim(-0.5, 1.3)
ax.set_xticks(np.arange(0, N + 1, FS))                  # trait principal : 1 s
ax.set_xticks(np.arange(0, N + 1, FS // 5), minor=True)  # trait fin : 0,2 s
ax.set_yticks(np.arange(-0.5, 1.31, 0.5))
ax.set_yticks(np.arange(-0.5, 1.31, 0.1), minor=True)
ax.grid(which="major", color="#1f3b2d", linewidth=0.9)
ax.grid(which="minor", color="#14251d", linewidth=0.4)
ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
for s in ax.spines.values():
    s.set_visible(False)

fig.text(0.02, 0.95, f"MONITEUR ECG  |  {PATIENT_ID}  |  Dérivation II",
         color="#9fb3a8", fontsize=11, va="top")
fig.text(0.02, 0.02, "25 mm/s simulé  |  grille : 0,2 s",
         color="#5d7367", fontsize=8)

y = np.full(N, np.nan)
(ligne,) = ax.plot(np.arange(N), y, color="#39ff14", linewidth=1.6)
txt_bpm = ax.text(0.985, 0.93, "-- BPM", transform=ax.transAxes, ha="right",
                  va="top", color="#39ff14", fontsize=28, fontweight="bold",
                  animated=True)
txt_etat = ax.text(0.985, 0.76, "CONNEXION...", transform=ax.transAxes,
                   ha="right", va="top", color="#9fb3a8", fontsize=13,
                   animated=True)

etat = {"pos": 0, "frac": 0.0, "lecture": False, "rate_in": float(FS),
        "t": time.time(), "t_rate": time.time(), "n_rate": 0}


def maj(_):
    now = time.time()
    dt = now - etat["t"]
    etat["t"] = now

    # estimation de la vitesse réelle d'arrivée des échantillons (1 fois par s)
    if now - etat["t_rate"] >= 1.0:
        r = (stats["recus"] - etat["n_rate"]) / (now - etat["t_rate"])
        if r > 0:
            etat["rate_in"] = 0.7 * etat["rate_in"] + 0.3 * r
        etat["n_rate"] = stats["recus"]
        etat["t_rate"] = now

    # démarre la lecture quand le tampon est suffisamment rempli
    if not etat["lecture"] and len(file) >= TARGET:
        etat["lecture"] = True
    if etat["lecture"] and len(file) == 0:
        etat["lecture"] = False            # tampon vide : on attend

    if etat["lecture"]:
        erreur = (len(file) - TARGET) / FS
        rate = etat["rate_in"] * (1 + float(np.clip(0.5 * erreur, -0.3, 0.5)))
        etat["frac"] += rate * dt
        k = int(etat["frac"])
        etat["frac"] -= k
        k = min(k, len(file))
        for _ in range(k):
            y[etat["pos"]] = file.popleft()
            etat["pos"] = (etat["pos"] + 1) % N
        y[(etat["pos"] + np.arange(GAP)) % N] = np.nan   # efface devant le curseur
        ligne.set_ydata(y)

    # affichage du BPM et de l'état
    b = stats["bpm"]
    if b is None or not etat["lecture"]:
        txt_bpm.set_text("-- BPM")
        txt_etat.set_text("EN ATTENTE DU SIGNAL")
        txt_etat.set_color("#9fb3a8")
    else:
        txt_bpm.set_text(f"{b:.0f} BPM")
        if b > 120:
            txt_etat.set_text("TACHYCARDIE")
            couleur = "#ff3b30"
        elif b < 50:
            txt_etat.set_text("BRADYCARDIE")
            couleur = "#ff9500"
        else:
            txt_etat.set_text("RYTHME NORMAL")
            couleur = "#39ff14"
        txt_etat.set_color(couleur)
        txt_bpm.set_color(couleur)
        ligne.set_color(couleur)
    return ligne, txt_bpm, txt_etat


ani = animation.FuncAnimation(fig, maj, interval=33, blit=True,
                              cache_frame_data=False)
plt.show()