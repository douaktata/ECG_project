import collections
import argparse
import csv
import re
import time
import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import butter, filtfilt, find_peaks

PREFIXE = "ecg-doua"
parser = argparse.ArgumentParser(description="Analyse le signal ECG d'un patient.")
parser.add_argument("patient_id", nargs="?", default="patient1",
                    help="identifiant MQTT du patient (défaut : patient1)")
parser.add_argument("--plot", action="store_true",
                    help="affiche un graphique après la première analyse")
parser.add_argument("--csv", dest="csv_path",
                    help="chemin du fichier CSV de sortie")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.patient_id):
    parser.error("l'identifiant doit contenir 1 à 64 lettres, chiffres, '_' ou '-'")

PATIENT_ID = args.patient_id
TOPIC_RAW = f"{PREFIXE}/{PATIENT_ID}/raw"
TOPIC_BPM = f"{PREFIXE}/{PATIENT_ID}/bpm"

FS = 100                       # identique à l'ESP32
FENETRE = 10                   # secondes analysées
signal = collections.deque(maxlen=FS * FENETRE)
bpm_esp = {"v": None}
b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connecté au broker, code :", reason_code)
    client.subscribe(f"{PREFIXE}/{PATIENT_ID}/#")


def on_message(client, userdata, msg):
    texte = msg.payload.decode(errors="ignore").strip()
    if not texte:
        return
    try:
        if msg.topic == TOPIC_RAW:
            signal.extend(int(v) / 1000 for v in texte.split(","))
        elif msg.topic == TOPIC_BPM:
            bpm_esp["v"] = float(texte)
    except ValueError:
        pass


def analyser(x):
    x = np.array(x)
    if len(x) < FS * 8:                               # attend au moins 8 s
        return None
    xf = filtfilt(b, a, x)
    xf = xf - np.mean(xf)
    pics, _ = find_peaks(xf, height=0.5 * np.max(xf), distance=int(0.25 * FS))
    pics = pics[(pics > FS * 0.5) & (pics < len(xf) - FS * 0.5)]   # évite les bords
    if len(pics) < 4:
        return None
    rr = np.diff(pics) / FS
    return {"x": x, "xf": xf, "pics": pics, "rr": rr,
            "bpm": 60.0 / np.mean(rr),
            "irr": float(np.std(rr) / np.mean(rr))}


def etat(bpm, irr):
    if irr > 0.15:
        return "IRRÉGULIER"
    if bpm > 120:
        return "TACHYCARDIE"
    if bpm < 50:
        return "BRADYCARDIE"
    return "NORMAL"


def tracer(r):
    import matplotlib.pyplot as plt
    t = np.arange(len(r["x"])) / FS
    fig, (h, bas) = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    h.plot(t, r["x"], color="gray")
    h.set_title("Signal brut reçu")
    bas.plot(t, r["xf"], color="green")
    bas.plot(r["pics"] / FS, r["xf"][r["pics"]], "rv", label="pics R détectés")
    bas.set_title(f"Signal filtré  -  BPM calculé : {r['bpm']:.1f}")
    bas.set_xlabel("Temps (s)")
    bas.legend()
    plt.tight_layout()
    plt.show()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

mode_plot = args.plot
csv_path = args.csv_path or (
    "mesures_bpm.csv" if PATIENT_ID == "patient1"
    else f"mesures_bpm_{PATIENT_ID}.csv")
fichier = open(csv_path, "w", newline="", encoding="utf-8")
csvw = csv.writer(fichier)
csvw.writerow(["heure", "bpm_calcule", "bpm_esp32", "ecart", "irregularite", "nb_pics"])
ecarts = []

print(f"Collecte du signal de {PATIENT_ID} (environ 10 secondes)... Ctrl+C pour arrêter.")
try:
    while True:
        time.sleep(2)
        r = analyser(list(signal))
        if r is None:
            print("Signal insuffisant, patientez...")
            continue
        esp = bpm_esp["v"]
        ecart = abs(r["bpm"] - esp) if esp is not None else None
        if ecart is not None:
            ecarts.append(ecart)
        print(f"calculé {r['bpm']:6.1f} | ESP32 {esp} | écart "
              f"{'?' if ecart is None else f'{ecart:4.1f}'} | "
              f"irrég. {r['irr']:.3f} | {len(r['pics'])} pics | {etat(r['bpm'], r['irr'])}")
        csvw.writerow([time.strftime("%H:%M:%S"), round(r["bpm"], 1), esp,
                       None if ecart is None else round(ecart, 1),
                       round(r["irr"], 3), len(r["pics"])])
        fichier.flush()
        if mode_plot:
            tracer(r)
            mode_plot = False                 # un seul graphique
except KeyboardInterrupt:
    if ecarts:
        print(f"\nÉcart moyen : {np.mean(ecarts):.2f} BPM | maximum : {np.max(ecarts):.2f} BPM "
              f"sur {len(ecarts)} mesures")
    fichier.close()