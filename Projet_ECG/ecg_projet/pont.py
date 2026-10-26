import collections
import json
import re
import threading
import time
from pathlib import Path
import numpy as np
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import butter, filtfilt, find_peaks

PREFIXE = "ecg-doua"          # topics : ecg-doua/<id_patient>/raw, /bpm, /mode, ...
DATABASE_URL = "https://ecg-project-a0448-default-rtdb.europe-west1.firebasedatabase.app"

FS = 100                    # identique à l'ESP32
FENETRE = 6                 # secondes analysées
SEUIL_IRR_HAUT = 0.15       # passage en rythme irrégulier
SEUIL_IRR_BAS = 0.10        # retour au rythme régulier
CONFIRMATIONS = 3           # mesures consécutives avant de changer d'état
DELAI_HORS_LIGNE = 10       # secondes sans signal => patient considéré hors ligne
SUJETS_MQTT = {"raw", "bpm", "mode", "pulse", "spo2", "temperature", "motion"}

# ---------- Firebase ----------
FICHIER_CLE = Path(__file__).resolve().parent / "cle-service.json"
if not FICHIER_CLE.is_file():
    raise FileNotFoundError(f"Clé de service Firebase introuvable : {FICHIER_CLE}")
cred = credentials.Certificate(str(FICHIER_CLE))
firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
print(f"Firebase Admin initialisé pour {DATABASE_URL}")

b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


class Patient:
    """Tout l'état d'un patient : signal, capteurs, état confirmé."""

    def __init__(self, pid):
        self.pid = pid
        self.signal = collections.deque(maxlen=FS * FENETRE)
        self.en_attente = []
        self.en_attente_lock = threading.Lock()
        self.recu = {"bpm_esp": None, "mode": "inconnu", "pulse": None,
                     "spo2": None, "temperature": None, "motion": None}
        self.dernier_raw = 0.0
        self.irr_actif = False
        self.seq = 0
        self.bpm, self.irr = None, 0.0
        self.etat, self.niveau = "En attente", "normal"
        self.candidat, self.compte = None, 0
        self.ecg_firebase_ok = False
        self.live_firebase_ok = False
        self.t_analyse = self.t_histo = time.time()

    # Références Firebase : /patients/<pid>/<categorie>
    def ref(self, categorie):
        return db.reference(f"patients/{self.pid}/{categorie}")


patients = {}
verrou = threading.Lock()


def get_patient(pid):
    with verrou:
        if pid not in patients:
            patients[pid] = Patient(pid)
            print(f"Nouveau patient détecté : {pid}")
        return patients[pid]


# ---------- MQTT ----------
def on_connect(client, userdata, flags, reason_code, properties=None):
    print("MQTT connecté, code :", reason_code)
    if reason_code != 0:
        print("Connexion MQTT refusée :", reason_code)
        return
    resultat, _ = client.subscribe(PREFIXE + "/+/#")
    if resultat != mqtt.MQTT_ERR_SUCCESS:
        print("Échec de l'abonnement MQTT :", resultat)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    print("MQTT déconnecté, code :", reason_code)


def on_message(client, userdata, msg):
    parties = msg.topic.split("/")
    if len(parties) != 3 or parties[0] != PREFIXE:
        return
    _, pid, sujet = parties
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid) or sujet not in SUJETS_MQTT:
        print("Topic MQTT patient invalide ignoré :", msg.topic)
        return
    texte = msg.payload.decode(errors="ignore").strip()
    if not texte:
        return
    p = get_patient(pid)
    try:
        if sujet == "raw":
            vals = [int(v) / 1000 for v in texte.split(",")]
            p.signal.extend(vals)
            with p.en_attente_lock:
                p.en_attente.extend(vals)
            p.dernier_raw = time.time()
        elif sujet == "bpm":
            p.recu["bpm_esp"] = float(texte)
        elif sujet == "mode":
            p.recu["mode"] = texte
        elif sujet == "pulse":
            p.recu["pulse"] = float(texte)
        elif sujet == "spo2":
            p.recu["spo2"] = float(texte)
        elif sujet == "temperature":
            p.recu["temperature"] = float(texte)
        elif sujet == "motion":
            mouvement = json.loads(texte)
            p.recu["motion"] = mouvement if isinstance(mouvement, dict) else None
    except ValueError:
        print(f"[{pid}] Message MQTT invalide sur {sujet} : {texte[:80]}")


# ---------- Analyse ----------
def analyser(x):
    x = np.array(x)
    if len(x) < FS * 5:
        return None
    xf = filtfilt(b, a, x)
    xf = xf - np.mean(xf)
    pics, _ = find_peaks(xf, height=0.5 * np.max(xf), distance=int(0.25 * FS))
    pics = pics[(pics > FS * 0.5) & (pics < len(xf) - FS * 0.5)]
    if len(pics) < 4:
        return None
    rr = np.diff(pics) / FS
    return 60.0 / np.mean(rr), float(np.std(rr) / np.mean(rr))


def etat_brut(p, bpm, irr):
    """Etat instantané, avec hystérésis sur l'irrégularité (propre à chaque patient)."""
    if irr > SEUIL_IRR_HAUT:
        p.irr_actif = True
    elif irr < SEUIL_IRR_BAS:
        p.irr_actif = False
    if p.irr_actif:
        return "Rythme irrégulier", "attention"
    if bpm > 120:
        return "Tachycardie", "urgence"
    if bpm < 50:
        return "Bradycardie", "attention"
    return "Normal", "normal"


def traiter(p, now_ms):
    en_ligne = time.time() - p.dernier_raw < DELAI_HORS_LIGNE
    if not en_ligne:
        return  # pas de nouveau signal : on n'écrit rien, le dashboard verra le timestamp ancien

    # 1) analyse toutes les 2 secondes, avec confirmation de l'état
    if time.time() - p.t_analyse >= 2:
        p.t_analyse = time.time()
        res = analyser(list(p.signal))
        if res:
            p.bpm, p.irr = res
            nouvel_etat, nouveau_niveau = etat_brut(p, p.bpm, p.irr)
            if nouvel_etat == p.candidat:
                p.compte += 1
            else:
                p.candidat, p.compte = nouvel_etat, 1
            if p.compte >= CONFIRMATIONS and p.candidat != p.etat:
                p.etat, p.niveau = p.candidat, nouveau_niveau
                print(f"[{p.pid}] CHANGEMENT D'ETAT : {p.etat} ({p.niveau})")
                if p.niveau != "normal":
                    try:
                        p.ref("alertes").push({
                            "type": p.etat, "niveau": p.niveau,
                            "bpm": round(p.bpm), "mode_reel": p.recu["mode"],
                            "timestamp": now_ms})
                    except Exception as e:
                        print(f"[{p.pid}] Erreur Firebase (alerte) :", e)

    # 2) ECG en direct : un paquet par seconde -> /patients/<pid>/ecg_live
    with p.en_attente_lock:
        chunk = [round(v, 3) for v in p.en_attente]
    if chunk:
        p.seq += 1
        try:
            p.ref("ecg_live").set({"seq": p.seq, "fs": FS, "samples": chunk, "t": now_ms})
            with p.en_attente_lock:
                del p.en_attente[:len(chunk)]
            if not p.ecg_firebase_ok:
                print(f"[{p.pid}] Écriture Firebase ECG OK : patients/{p.pid}/ecg_live")
                p.ecg_firebase_ok = True
        except Exception as e:
            print(f"[{p.pid}] Erreur Firebase (ecg) :", e)

    # 3) valeurs en direct (capteurs inclus) -> /patients/<pid>/live
    if p.bpm is not None:
        try:
            p.ref("live").set({
                "bpm": round(p.bpm), "bpm_esp": p.recu["bpm_esp"],
                "irregularite": round(p.irr, 3),
                "etat": p.etat, "niveau": p.niveau,
                "mode_reel": p.recu["mode"], "timestamp": now_ms,
                "pulse": p.recu["pulse"], "spo2": p.recu["spo2"],
                "temperature": p.recu["temperature"], "motion": p.recu["motion"]})
            if not p.live_firebase_ok:
                print(f"[{p.pid}] Écriture Firebase live OK : patients/{p.pid}/live")
                p.live_firebase_ok = True
        except Exception as e:
            print(f"[{p.pid}] Erreur Firebase (live) :", e)

        # 4) historique : une mesure par minute -> /patients/<pid>/historique
        if time.time() - p.t_histo >= 60:
            p.t_histo = time.time()
            try:
                p.ref("historique").push({
                    "bpm": round(p.bpm), "etat": p.etat, "timestamp": now_ms})
            except Exception as e:
                print(f"[{p.pid}] Erreur Firebase (historique) :", e)

    bpm_txt = "--" if p.bpm is None else round(p.bpm)
    print(f"[{p.pid}] BPM {bpm_txt} | irrég. {p.irr:.3f} | {p.etat} "
          f"| mode réel : {p.recu['mode']} | paquet {p.seq}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

print("Pont multi-patients démarré. Ctrl+C pour arrêter.")
while True:
    time.sleep(1)
    now_ms = int(time.time() * 1000)
    with verrou:
        liste = list(patients.values())
    for p in liste:
        traiter(p, now_ms)