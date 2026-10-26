import collections
import datetime as dt
import json
import re
import sys
import threading
import time
from pathlib import Path
import numpy as np
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import butter, filtfilt, find_peaks

# Modèle IA (Random Forest entraîné sur MIT-BIH) et rapport de synthèse : dossier ia/
DOSSIER_IA = Path(__file__).resolve().parent / "ia"
sys.path.insert(0, str(DOSSIER_IA))
from ecg_monitor import ECGMonitor, load_model      # noqa: E402
from ecg_report import compute_report, render_html  # noqa: E402

PREFIXE = "ecg-doua"          # topics : ecg-doua/<id_patient>/raw, /bpm, /mode, ...
DATABASE_URL = "https://ecg-project-a0448-default-rtdb.europe-west1.firebasedatabase.app"

FS = 100                    # identique à l'ESP32
FENETRE = 6                 # secondes analysées
SEUIL_IRR_HAUT = 0.15       # passage en rythme irrégulier
SEUIL_IRR_BAS = 0.10        # retour au rythme régulier
CONFIRMATIONS = 3           # mesures consécutives avant de changer d'état
DELAI_HORS_LIGNE = 10       # secondes sans signal => patient considéré hors ligne
SUJETS_MQTT = {"raw", "bpm", "mode", "pulse", "spo2", "temperature", "motion"}
SECTEUR_HZ = 50             # fréquence du secteur (filtre coupe-bande de l'IA)
PERIODE_RAPPORT = 15        # secondes entre deux rapports de synthèse automatiques
# alertes de l'IA -> (type affiché, niveau) dans /patients/<pid>/alertes
ALERTES_IA = {"abnormal beats": ("Battements anormaux (IA)", "attention"),
              "heart rate": ("Fréquence cardiaque (IA)", "urgence"),
              "no beat": ("Aucun battement (IA) : vérifier les électrodes", "attention")}

MODELE_IA = load_model(DOSSIER_IA / "rf_model.joblib")
print(f"Modèle IA chargé (seuil d'alarme {MODELE_IA['threshold']:.2f})")


# ---------- Firebase ----------
def initialiser_firebase():
    fichier_cle = Path(__file__).resolve().parent / "cle-service.json"
    if not fichier_cle.is_file():
        raise FileNotFoundError(f"Clé de service Firebase introuvable : {fichier_cle}")
    cred = credentials.Certificate(str(fichier_cle))
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
        # IA : un moniteur par patient (filtre, pics R, Random Forest, alertes)
        self.ia = ECGMonitor(MODELE_IA, fs=FS, powerline_hz=SECTEUR_HZ)
        self.ia_attente = []                 # échantillons pas encore analysés par l'IA
        self.debut = None                    # heure du premier échantillon (pour le rapport)
        self.envoyes = 0                     # échantillons déjà envoyés dans ecg_live
        self.t_rapport = 0.0
        self.t_demande = 0.0
        self.derniere_demande = None         # dernière valeur lue dans rapport_demande

    # Références Firebase : /patients/<pid>/<categorie>
    def ref(self, categorie):
        return db.reference(f"patients/{self.pid}/{categorie}")

    def demande_rapport(self):
        """Le bouton « Générer le rapport » du tableau de bord écrit l'heure dans rapport_demande.

        Lu toutes les 2 s depuis la boucle principale (et non avec listen(), qui ouvre un 2e fil
        de connexion HTTPS : deux fils SSL en même temps font planter Python 3.12.0 sous Windows)."""
        if time.time() - self.t_demande < 2:
            return False
        self.t_demande = time.time()
        try:
            valeur = self.ref("rapport_demande").get()
        except Exception:
            return False
        valeur = valeur if valeur is not None else 0
        nouvelle = self.derniere_demande is not None and valeur != self.derniere_demande
        self.derniere_demande = valeur       # la première lecture sert seulement de référence
        return nouvelle


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
                p.ia_attente.extend(vals)
            if p.debut is None:
                p.debut = dt.datetime.now()
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


def analyser_ia(p, now_ms):
    """Donne la dernière seconde de signal au modèle IA et publie ses alertes."""
    with p.en_attente_lock:
        chunk, p.ia_attente = p.ia_attente, []
    if not chunk:
        return
    for alerte in p.ia.push(np.array(chunk)):
        type_alerte, niveau = ALERTES_IA.get(alerte["type"], (alerte["type"], "attention"))
        bpm = round(p.ia.beats[-1]["heart_rate"]) if p.ia.beats else None
        print(f"[{p.pid}] ALERTE IA : {alerte['message']}")
        try:
            p.ref("alertes").push({
                "type": type_alerte, "niveau": niveau, "bpm": bpm, "source": "ia",
                "message": alerte["message"], "mode_reel": p.recu["mode"], "timestamp": now_ms})
        except Exception as e:
            print(f"[{p.pid}] Erreur Firebase (alerte IA) :", e)


def resume_ia(p):
    """Valeurs de l'IA pour /live : fréquence, battements anormaux sur la dernière minute."""
    battements = list(p.ia.beats)
    recents = [b for b in battements if b["time"] > p.ia.now() - 60]
    return {"bpm": round(battements[-1]["heart_rate"]) if battements else None,
            "battements": len(battements),
            "anormaux_1min": int(sum(b["abnormal"] for b in recents)),
            "anormaux_total": int(sum(b["abnormal"] for b in battements))}


def generer_rapport(p, now_ms):
    """Rapport de synthèse (type Holter) de tout le signal reçu -> /patients/<pid>/rapport."""
    try:
        infos = p.ref("infos").get() or {}
    except Exception:
        infos = {}
    rapport = compute_report(list(p.ia.beats), p.ia.now(), start_time=p.debut, alerts=list(p.ia.alerts))
    patient = {"name": infos.get("nom") or p.pid, "patient_id": p.pid, "dob": infos.get("naissance"),
               "gender": infos.get("sexe"), "physician": infos.get("medecin"),
               "device_id": f"ESP32 ({p.pid})", "lead": "1 lead"}
    try:
        p.ref("rapport").set({"html": render_html(rapport, patient), "t": now_ms,
                              "duree_s": round(p.ia.now()), "battements": rapport["n_beats"]})
    except Exception as e:
        print(f"[{p.pid}] Erreur Firebase (rapport) :", e)


def traiter(p, now_ms):
    en_ligne = time.time() - p.dernier_raw < DELAI_HORS_LIGNE
    demande = p.demande_rapport()
    if not en_ligne:
        if demande:                  # le rapport reste disponible après la fin du signal
            generer_rapport(p, now_ms)
        return  # pas de nouveau signal : on n'écrit rien, le dashboard verra le timestamp ancien

    # 0) IA : classification de chaque battement + alertes, chaque seconde
    analyser_ia(p, now_ms)
    if demande or time.time() - p.t_rapport >= PERIODE_RAPPORT:
        p.t_rapport = time.time()
        generer_rapport(p, now_ms)

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
        # positions (n° d'échantillon depuis le début) des battements jugés anormaux par l'IA
        # dans les 10 dernières secondes : le tableau de bord les colore en rouge sur la courbe
        recents = p.ia.received - 10 * FS
        anormaux = [int(b["position"]) for b in p.ia.beats[-30:] if b["abnormal"] and b["position"] > recents]
        try:
            p.ref("ecg_live").set({"seq": p.seq, "fs": FS, "samples": chunk, "t": now_ms,
                                   "debut": p.envoyes, "anormaux": anormaux})
            p.envoyes += len(chunk)
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
                "temperature": p.recu["temperature"], "motion": p.recu["motion"],
                "ia": resume_ia(p)})
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


def main():
    initialiser_firebase()
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


if __name__ == "__main__":
    main()