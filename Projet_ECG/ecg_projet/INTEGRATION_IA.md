# AI integration (Random Forest + summary report)

This folder now uses the AI model of the ECG anomaly detection project (notebooks 01–06, trained on MIT-BIH).
Nothing changes for the ESP32: it still publishes the ECG over MQTT. The AI runs inside `pont.py`.

```
ESP32 (Wokwi) ──MQTT──> pont.py ──Firebase──> dashboard.html
                          │
                          ├─ existing analysis (BPM, irregularity, state)          (unchanged)
                          └─ AI, one monitor per patient (ia/ecg_monitor.py):
                               filter → R peaks → Random Forest on each beat → alerts
                             + summary report every 15 s or on request (ia/ecg_report.py)
```

## What was added

| Where | What |
|---|---|
| `ia/` | `ecg_monitor.py`, `ecg_report.py`, `rf_model.joblib` (copied from the `realtime/` folder of the AI project) |
| `pont.py` | Each patient gets an AI monitor. Every second it is given the new samples. |
| `pont.py` → `/patients/<pid>/alertes` | AI alerts, with `"source": "ia"`: *Battements anormaux (IA)* (attention), *Fréquence cardiaque (IA)* (urgence), *Aucun battement (IA)* (attention) |
| `pont.py` → `/patients/<pid>/live/ia` | `bpm`, `battements`, `anormaux_1min`, `anormaux_total` |
| `pont.py` → `/patients/<pid>/ecg_live` | two new fields: `debut` (number of the first sample since the start) and `anormaux` (positions of the abnormal beats of the last 10 s) |
| `pont.py` → `/patients/<pid>/rapport` | `{html, t, duree_s, battements}`: the summary report, rewritten every 15 s, and within ~2 s when the dashboard asks (`pont.py` checks `rapport_demande` every 2 s) |
| `dashboard.html` | "IA" value in the rhythm strip; abnormal beats drawn **in red** on the curve (requirement F6); a **Rapport de synthèse** panel (requirement F13) |
| `pont.py` | the startup code is now in `main()` (so the file can be imported for tests); behaviour is the same |

## How to run

```bash
cd Projet_ECG/ecg_projet
pip install -r requirements.txt     # use the scikit-learn version printed by Notebook 4
python pont.py                      # needs cle-service.json, as before
```

Then start the Wokwi simulation and open `dashboard.html`, as before.

**Rapport de synthèse panel:**
- Fill in the patient fields (optional), then **Générer le rapport**: a new tab shows the report of everything received so far. Its *Print / Save as PDF* button makes the PDF.
- **Rapport en direct : oui**: a tab that updates itself each time `pont.py` writes a new report (every 15 s).
- The fields are saved in `/patients/<pid>/infos` and the request in `/patients/<pid>/rapport_demande`.

## Firebase rules

The current rules (October 2026) only let the browser **read `patients/patient1`**, and write nothing:

```json
{ "rules": { "patients": { "patient1": { ".read": true } } } }
```

Consequences: the dashboard only works for `patient1` (any other patient ID shows `PERMISSION_DENIED`), and the
dashboard's two small writes are refused. The report buttons still work, but they show the last automatic report
(at most 15 s old), and the patient fields are not saved.

To allow the writes (and keep the same reading rule), change the rules in the Firebase console
(Realtime Database → Rules) to:

```json
{
  "rules": {
    "patients": {
      "patient1": {
        ".read": true,
        "infos": { ".write": true },
        "rapport_demande": { ".write": true }
      }
    }
  }
}
```

Use `"$pid"` instead of `"patient1"` to open every patient ID, but remember that `".read": true` makes these
health data readable by anyone who knows the database address.

Keep `rapport` **read-only for the browser**: the dashboard displays its HTML in a new tab, so nobody except
`pont.py` (the Admin SDK ignores rules) should be able to write it. When the project gets accounts
(requirement F9), replace `true` by a rule on `auth`.

## What to expect from the AI with the simulated ECG

The model was trained on **real** ECGs (MIT-BIH). The ESP32 sends a **synthetic** ECG (a sum of Gaussian curves at 100 Hz).
Tested by reproducing the ESP32 generator in Python (3 minutes per mode):

| ESP32 mode | AI result |
|---|---|
| normal | 0 abnormal beats, regular rhythm |
| extrasystole (blue button) | 44 early beats found out of 45, counted as supraventricular in the report; the Random Forest itself flags none (they have a normal shape: only their timing is abnormal) |
| fibrillation (red button) | rhythm "irregular"; about 1 beat in 7 flagged abnormal, some of them counted as ventricular (false: the generator has no ventricular beats) |

The ESP32 has no "ventricular extrasystole" mode (wide, strange beat), which is what the Random Forest detects best.

## Limits

- Decision support only, not a diagnosis (same as the rest of the project).
- The AI was tested on MIT-BIH (real patients) and on the simulator, never on the AD8232 sensor. Before any real use,
  record real signals with the sensor and check the results (see `PROJECT_GUIDE.md`, section 8, of the AI project).
- The specification asks for 250 Hz; the ESP32 sends 100 Hz. The AI accepts both (each beat is resampled to 360 Hz),
  but 250 Hz gives a more precise beat shape.
- The report covers everything received since `pont.py` started (it is kept in memory). Restarting `pont.py` starts a new report.
