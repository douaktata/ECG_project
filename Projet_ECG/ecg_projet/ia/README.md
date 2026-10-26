# ia/ : the AI model used by pont.py

Copied from the `realtime/` folder of the ECG anomaly detection project (notebooks 01-06):

- `ecg_monitor.py`: real-time monitor (filter, R-peak detection, Random Forest, alerts).
- `ecg_report.py`: Holter-style summary report (heart rate, HRV, S/V ectopic beats).
- `rf_model.joblib`: the Random Forest trained by Notebook 4 on MIT-BIH.
  It must be loaded with the same scikit-learn version as the one used to train it.

`realtime/` is the reference copy: if you change these files there, copy them here again.
