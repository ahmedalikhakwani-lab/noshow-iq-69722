"""
main.py — train the NoShowIQ model and persist everything needed by api.py
Run once before starting the API:  python main.py
"""

import os
import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from noshow_iq.loading_dataset import LoadData
from noshow_iq.preprocess import PreProcess
from noshow_iq.model import ClassificationModel

# ── MongoDB ───────────────────────────────────────────────────────────────────

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client     = MongoClient(MONGO_URI)
db         = client["noshow_iq"]
train_runs = db["training_runs"]

# ── Load ──────────────────────────────────────────────────────────────────────

dataloader = LoadData("KaggleV2-May-2016.csv")
data       = dataloader.data()

# ── Preprocess ────────────────────────────────────────────────────────────────

preprocessor = PreProcess(data=data)
preprocessor.DataInfo()

data = preprocessor.ColNames()

data = preprocessor.TimeCorrection()

data = preprocessor.DropCols(
    cols=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"]
)
data = preprocessor.RemoveRows()

# ── Model ─────────────────────────────────────────────────────────────────────

modeltrainer = ClassificationModel(
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    encoder=LabelEncoder(),
)

# Save neighbourhood encoding map so api.py can replicate inference encoding
neighbourhood_labels = sorted(data["Neighbourhood"].unique())
neighbourhood_map    = {name: idx for idx, name in enumerate(neighbourhood_labels)}
joblib.dump(neighbourhood_map, "neighbourhood_map.joblib")
print(f"[main] Neighbourhood map saved ({len(neighbourhood_map)} entries).")

data = modeltrainer.encode_columns(data, ["Gender", "Neighbourhood", "PartOfDay"])

X, y            = modeltrainer.split_feature(data, "NoShow")
X_test, y_test  = modeltrainer.train(X, y)
acc, report     = modeltrainer.modelevaluation(X_test, y_test)

# ── Persist ───────────────────────────────────────────────────────────────────

modeltrainer.save("noshow_model.joblib")
modeltrainer.log_training_run(len(X), report, train_runs)

client.close()
print("[main] Done — model and neighbourhood map saved, training run logged.")
