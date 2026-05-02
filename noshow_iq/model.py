from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from datetime import datetime, timezone


class ClassificationModel:

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    # SYNTAX FIX: Method was named 'encoder' which silently shadowed the self.encoder
    # instance attribute set in __init__. Renamed to encode_columns to avoid the clash.
    def encode_columns(self, df, columns_to_encode):
        # LOGICAL FIX: LabelEncoder can only process one column at a time.
        # Original code passed a list of columns to fit_transform() in one call,
        # which would produce wrong results or raise an error.
        # Fixed by looping over each column and encoding it separately.
        for col in columns_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df

    def split_feature(self, df, targetcolumn):
        # SYNTAX FIX: df.drop(columns=df[targetcolumn]) passed a full Series to columns=.
        # Fixed to pass the column name string directly using [targetcolumn].
        # LOGICAL FIX: Wrapping targetcolumn in a list here was consistent with the
        # original intent — dropping one named target column by string name.
        X = df.drop(columns=[targetcolumn])
        y = df[targetcolumn]

        return X, y

    def train(self, X, y):
        # SYNTAX FIX: train_test_split returns (X_train, X_test, y_train, y_test).
        # Original had (X_train, y_train, X_test, y_test) — X_test and y_train were swapped,
        # meaning the model would train on labels and evaluate on training features silently.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=11
        )

        # LOGICAL FIX: No class imbalance handling existed. stratify=y only preserves
        # the ratio in the split — it does not fix imbalance during training.
        # class_weight='balanced' tells RandomForest to penalise misclassifying
        # the minority class (no-show) proportionally to its rarity.
        # This is required by the exam and is critical for a useful model.
        self.model.set_params(class_weight='balanced')
        self.model.fit(X_train, y_train)

        return X_test, y_test

    # SYNTAX FIX: modelevaluation() was defined with (self, X_test, y_test) as parameters
    # but was called with only (X_test) in main.py — y_test was missing.
    # The definition is correct here; the call in main.py has been fixed.
    
    def modelevaluation(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)

        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        return acc, report
    
    def save(self, path: str = "noshow_model.joblib"):
        joblib.dump(self.model, path)
        print(f"[model] Saved to {path}")

    @classmethod
    def load(cls, path: str = "noshow_model.joblib"):
        loaded_model = joblib.load(path)
        instance = cls(model=loaded_model, encoder=LabelEncoder())
        print(f"[model] Loaded from {path}")
        return instance

    def log_training_run(self, train_size: int, report: dict, mongo_collection):
        def _metrics(label: str):
            entry = report.get(str(label), {})
            return {
                "precision": round(entry.get("precision", 0.0), 4),
                "recall":    round(entry.get("recall",    0.0), 4),
                "f1":        round(entry.get("f1-score",  0.0), 4),
            }

        doc = {
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "training_size":       train_size,
            "imbalance_technique": "class_weight=balanced (RandomForest)",
            "class_0_show":        _metrics("0"),
            "class_1_noshow":      _metrics("1"),
            "accuracy":            round(report.get("accuracy", 0.0), 4),
        }
        mongo_collection.insert_one(doc)
        print("[model] Training run logged to MongoDB.")
