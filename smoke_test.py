"""
smoke_test.py
Usage:  python smoke_test.py https://your-space.hf.space
Hits /health, /predict, and /stats. Prints PASS or FAIL for each.
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

SAMPLE = {
    "Gender":         "F",
    "Age":            29,
    "Neighbourhood":  "JARDIM DA PENHA",
    "Scholarship":    0,
    "Hypertension":   0,
    "Diabetes":       0,
    "Alcoholism":     0,
    "Handicap":       0,
    "SMS_received":   1,
    "ScheduledDay":   "2016-04-29T08:00:00Z",
    "AppointmentDay": "2016-05-04T00:00:00Z",
}


def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())


def post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())


def check(label, status, body, required_keys):
    ok = status == 200 and all(k in body for k in required_keys)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}  →  HTTP {status}  |  keys: {list(body.keys())}")
    return ok


results = []

try:
    status, body = get("/health")
    results.append(check("/health", status, body, ["status"]))
except Exception as e:
    print(f"[FAIL] /health  →  {e}"); results.append(False)

try:
    status, body = post("/predict", SAMPLE)
    results.append(check("/predict", status, body,
                         ["risk_level", "probability", "recommendation"]))
except Exception as e:
    print(f"[FAIL] /predict  →  {e}"); results.append(False)

try:
    status, body = get("/stats")
    results.append(check("/stats", status, body,
                         ["total_predictions", "average_probability"]))
except Exception as e:
    print(f"[FAIL] /stats  →  {e}"); results.append(False)

print()
print("Overall:", "PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
