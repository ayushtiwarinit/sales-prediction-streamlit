import json

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)