import json
import os

def load_docs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "docs.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
