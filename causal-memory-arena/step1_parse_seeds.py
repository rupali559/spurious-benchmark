import json
from pathlib import Path

CLOMO_PATH = "../CLOMO/data"
OUTPUT_FILE = "data/seeds/seeds.json"

def parse_clomo():
    seeds = []
    data_path = Path(CLOMO_PATH)
    for file in data_path.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
        for i, item in enumerate(data):
            input_info = item.get("input_info", {})
            seed = {
                "id": item.get("id_string", f"{file.stem}_{i}"),
                "premise": input_info.get("P", ""),   # the argument/statement
                "hypothesis": input_info.get("O", ""), # the correct assumption
                "alt_hypothesis": input_info.get("Om", ""), # the wrong assumption
                "label": item.get("qtype", "")
            }
            seeds.append(seed)
    return seeds

def main():
    seeds = parse_clomo()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"Saved {len(seeds)} seeds to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()