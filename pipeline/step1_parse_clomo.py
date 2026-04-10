import json
from pathlib import Path

CLOMO_PATH = "CLOMO/data"
OUTPUT_FILE = "data/seeds/seeds.json"

def parse_clomo():
    seeds = []
    data_path = Path(CLOMO_PATH)
    for file in sorted(data_path.glob("*.json")):
        with open(file) as f:
            data = json.load(f)
        for item in data:
            input_info = item.get("input_info", {})
            seed = {
                "id":             item.get("id_string", ""),
                "premise":        input_info.get("P", ""),
                "hypothesis":     input_info.get("O", ""),
                "alt_hypothesis": input_info.get("Om", ""),
                "question":       input_info.get("Q", ""),
                "label":          item.get("qtype", "")
            }
            seeds.append(seed)
    return seeds

def main():
    seeds = parse_clomo()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"Saved {len(seeds)} seeds to {OUTPUT_FILE}")
    print(f"Sample: {seeds[0]}")

if __name__ == "__main__":
    main()
