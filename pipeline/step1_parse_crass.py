"""
step1_parse_crass.py
Converts CRASS CSV directly into spurious_features_validated.json format.

CRASS structure:
  Premise       -> base real-world fact
  QCC           -> counterfactual question
  CorrectAnswer -> causally correct answer  (ground truth = Yes)
  Answer1       -> wrong answer             (ground truth = No)
  Answer2       -> wrong answer             (ground truth = No)
  PossibleAnswer3 -> wrong answer if present (ground truth = No)

Usage:
  python3 pipeline/step1_parse_crass.py \
      --input  data/crass/crass_raw.csv \
      --output data/crass/spurious_features_validated_crass.json
"""

import csv
import json
import argparse
from pathlib import Path


def parse_crass(input_path, output_path):
    entries = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            pct_id  = row.get("PCTID", "").strip()
            premise = row.get("Premise", "").strip()
            qcc     = row.get("QCC", "").strip()
            correct = row.get("CorrectAnswer", "").strip()
            ans1    = row.get("Answer1", "").strip()
            ans2    = row.get("Answer2", "").strip()
            ans3    = row.get("PossibleAnswer3", "").strip()

            if not premise or not correct:
                continue

            # build spurious features from wrong answers
            # skip "That is not possible." — not a meaningful spurious feature
            spurious = []
            for ans in [ans1, ans2, ans3]:
                if ans and ans.lower().strip(".") != "that is not possible":
                    spurious.append({
                        "description": ans,
                        "causal_judgment": {"verdict": "spurious"}
                    })

            if not spurious:
                continue

            # hypothesis = premise + counterfactual question
            hypothesis = f"{premise} {qcc}".strip()

            entries.append({
                "id":               f"crass_{pct_id}",
                "causal_feature":   correct,
                "hypothesis":       hypothesis,
                "spurious_features": spurious
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    total_spurious = sum(len(e["spurious_features"]) for e in entries)
    print(f"Parsed {len(entries)} CRASS instances")
    print(f"  Causal queries   : {len(entries)}")
    print(f"  Spurious queries : {total_spurious}")
    print(f"  Total queries    : {len(entries) + total_spurious}")
    print(f"Saved -> {output_path}")

    print("\n--- Example entry ---")
    print(json.dumps(entries[0], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/crass/crass_raw.csv")
    parser.add_argument("--output", default="data/crass/spurious_features_validated_crass.json")
    args = parser.parse_args()
    parse_crass(args.input, args.output)


if __name__ == "__main__":
    main()
