import json
import argparse
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import call_llm_json, call_llm

PROMPT = """Premise: {premise}
Hypothesis: {hypothesis}

Generate exactly 3 spurious (fake/correlating but non-causal) features for this argument.
Return ONLY this JSON, no explanation:
{{
 "causal_feature": "one short phrase describing the real cause",
 "spurious_feature_1": "fake correlation 1",
 "spurious_feature_2": "fake correlation 2",
 "spurious_feature_3": "fake correlation 3"
}}"""

RETRY_PROMPT = """Return ONLY valid JSON.
Premise: {premise}
Hypothesis: {hypothesis}
Give 1 real cause and 3 fake correlations:
{{"causal_feature": "X", "spurious_feature_1": "A", "spurious_feature_2": "B", "spurious_feature_3": "C"}}
JSON:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/clomo/seeds.json")
    parser.add_argument("--output", default="data/clomo/spurious_features.json")
    parser.add_argument("--limit",  type=int, default=200)
    args = parser.parse_args()

    seeds = json.load(open(args.input))[:args.limit]
    print(f"Running step2 on {len(seeds)} seeds -> 3 spurious features each...")
    results = []

    for item in tqdm(seeds):
        premise    = item["premise"]
        hypothesis = item["hypothesis"]
        response   = None

        try:
            response = call_llm_json(PROMPT.format(premise=premise, hypothesis=hypothesis))
        except Exception:
            pass

        if not response:
            try:
                raw = call_llm(RETRY_PROMPT.format(premise=premise, hypothesis=hypothesis))
                import re
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    response = json.loads(m.group())
            except Exception:
                pass

        if not response or "causal_feature" not in response:
            continue

        # collect up to 3 spurious features
        spurious = []
        for key in ["spurious_feature_1", "spurious_feature_2", "spurious_feature_3"]:
            val = response.get(key, "").strip()
            if val:
                spurious.append({
                    "description": val,
                    "causal_judgment": {"verdict": "spurious"}
                })

        if not spurious:
            continue

        results.append({
            "id":               item["id"],
            "premise":          premise,
            "hypothesis":       hypothesis,
            "causal_feature":   response["causal_feature"],
            "spurious_features": spurious
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_spurious = sum(len(r["spurious_features"]) for r in results)
    print(f"Saved {len(results)} instances -> {args.output}")
    print(f"  Causal queries   : {len(results)}")
    print(f"  Spurious queries : {total_spurious}")
    print(f"  Total queries    : {len(results) + total_spurious}")

if __name__ == "__main__":
    main()
