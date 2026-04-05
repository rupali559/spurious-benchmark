import json
from tqdm import tqdm
from utils import call_llm_json, call_llm

INPUT_FILE = "data/seeds/seeds.json"
OUTPUT_FILE = "data/spurious_features/spurious_features.json"

PROMPT = """Premise: {premise}
Hypothesis: {hypothesis}

Return ONLY this JSON, no explanation:
{{
 "causal_feature": "one short phrase",
 "spurious_feature": "one short phrase"
}}"""

RETRY_PROMPT = """Return ONLY valid JSON like this example:
{{"causal_feature": "X causes Y", "spurious_feature": "Z correlates with Y"}}

Premise: {premise}
Hypothesis: {hypothesis}

JSON:"""


def parse_loose(text):
    """Handle cases where model returns loose text instead of JSON."""
    lines = [l.strip().strip('"').strip("'").strip(',') for l in text.strip().splitlines() if l.strip()]
    # filter out label-like lines
    clean = [l for l in lines if ':' not in l and len(l) > 3]
    if len(clean) >= 2:
        return {"causal_feature": clean[0], "spurious_feature": clean[1]}
    # try splitting on comma
    parts = text.split(',')
    if len(parts) >= 2:
        return {"causal_feature": parts[0].strip().strip('"'), "spurious_feature": parts[1].strip().strip('"')}
    return None


def main():
    seeds = json.load(open(INPUT_FILE))[:50]
    results = []

    for item in tqdm(seeds):
        premise = item["premise"]
        hypothesis = item["hypothesis"]

        response = None

        # first attempt
        try:
            prompt = PROMPT.format(premise=premise, hypothesis=hypothesis)
            response = call_llm_json(prompt)
        except Exception:
            pass

        # retry with stricter prompt
        if not response:
            try:
                prompt2 = RETRY_PROMPT.format(premise=premise, hypothesis=hypothesis)
                response = call_llm_json(prompt2)
            except Exception:
                pass

        # fallback: parse loose output
        if not response:
            try:
                raw = call_llm(PROMPT.format(premise=premise, hypothesis=hypothesis))
                response = parse_loose(raw)
            except Exception:
                pass

        if not response:
            print(f"Skipping {item['id']} — could not extract features")
            continue

        results.append({
            "id": item["id"],
            "premise": premise,
            "hypothesis": hypothesis,
            "causal_feature": response.get("causal_feature", ""),
            "spurious_feature": response.get("spurious_feature", "")
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()