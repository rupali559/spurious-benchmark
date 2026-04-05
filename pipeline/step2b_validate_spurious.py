import json
import argparse
import time
from tqdm import tqdm
from utils import call_llm_json


JUDGE_PROMPT = """
You are evaluating whether a feature is spurious or causal in a reasoning example.

A SPURIOUS feature is one that:
- Correlates with the outcome but does not cause it
- Is a surface-level keyword, style, or phrasing pattern
- Would NOT change the logical conclusion if removed

A CAUSAL feature is one that:
- Is logically necessary for the conclusion
- Removing it would break the argument

Context: {context}
Cause (X): {X}
Outcome (Y): {Y}

Feature being evaluated: {feature}

Is this feature spurious (surface-level correlation) or causal (logically necessary)?

Return ONLY this JSON:
{{
 "verdict": "spurious" or "causal" or "ambiguous",
 "confidence": 0.0 to 1.0,
 "reasoning": "one short sentence"
}}
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    skeletons = json.load(open(args.input_file))

    total_before = 0
    total_after = 0

    for sk in tqdm(skeletons, desc="Validating"):

        context = sk.get("premise", "")
        X = sk.get("causal_feature", "")
        Y = sk.get("hypothesis", "")

        # convert step2 format
        if "spurious_feature" in sk:
            sk["spurious_features"] = [{
                "name": "spurious_feature",
                "description": sk["spurious_feature"]
            }]

        validated_features = []

        for feat in sk.get("spurious_features", []):

            total_before += 1

            prompt = JUDGE_PROMPT.format(
                context=context,
                X=X,
                Y=Y,
                feature=feat["description"]
            )

            try:
                judgment = call_llm_json(prompt)
                feat["causal_judgment"] = judgment

                verdict = judgment.get("verdict", "error")

                if verdict in ["spurious", "ambiguous"]:
                    validated_features.append(feat)
                    total_after += 1

            except Exception as e:
                print("LLM error:", e)

            time.sleep(0.3)

        sk["spurious_features"] = validated_features

    with open(args.output_file, "w") as f:
        json.dump(skeletons, f, indent=2)

    print("\n=== DONE ===")
    print("Features before validation:", total_before)
    print("Features after validation:", total_after)
    print("Output saved to:", args.output_file)


if __name__ == "__main__":
    main()