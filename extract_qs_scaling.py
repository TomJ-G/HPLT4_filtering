import os
import json
import re
import random
import math
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import fsolve

# Variables for scoring func
L_wds = 0.5
L_bsc = 1.0
L_reg = 1.0
k_bsc = 2.2
midp_bsc = 2.5

# Variables for UD-sampling
k_main = 3.4


# ----------------------------
# SIZE -> SAMPLING BOUNDS
# Anchor points are in BILLIONS of tokens
TOKEN_ANCHORS_B = np.array([15, 25, 50, 100, 250, 500], dtype=float)
MAX_ANCHORS = np.array([4.0, 4.0, 4.0, 4.0, 2.0, 1.0], dtype=float)
MIN_ANCHORS = np.array([4.0, 2.0, 1.0, 0.005, 0.005, 0.005], dtype=float)

def load_token_count(counts_json_path):
    with open(counts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tokens" not in data:
        raise ValueError(f"'tokens' not found in {counts_json_path}")
    return float(data["tokens"])

def interpolate_ratio(tokens_b, anchor_tokens_b, anchor_values):
    """
    Continuous piecewise-linear interpolation in log-token space.
    This is stable and monotonic, and saturates outside the anchor range.
    """
    x = np.log10(tokens_b)
    xp = np.log10(anchor_tokens_b)
    fp = np.array(anchor_values, dtype=float)
    return float(np.interp(x, xp, fp))

def get_size_ratios(tokens):
    """
    Returns (min_ratio, max_ratio) based on total token count.
    tokens is expected in raw token units, not billions.
    """
    tokens_b = tokens / 1e9
    min_ratio = interpolate_ratio(tokens_b, TOKEN_ANCHORS_B, MIN_ANCHORS)
    max_ratio = interpolate_ratio(tokens_b, TOKEN_ANCHORS_B, MAX_ANCHORS)

    # Safety: ensure the bounds are ordered correctly
    min_ratio = min(min_ratio, max_ratio)
    return min_ratio, max_ratio


# ----------------------------
# SIGMOID CENTERS
# ----------------------------
def get_sigmoid_center(k):
    objective = lambda x0: 4 * (np.tanh(k * (0.25 - x0)) + np.tanh(k * x0)) / (
        np.tanh(k * (1 - x0)) + np.tanh(k * x0)
    ) - 1
    initial_guess = 0.25 if k > 10 else 0.35
    return fsolve(objective, initial_guess)[0]

x0_main = get_sigmoid_center(k_main)

def tanh_ease_in_out(k, x, x0, maxpoint, shift):
    """
    maxpoint = size of the dynamic range
    shift    = lower bound of the range
    """
    return maxpoint * (np.tanh(k * (x - x0)) + np.tanh(k * x0)) / (np.tanh(k * (1 - x0)) + np.tanh(k * x0)) + shift


# REGISTER VARIABLES
REGISTERS = ["dtp", "HI", "HI-IN", "ID", "IN", "IP", "MT", "NA", "ne", "OP", "SP", "LY", "no-label"]

LABEL_HIERARCHY = {
    "MT": [], "LY": [], "SP": ["it"], "ID": [],
    "NA": ["ne", "sr", "nb"], "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"], "IP": ["ds", "ed"],
}
LABEL_PARENT = {c: p for p, cs in LABEL_HIERARCHY.items() for c in cs}

def assign_labels(probabilities, threshold=0.4):
    labels = set()
    for label, prob in probabilities.items():
        if prob >= threshold:
            labels.add(label)
            if label in LABEL_PARENT:
                labels.add(LABEL_PARENT[label])
    return labels

def is_hybrid(labels):
    if len(labels) > 2:
        return True
    if len(labels) == 2:
        l1, l2 = list(labels)
        return not (
            (l1 in LABEL_PARENT and LABEL_PARENT[l1] == l2) or 
            (l2 in LABEL_PARENT and LABEL_PARENT[l2] == l1)
        )
    return False



# PARSING & ADVANCED EVALUATION
def parse_numeric_rule(rule_str):
    rule_str = str(rule_str).strip()
    match = re.match(r"^([>=<!]+)?\s*([\d.-]+)$", rule_str)
    if not match:
        try: return ("==", float(rule_str))
        except ValueError: return None
    op, num_str = match.groups()
    return (op or "==", float(num_str))

def compile_single_block(block_dict):
    """Compiles a dictionary block into execution patterns."""
    compiled = {"propella": {}, "root": {}}
    if not block_dict: return compiled
    
    if "downsample" in block_dict:
        compiled["downsample"] = float(block_dict["downsample"])
        
    if "propella" in block_dict:
        for feat, rules in block_dict["propella"].items():
            if isinstance(rules, str): rules = [rules]
            # Numeric evaluation checks
            if feat == "score_AR" or feat.endswith(("_score", "_value")) or any(isinstance(r, (int, float)) or any(o in str(r) for o in ['>', '<', '=']) for r in rules):
                parsed = [parse_numeric_rule(r) for r in rules if parse_numeric_rule(r)]
                compiled["propella"][feat] = ("numeric", parsed)
            else:
                compiled["propella"][feat] = ("categorical", set(str(r) for r in rules))
                
    for signal, rules in block_dict.items():
        if signal in ["downsample", "propella"]: continue
        if isinstance(rules, (int, float)):
            compiled["root"][signal] = [(">=", float(rules))]
        else:
            if isinstance(rules, str): rules = [rules]
            compiled["root"][signal] = [parse_numeric_rule(r) for r in rules if parse_numeric_rule(r)]
            
    return compiled

def compile_filter_config(config_dict):
    """Handles parsing both flat retro-compatible layouts and structured AND/OR logic maps."""
    if "and" not in config_dict and "or" not in config_dict:
        return {
            "and": compile_single_block(config_dict),
            "or": []
        }
    
    return {
        "and": compile_single_block(config_dict.get("and", {})),
        "or": [compile_single_block(b) for b in config_dict.get("or", [])]
    }

def match_numeric(val, parsed_rules):
    try: val = float(val)
    except (ValueError, TypeError): return False
    for op, num in parsed_rules:
        if op == ">=" and not (val >= num): return False
        elif op == "<=" and not (val <= num): return False
        elif op == ">" and not (val > num): return False
        elif op == "<" and not (val < num): return False
        elif op == "==" and not (val == num): return False
        elif op == "!=" and not (val != num): return False
    return True

def evaluate_block(record, compiled_block):
    """Evaluates whether a single compiled criterion dictionary evaluates to True."""
    # 1. Downsample
    if "downsample" in compiled_block:
        if random.random() >= compiled_block["downsample"]: return False
            
    # 2. Propella features
    if compiled_block["propella"]:
        prop = record.get("propella-4b", {})
        if not prop: return False
        
        #if "score_AR" not in prop and "score_AR" in compiled_block["propella"]:
        #    prop["score_AR"] = calculate_score_ar(prop)
            
        for feat, (filter_type, rules) in compiled_block["propella"].items():
            val = prop.get(feat)
            if val is None: return False
            
            if filter_type == "numeric":
                if not match_numeric(val, rules): return False
            else:
                if isinstance(val, list):
                    if not any(str(v) in rules for v in val): return False
                else:
                    if str(val) not in rules: return False
                        
    # 3. Standard root quality tracks
    if compiled_block["root"]:
        for signal, rules in compiled_block["root"].items():
            val = record.get(signal)
            if val is None or not match_numeric(val, rules): return False
                
    return True

def propella_hq(record):
    # Assign score based on propella quality metrics
    # Since I want to prioritize bsc-edu, propella will affect scores minimally
    # We do not use edu feature because it strongly correlates with bsc-edu and is weaker than bsc-edu
    # content_quality ['excellent':0.1,'good':0.05]
    # time_sensitivity ['evergreen':0.05] # we would like to preserve this type of content
    score = 0.0
    prop = record.get('propella-4b',{})
    cq = prop.get('content_quality',{})
    ed = prop.get('educational_value',{})
    id = prop.get('information_density',{})
    cr = prop.get('content_ratio',{})
    ci = prop.get('content_integrity',{})

    if ed == 'high':
        score += 0.1
    elif ed == 'moderate':
        score += 0.05
    elif ed == 'minimal':
        score -= 0.1
    elif ed == 'none':
        score -= 0.2
    
    if cq == 'excellent':
        score += 0.03
    elif cq == 'good':
        score += 0.015
    elif cq == 'poor':
        score -= 0.2
    elif cq == 'unacceptable':
        score -= 0.5
    
    if cr == 'mostly_navigation':
        score -= 0.2
    elif cr == 'minimal_content':
        score -= 0.5

    if prop.get('time_sensitivity',{}) == 'evergreen':
        score += 0.03
    
    if id == 'thin':
        score -= 0.2
    elif id == 'empty':
        score -= 0.5

    if ci == 'fragment':
        score -= 0.2
    elif ci =='severely_degraded':
        score -= 0.5

    return score

def get_bucket_id(score, max_val):
    if score is None: return "unknown"
    return str(min(int((score / max_val) * 10) + 1, 10))

# Main processing function
def process_file(args):
    input_path, output_dir, filters_json_str, bucket_feature, sampling, min_ratio, max_ratio = args
    kept, total, upsm, dnsm = 0, 0, 0, 0
    out_files = {}

    compiled_filters = compile_filter_config(json.loads(filters_json_str))

    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                total += 1
                try:
                    record = json.loads(line)
                except:
                    continue

                if not evaluate_block(record, compiled_filters["and"]):
                    continue

                if compiled_filters["or"]:
                    passed_or = False
                    for or_block in compiled_filters["or"]:
                        if evaluate_block(record, or_block):
                            passed_or = True
                            break
                    if not passed_or:
                        continue

                if len(record.get("text", "")) < 200:
                    continue

                kept += 1

                WDS = record.get("doc_scores", [0])[0] * 10
                BSC = record.get("bsc-edu", 0)
                probs = record.get("web-register", None)
                propella_hq_Score = propella_hq(record)

                r = assign_labels(probs, 0.4)

                if len(r) == 0:
                    register = "no-label"
                elif is_hybrid(r):
                    if r == {"HI", "IN"}:
                        register = "HI-IN"
                    else:
                        continue
                else:
                    selected = [j for j in r if j in REGISTERS]
                    register = "-".join(sorted(selected))

                    if register in ["NA-ne", "ne-NA"]:
                        register = "ne"
                    if register in ["IN-dtp", "dtp-IN"]:
                        register = "dtp"

                reg_up = 0
                if register in ["HI", "HI-IN", "OP", "dtp"]:
                    reg_up = 0.1

                Score_x = min(
                    0.25
                    + (L_wds * min(0, (WDS - 7) / 2))
                    + (L_bsc * ((math.tanh(k_bsc * (BSC - midp_bsc)) + 1) / 2))
                    + (L_reg * reg_up)
                    + propella_hq_Score,
                    1
                )

                bucket_id = "default"
                if bucket_feature:
                    if bucket_feature in ["fineweb2-hq", "finepdfs-edu", "bsc-edu"]:
                        score = record.get(bucket_feature)
                        bucket_id = get_bucket_id(score, 5.0 if "edu" in bucket_feature else 1.0) if score is not None else "unknown"
                    else:
                        prop = record.get("propella-4b", {})
                        bucket_id = str(prop.get(bucket_feature, "unknown"))

                if bucket_id not in out_files:
                    target_dir = os.path.join(output_dir, bucket_id) if bucket_id != "default" else output_dir
                    os.makedirs(target_dir, exist_ok=True)
                    out_files[bucket_id] = open(os.path.join(target_dir, os.path.basename(input_path)), "w", encoding="utf-8")

                # Sampling ratio
                if sampling == "flat":
                    S = 1.0
                elif sampling == "linear":
                    S = min_ratio + (max_ratio - min_ratio) * Score_x
                elif sampling == "ease_in_out":
                    if min_ratio <= 0.005:
                        shift = 0
                    else:
                        shift = min_ratio
                    S = tanh_ease_in_out(
                        k_main, Score_x, x0_main,
                        maxpoint=(max_ratio - shift),
                        shift=shift
                    )
                else:
                    raise ValueError(f"Unknown sampling mode: {sampling}")

                # Safety clip
                S = float(np.clip(S, min_ratio, max_ratio))

                while S >= 1:
                    out_files[bucket_id].write(json.dumps(record) + "\n")
                    S -= 1
                    upsm += 1

                if random.random() < S:
                    out_files[bucket_id].write(json.dumps(record) + "\n")
                else:
                    dnsm += 1

    finally:
        for f in out_files.values():
            f.close()

    print(f"[DONE] {os.path.basename(input_path)} | kept={kept}/{total} U/D={upsm}/{dnsm}", flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--filters", required=True)
    parser.add_argument("--bucket_feature")
    parser.add_argument("--sampling", required=True, default="ease_in_out")  # flat, linear, ease_in_out
    parser.add_argument("--counts_json", required=True)  # path to counts file

    args = parser.parse_args()
    print(f"Script launched with arguments:\n{args}")

    if os.path.exists(args.filters):
        with open(args.filters, "r", encoding="utf-8") as f:
            filters_json_str = json.dumps(json.load(f))
    else:
        filters_json_str = args.filters

    tokens = load_token_count(args.counts_json)
    min_ratio, max_ratio = get_size_ratios(tokens)

    print(f"[SIZE] tokens={tokens:.3e} -> min_ratio={min_ratio:.4f}, max_ratio={max_ratio:.4f}")

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
    tasks = [
        (
            os.path.join(args.input_dir, f),
            args.output_dir,
            filters_json_str,
            args.bucket_feature,
            args.sampling,
            min_ratio,
            max_ratio,
        )
        for f in files
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        executor.map(process_file, tasks)

if __name__ == "__main__":
    main()
