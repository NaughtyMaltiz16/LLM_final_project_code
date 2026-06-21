import argparse, json
import numpy as np
import pandas as pd
from utils import consistency_agreement, semantic_entropy, majority_answer, is_correct


def score_record(rec):
    answers = [s["answer"] for s in rec["samples"]]
    confs = [s["confidence"] for s in rec["samples"] if s["confidence"] is not None]
    lps = [s["mean_logprob"] for s in rec["samples"]
           if s["mean_logprob"] is not None and not np.isnan(s["mean_logprob"])]
    maj = majority_answer(answers)
    correct = is_correct(maj, rec["gold_aliases"]) if rec["answerable"] else None
    return {
        "model": rec["model"], "id": rec["id"], "band": rec["band"],
        "answerable": rec["answerable"], "question": rec["question"],
        "majority_answer": maj,
        "stated_conf": (np.mean(confs) / 100.0) if confs else np.nan,
        "consistency": consistency_agreement(answers),
        "sem_entropy": semantic_entropy(answers),
        "logprob": np.mean(lps) if lps else np.nan,
        "correct": correct,
        "n_samples": len(answers), "n_conf_parsed": len(confs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("-o", "--out", default="scored.csv")
    args = ap.parse_args()
    rows = []
    for path in args.raw:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(score_record(json.loads(line)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"scored {len(df)} rows -> {args.out}")
    print(df.groupby(['model', 'band'])['n_samples'].count())
    # quick health check
    print("\nconfidence parse rate:", round((df['n_conf_parsed'] / df['n_samples']).mean(), 3))
    print("answerable accuracy:")
    print(df[df.answerable].groupby('model')['correct'].mean())


if __name__ == "__main__":
    main()
