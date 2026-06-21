import json
import random


def load_triviaqa(limit=100, split="validation", seed=0):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    items = []
    for i, ex in enumerate(ds):
        ans = ex["answer"]
        aliases = list(set(ans.get("normalized_aliases", []) + ans.get("aliases", []) + [ans["value"]]))
        items.append({
            "id": f"tqa_{i}",
            "question": ex["question"],
            "gold_aliases": aliases,
            "answerable": True,
            "band": "triviaqa",
        })
    return items


def load_selfaware_unanswerable(limit=30, seed=0):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="ShuoZheLi/SelfAware",
                           filename="SelfAware.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)

    records = data["example"] if isinstance(data, dict) and "example" in data else data

    def is_unanswerable(r):
        v = r.get("answerable", r.get("answer_able"))
        return v in (False, "false", "False", 0)

    unans = [r for r in records if is_unanswerable(r)]
    random.Random(seed).shuffle(unans)
    unans = unans[:limit]

    items = []
    for i, r in enumerate(unans):
        items.append({
            "id": f"sa_{r.get('question_id', i)}",
            "question": r["question"],
            "gold_aliases": None,
            "answerable": False,
            "band": "selfaware_unans",
        })
    return items


def load_all(n_answerable=100, n_unanswerable=30, seed=0):
    return (load_triviaqa(limit=n_answerable, seed=seed)
            + load_selfaware_unanswerable(limit=n_unanswerable, seed=seed))


if __name__ == "__main__":
    items = load_all(5, 5)
    for it in items:
        print(it["band"], "|", it["answerable"], "|", it["question"][:70])
