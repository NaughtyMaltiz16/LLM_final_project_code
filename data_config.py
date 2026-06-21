import json
from data import load_triviaqa, load_selfaware_unanswerable

N_ANSWERABLE = 100
N_UNANSWERABLE = 50

items = load_triviaqa(limit=N_ANSWERABLE) + load_selfaware_unanswerable(limit=N_UNANSWERABLE)

with open("data.json", "w") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

from collections import Counter
print(f"total: {len(items)}")
print(Counter(it["band"] for it in items))
print("\nsample answerable:", next(it["question"] for it in items if it["answerable"]))
print("sample unanswerable:", next(it["question"] for it in items if not it["answerable"]))