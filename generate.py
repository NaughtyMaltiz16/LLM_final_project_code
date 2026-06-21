# command line example    python generate.py --model Qwen/Qwen2.5-7B-Instruct --out raw_qwen.jsonl

import argparse, json, re

ANSWER_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)
CONF_RE = re.compile(r"confidence\s*[:\-]\s*(\d{1,3})", re.IGNORECASE)

SYSTEM = ("You are a careful question-answering assistant. Answer with your single best "
          "guess as briefly as possible (a few words), even if you are unsure.")
USER_TMPL = ("Question: {q}\n\nRespond in EXACTLY this format and nothing else:\n"
             "Answer: <your short answer>\n"
             "Confidence: <an integer 0-100 = your probability the answer is correct>")


def build_prompt(tok, q):
    user = SYSTEM + "\n\n" + USER_TMPL.format(q=q)
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_sample(text):
    am, cm = ANSWER_RE.search(text), CONF_RE.search(text)
    if am:
        answer = am.group(1).strip().splitlines()[0].strip()
    else:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        answer = lines[0] if lines else ""
    conf = max(0, min(100, int(cm.group(1)))) if cm else None
    return answer, conf


def mean_token_logprob(comp):
    lps = comp.logprobs
    if not lps:
        return float("nan")
    vals = [lps[i][tid].logprob for i, tid in enumerate(comp.token_ids)
            if lps[i] and tid in lps[i]]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    with open(args.data) as f:
        items = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="auto",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens, logprobs=1)

    prompts = [build_prompt(tok, it["question"]) for it in items]
    outputs = llm.generate(prompts, sp)

    with open(args.out, "w") as f:
        for it, out in zip(items, outputs):
            samples = []
            for comp in out.outputs:
                answer, conf = parse_sample(comp.text)
                samples.append({"raw": comp.text, "answer": answer, "confidence": conf,
                                "mean_logprob": mean_token_logprob(comp)})
            rec = {"model": args.model, "id": it["id"], "band": it["band"],
                   "answerable": it["answerable"], "question": it["question"],
                   "gold_aliases": it["gold_aliases"], "samples": samples}
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(items)} records to {args.out}")


if __name__ == "__main__":
    main()
