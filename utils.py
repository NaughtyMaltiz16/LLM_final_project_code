import string
import numpy as np

_ARTICLES = {"a", "an", "the"}


def normalize_answer(s):
    if s is None:
        return ""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def cluster_answers(answers):
    clusters = {}
    for a in answers:
        k = normalize_answer(a)
        clusters[k] = clusters.get(k, 0) + 1
    return clusters


def consistency_agreement(answers):
    if not answers:
        return 0.0
    return max(cluster_answers(answers).values()) / len(answers)


def semantic_entropy(answers):
    if not answers:
        return 0.0
    counts = np.array(list(cluster_answers(answers).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def majority_answer(answers):
    if not answers:
        return ""
    clusters = cluster_answers(answers)
    top = max(clusters, key=clusters.get)
    for a in answers:
        if normalize_answer(a) == top:
            return a
    return answers[0]


def is_correct(pred, gold_aliases):
    if gold_aliases is None:
        return False
    npred = normalize_answer(pred)
    if not npred:
        return False
    golds = {normalize_answer(g) for g in gold_aliases}
    if npred in golds:
        return True
    return any(g and (g in npred or npred in g) for g in golds)


def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    n, e = len(conf), 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def selective_accuracy_curve(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    cs = correct[order]
    n = len(conf)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(cs) / np.arange(1, n + 1)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return cov, acc, float(trap(acc, cov))


def reliability_bins(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out
