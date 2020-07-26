from collections import Counter
from src.processing import nlp


def _token_msr_counts(sentence: nlp.Sentence) -> dict:
    counts = {}
    for token_id in sentence.lattice:
        msr_counts = Counter()
        analysis = sentence.analysis(token_id)
        for m in analysis.prefixes + analysis.hosts + analysis.suffixes:
            msr_counts[m.msr] += 1
        counts[token_id] = msr_counts
    return counts


def _token_seg_tag_counts(sentence: nlp.Sentence) -> dict:
    token_msr_counts = _token_msr_counts(sentence)
    counts = {}
    for token_id in token_msr_counts:
        seg_tag_counts = Counter()
        msr_counts = token_msr_counts[token_id]
        for msr in msr_counts:
            msr_count = msr_counts[msr]
            seg_tag = msr[:-1]
            seg_tag_counts[seg_tag] += msr_count
        counts[token_id] = seg_tag_counts
    return counts


def _token_tag_feats_counts(sentence: nlp.Sentence) -> dict:
    token_msr_counts = _token_msr_counts(sentence)
    counts = {}
    for token_id in token_msr_counts:
        tag_feats_counts = Counter()
        msr_counts = token_msr_counts[token_id]
        for msr in msr_counts:
            msr_count = msr_counts[msr]
            tag_feats = msr[1:]
            tag_feats_counts[tag_feats] += msr_count
        counts[token_id] = tag_feats_counts
    return counts


def _token_seg_counts(sentence: nlp.Sentence) -> dict:
    token_msr_counts = _token_msr_counts(sentence)
    counts = {}
    for token_id in token_msr_counts:
        seg_counts = Counter()
        msr_counts = token_msr_counts[token_id]
        for msr in msr_counts:
            msr_count = msr_counts[msr]
            seg = msr[0]
            seg_counts[seg] += msr_count
        counts[token_id] = seg_counts
    return counts


def _token_tag_counts(sentence: nlp.Sentence) -> dict:
    token_msr_counts = _token_msr_counts(sentence)
    counts = {}
    for token_id in token_msr_counts:
        tag_counts = Counter()
        msr_counts = token_msr_counts[token_id]
        for msr in msr_counts:
            msr_count = msr_counts[msr]
            tag = msr[1]
            tag_counts[tag] += msr_count
        counts[token_id] = tag_counts
    return counts


def _token_feats_counts(sentence: nlp.Sentence) -> dict:
    token_msr_counts = _token_msr_counts(sentence)
    counts = {}
    for token_id in token_msr_counts:
        feats_counts = Counter()
        msr_counts = token_msr_counts[token_id]
        for msr in msr_counts:
            msr_count = msr_counts[msr]
            feats = msr[2]
            feats_counts[feats] += msr_count
        counts[token_id] = feats_counts
    return counts


def _eval(gold_sentence: nlp.Sentence, decoded_sentence: nlp.Sentence, counts_func) -> (dict, dict, dict):
    gold_counts = counts_func(gold_sentence)
    decoded_counts = counts_func(decoded_sentence)
    intersect_counts = {}
    for token_id in gold_sentence.lattice:
        gold_token_counts = gold_counts[token_id]
        if token_id in decoded_counts:
            decoded_token_counts = decoded_counts[token_id]
        else:
            decoded_token_counts = Counter()
        intersect_counts[token_id] = gold_token_counts & decoded_token_counts
    return gold_counts, decoded_counts, intersect_counts


def _eval_sentences(gold: list, decoded: list, counts_func) -> (float, float, float):
    total_gold = 0
    total_decoded = 0
    total_intersect = 0
    for gold_sentence, decoded_sentence in zip(gold, decoded):
        gold_counts, decoded_counts, intersect_counts = _eval(gold_sentence, decoded_sentence, counts_func)
        for token_id in gold_sentence.lattice:
            gold_token_counts = gold_counts[token_id]
            if token_id in decoded_counts:
                decoded_token_counts = decoded_counts[token_id]
            else:
                decoded_token_counts = Counter()
            intersect_token_counts = intersect_counts[token_id]
            total_gold += sum(gold_token_counts.values())
            total_decoded += sum(decoded_token_counts.values())
            total_intersect += sum(intersect_token_counts.values())
    precision = total_intersect / total_decoded if total_decoded else 0.0
    recall = total_intersect / total_gold if total_gold else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def eval_msr(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_msr_counts)


def eval_seg_tag(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_seg_tag_counts)


def eval_tag_feats(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_tag_feats_counts)


def eval_tag(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_tag_counts)


def eval_seg(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_seg_counts)


def eval_feats(gold: list, decoded: list) -> (float, float, float):
    return _eval_sentences(gold, decoded, _token_feats_counts)