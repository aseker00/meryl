from collections import Counter

import nlp


# Macro Average
# Compute metric (F1, Acc) for each class independently, then take the average across all classes.
# Treat all classes equally.

# Micro Average
# Aggregate the contribution of all classes for the metric, then take the average.
# So each class contributes its relative ratio.
# This is preferable in case your labels are unbalanced.


def get_morpheme_counts(sentence: nlp.Sentence) -> (dict, dict):
    seg_pos = {}
    seg_pos_feats = {}
    for token_id in sentence.lattice:
        seg_pos_token_counts = Counter()
        seg_pos_feats_token_counts = Counter()
        analysis = sentence.analysis(token_id)
        for m in analysis.prefixes + analysis.hosts + analysis.suffixes:
            seg_pos_token_counts[m.msr[:-1]] += 1
            seg_pos_feats_token_counts[m.msr] += 1
        seg_pos[token_id] = seg_pos_token_counts
        seg_pos_feats[token_id] = seg_pos_feats_token_counts
    return seg_pos, seg_pos_feats


def evaluate_sentence(gold_sentence: nlp.Sentence, decoded_sentence: nlp.Sentence) -> ((dict, dict, dict),
                                                                                       (dict, dict, dict)):
    seg_pos_intersection = {}
    seg_pos_feats_intersection = {}
    seg_pos_gold, seg_pos_feats_gold = get_morpheme_counts(gold_sentence)
    seg_pos_decoded, seg_pos_feats_decoded = get_morpheme_counts(decoded_sentence)
    for token_id in gold_sentence.lattice:

        seg_pos_gold_token_counts = seg_pos_gold[token_id]
        seg_pos_decoded_token_counts = seg_pos_decoded[token_id]
        seg_pos_intersection[token_id] = seg_pos_gold_token_counts & seg_pos_decoded_token_counts

        seg_pos_feats_gold_token_counts = seg_pos_feats_gold[token_id]
        seg_pos_feats_decoded_token_counts = seg_pos_feats_decoded[token_id]
        seg_pos_feats_intersection[token_id] = seg_pos_feats_gold_token_counts & seg_pos_feats_decoded_token_counts

    return ((seg_pos_gold, seg_pos_decoded, seg_pos_intersection),
            (seg_pos_feats_gold, seg_pos_feats_decoded, seg_pos_feats_intersection))


def evaluate_sentences(gold_sentences: list, decoded_sentences:list) -> ((float, float, float),
                                                                         (float, float, float)):

    num_total_seg_pos_gold = 0
    num_total_seg_pos_decoded = 0
    num_total_seg_pos_intersection = 0

    num_total_seg_pos_feats_gold = 0
    num_total_seg_pos_feats_decoded = 0
    num_total_seg_pos_feats_intersection = 0

    for gold_sentence, decoded_sentence in zip(gold_sentences, decoded_sentences):
        ((seg_pos_gold, seg_pos_decoded, seg_pos_intersection),
         (seg_pos_feats_gold, seg_pos_feats_decoded, seg_pos_feats_intersection)) = evaluate_sentence(gold_sentence,
                                                                                                      decoded_sentence)
        for token_id in gold_sentence.lattice:

            seg_pos_gold_token_counts = seg_pos_gold[token_id]
            seg_pos_decoded_token_counts = seg_pos_decoded[token_id]
            seg_pos_token_intersection = seg_pos_intersection[token_id]
            num_total_seg_pos_gold += sum(seg_pos_gold_token_counts.values())
            num_total_seg_pos_decoded += sum(seg_pos_decoded_token_counts.values())
            num_total_seg_pos_intersection += sum(seg_pos_token_intersection.values())

            seg_pos_feats_gold_token_counts = seg_pos_feats_gold[token_id]
            seg_pos_feats_decoded_token_counts = seg_pos_feats_decoded[token_id]
            seg_pos_feats_token_intersection = seg_pos_feats_intersection[token_id]
            num_total_seg_pos_feats_gold += sum(seg_pos_feats_gold_token_counts.values())
            num_total_seg_pos_feats_decoded += sum(seg_pos_feats_decoded_token_counts.values())
            num_total_seg_pos_feats_intersection += sum(seg_pos_feats_token_intersection.values())

    seg_pos_precision = num_total_seg_pos_intersection / num_total_seg_pos_decoded
    seg_pos_recall = num_total_seg_pos_intersection / num_total_seg_pos_gold
    seg_pos_f1 = 2.0 * (seg_pos_precision * seg_pos_recall) / (seg_pos_precision +
                                                               seg_pos_recall)

    seg_pos_feats_precision = num_total_seg_pos_feats_intersection / num_total_seg_pos_feats_decoded
    seg_pos_feats_recall = num_total_seg_pos_feats_intersection / num_total_seg_pos_feats_gold
    seg_pos_feats_f1 = 2.0 * (seg_pos_feats_precision * seg_pos_feats_recall) / (seg_pos_feats_precision +
                                                                                 seg_pos_feats_recall)

    return ((seg_pos_precision, seg_pos_recall, seg_pos_f1),
            (seg_pos_feats_precision, seg_pos_feats_recall, seg_pos_feats_f1))


def precision(gold: list, pred: list) -> float:
    if not pred:
        return 0.0
    true_positives = len(set(gold).intersection(set(pred)))
    return float(true_positives)/float(len(pred))


def recall(gold: list, pred: list) -> float:
    if not gold:
        return 0.0
    true_positives = len(set(gold).intersection(set(pred)))
    return float(true_positives) / float(len(gold))


def f1(p: float, r: float) -> float:
    return 2.0 * (p * r) / (p + r) if (p + r) else 0.0

# import morph
# import nlp
#
# tokens = 'בבית הלבן'.split()
# m1 = morph.Morpheme('ב', 'ב', 'PREPOSITION', morph.Features())
# m2 = morph.Morpheme('ה', 'ה', 'DET', morph.Features())
# m3 = morph.Morpheme('בית', 'בית', 'NOUN', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR))
# m4 = morph.Morpheme('ה', 'ה', 'DET', morph.Features())
# m5 = morph.Morpheme('לבן', 'לבן', 'JJ', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR))
# m6 = morph.Morpheme('הלבן', 'הלבן', 'JJ', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR))
# lattice = morph.Lattice()
# gold_lattice = morph.Lattice()
# decoded_lattice = morph.Lattice()
# for i in range(len(tokens)):
#     lattice[i + 1] = []
# gold_lattice[1] = [morph.Analysis([m1, m2], [m3], [])]
# gold_lattice[2] = [morph.Analysis([m4], [m5], [])]
# decoded_lattice[1] = [morph.Analysis([m1], [m3], [])]
# decoded_lattice[2] = [morph.Analysis([], [m6], [])]
# gold_sent = nlp.Sentence(tokens, lattice, gold_lattice)
# decoded_sent = nlp.Sentence(tokens, lattice, decoded_lattice)
# (seg_pos_precision, seg_pos_recall, seg_pos_f1), (seg_pos_feats_precision, seg_pos_feats_recall, seg_pos_feats_f1) = evaluate_sentences([gold_sent], [decoded_sent])
# print('seg/pos p {}, r {}, f1 {}'.format(seg_pos_precision, seg_pos_recall, seg_pos_f1))
# print('seg/pos/feats p {}, r {}, f1 {}'.format(seg_pos_feats_precision, seg_pos_feats_recall, seg_pos_feats_f1))
