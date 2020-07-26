from collections import defaultdict
from src.processing.spmrl import treebank as tb
from src.processing.spmrl import lexicon as lex
from src.processing.spmrl.treebank import format_gold_lattice2

src_suffix = '04'
dst_suffix = '05'
tb_files = {'train-hebtb.tokens': 'data/clean/spmrl/hebtb/train-hebtb-{}-tokens.txt'.format('01'),
            'train-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(src_suffix),
            'dev-hebtb.tokens': 'data/clean/spmrl/hebtb/dev-hebtb-{}-tokens.txt'.format('01'),
            'dev-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(src_suffix),
            'test-hebtb.tokens': 'data/clean/spmrl/hebtb/test-hebtb-{}-tokens.txt'.format('01'),
            'test-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(src_suffix)}
lex_files = {'pref-lex': 'data/raw/spmrl/bgulex/bgupreflex_withdef.utf8.hr',
             'lex': 'data/clean/spmrl/bgulex/bgulex-03.hr'}


def save(f, sentences):
    for sentence in sentences:
        lines = format_gold_lattice2(sentence)
        for line in lines:
            f.write(line)
            f.write('\n')
        f.write('\n')


bgulex = lex.Lexicon(lex_files)
hebtb = tb.Treebank(bgulex, tb_files)
for sentences in [hebtb.train_sentences, hebtb.dev_sentences, hebtb.test_sentences]:
    for sent in sentences:
        for i, token in enumerate(sent.tokens):
            token_id = i + 1
            gold_analysis = sent.analysis(token_id)
            if not gold_analysis.hosts:
                continue
            token_analysis = defaultdict(list)
            token_lattice = sent.lattice[token_id]
            for analysis in token_lattice:
                if not analysis.hosts:
                    continue
                host = analysis.hosts[0]
                host_key = (host.form, host.tag)
                token_analysis[host_key].append(analysis)
            gold_host = gold_analysis.hosts[0]
            gold_key = (gold_host.form, gold_host.tag)
            if gold_key not in token_analysis:
                continue
            if len(token_analysis[gold_key]) > 1:
                continue
            analysis = token_analysis[gold_key][0]
            host = analysis.hosts[0]
            if host.tag[:2] != 'yy':
                continue
            gold_host.feats = host.feats
            gold_host.lemma = host.lemma
            if analysis.suffixes and not gold_analysis.suffixes:
                suff = analysis.suffixes[0]
                gold_analysis.suffixes.append(suff)
with open('data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.train_sentences)
with open('data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.dev_sentences)
with open('data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.test_sentences)
