from src.processing.spmrl import treebank as tb
from src.processing.spmrl import lexicon as lex
from src.processing.spmrl.treebank import format_gold_lattice2

src_suffix = '03'
dst_suffix = '04'
tb_files = {'train-hebtb.tokens': 'data/clean/spmrl/hebtb/train-hebtb-{}-tokens.txt'.format('01'),
            'train-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(src_suffix),
            'dev-hebtb.tokens': 'data/clean/spmrl/hebtb/dev-hebtb-{}-tokens.txt'.format('01'),
            'dev-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(src_suffix),
            'test-hebtb.tokens': 'data/clean/spmrl/hebtb/test-hebtb-{}-tokens.txt'.format('01'),
            'test-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(src_suffix)}
lex_files = {'pref-lex': 'data/raw/spmrl/bgulex/bgupreflex_withdef.utf8.hr',
             'lex': 'data/clean/spmrl/bgulex/bgulex-03.hr'}


def save(f, sentences):
    for i, sentence in enumerate(sentences):
        lines = format_gold_lattice2(sentence)
        for line in lines:
            f.write(line)
            f.write('\n')
        f.write('\n')


bgulex = lex.Lexicon(lex_files)
hebtb = tb.Treebank(bgulex, tb_files)
with open('data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.train_sentences)
with open('data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.dev_sentences)
with open('data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.test_sentences)
