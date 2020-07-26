from src.processing import morph
from src.processing.spmrl import treebank as tb
from src.processing.spmrl import lexicon as lex
from src.processing.spmrl.treebank import format_gold_lattice2

src_suffix = '09'
dst_suffix = '10'
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
morph_features = [morph.Gender, morph.Number, morph.Person]
for sentences in [hebtb.train_sentences, hebtb.dev_sentences, hebtb.test_sentences]:
    for n, sent in enumerate(sentences):
        for i, token in enumerate(sent.tokens):
            token_id = i + 1
            gold_analysis = sent.analysis(token_id)
            if not gold_analysis.hosts:
                continue
            if not gold_analysis.suffixes:
                continue
            token_lattice = sent.lattice[token_id]
            analysis_found = None
            gold_host_form = gold_analysis.hosts[-1].form
            for analysis in token_lattice:
                if not analysis.hosts:
                    continue
                if not analysis.suffixes:
                    continue
                gold_analysis.hosts[-1].form = analysis.hosts[-1].form
                if morph.analysis_equals(analysis, gold_analysis, morph_features):
                    analysis_found = analysis
                    break
            if not analysis_found:
                gold_analysis.hosts[-1].form = gold_host_form
                continue
            host = analysis_found.hosts[-1]
            gold_analysis.hosts[0].feats = host.feats
            gold_analysis.hosts[0].lemma = host.lemma
            gold_analysis.hosts[0].form = host.form
with open('data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.train_sentences)
with open('data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.dev_sentences)
with open('data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, hebtb.test_sentences)
