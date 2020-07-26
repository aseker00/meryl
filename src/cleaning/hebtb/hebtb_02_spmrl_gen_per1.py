from pathlib import Path
from src.processing.spmrl import conllu
import re


src_suffix = '01'
dst_suffix = '02'
tb_files = {'train-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(src_suffix),
            'dev-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(src_suffix),
            'test-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(src_suffix)}


host_regex = re.compile(r"gen=[MF]\|num=(.)\|per=1")
suff_regex = re.compile(r"suf_gen=[MF]\|suf_num=(.)\|suf_per=1")


def save(f, lines):
    for line in lines:
        line = host_regex.sub(r"gen=M|gen=F|num=\1|per=1", line)
        line = suff_regex.sub(r"suf_gen=M|suf_gen=F|suf_num=\1|suf_per=1", line)
        f.write(line)


train_lines = conllu.read_file(Path(tb_files['train-hebtb-gold.lattices']))
dev_lines = conllu.read_file(Path(tb_files['dev-hebtb-gold.lattices']))
test_lines = conllu.read_file(Path(tb_files['test-hebtb-gold.lattices']))
with open('data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, train_lines)
with open('data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, dev_lines)
with open('data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(dst_suffix), 'w') as f:
    save(f, test_lines)
