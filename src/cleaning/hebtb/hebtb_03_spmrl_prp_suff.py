from pathlib import Path
from src.processing.spmrl import conllu
import re


src_suffix = '02'
dst_suffix = '03'
tb_files = {'train-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/train-hebtb-{}-gold.lattices'.format(src_suffix),
            'dev-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/dev-hebtb-{}-gold.lattices'.format(src_suffix),
            'test-hebtb-gold.lattices': 'data/clean/spmrl/hebtb/test-hebtb-{}-gold.lattices'.format(src_suffix)}

prp_regex = re.compile(r"עצמו\tPRP\tPRP\tgen=(.)\|num=(.)\|per=(.)")
prp_regex_per1 = re.compile(r"עצמו\tPRP\tPRP\tgen=M\|gen=F\|num=(.)\|per=1")
prp_regex_prn = re.compile(r"עצמו\tPRP\tS_PRP")


def save(f, lines):
    for line in lines:
        line = prp_regex.sub(r"עצמו\tPRP\tS_PRN\tsuf_gen=\1|suf_num=\2|suf_per=\3", line)
        line = prp_regex_per1.sub(r"עצמו\tPRP\tS_PRN\tsuf_gen=M|suf_gen=F|suf_num=\1|suf_per=1", line)
        line = prp_regex_prn.sub(r"עצמו\tPRP\tS_PRN", line)
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
