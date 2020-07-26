from pathlib import Path
from src.processing.spmrl import conllu
import re


clean_data_file_path = Path('data/raw/spmrl/hebtb/spmrl_fixed.conllu')
lattice_sentences = conllu.read_conllu_sentences(clean_data_file_path, 'spmrl')
dev_lattice_sentences = lattice_sentences[:500]
train_lattice_sentences = lattice_sentences[500:(500+4937)]
test_lattice_sentences = lattice_sentences[(500+4937):]


lattice_line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'
def format_lattice_line(from_node_id: int, node: dict) -> str:
    return lattice_line.format(from_node_id, from_node_id + 1, node['form'], node['lemma'], node['cpostag'], node['postag'], node['feats_str'], node['misc']['token_id'])


dups = []
for i, sent in enumerate(lattice_sentences):
    sent_id = sent['id']
    sim_ids = sent['very_similar_ids']
    if sim_ids:
        sim_ids.add(sent_id)
        dup_match = 0
        for dup in dups:
            if len(dup.intersection(sim_ids)) > 0:
                dup.update(sim_ids)
                dup_match += 1
        if dup_match == 0:
            dups.append(sim_ids)

train_dedups = set()
train_sentences = []
for i, sent in enumerate(train_lattice_sentences):
    sent_id = sent['id']
    sent_dup = [dup for dup in dups if sent_id in dup]
    tokens = [token_node[0]['misc']['token_str'] for token_node in sent['token_nodes']]
    lattice_lines = [format_lattice_line(j, node) for j, node in enumerate(sent['nodes'])]
    if sent_dup:
        if len(sent_dup[0].intersection(train_dedups)) > 0:
            print('train dedup {}, {}: {}'.format('train',  sent_dup[0],  ' '.join(tokens)))
            continue
        train_dedups.add(sent_id)
    train_sentences.append((tokens, lattice_lines))


dev_dedups = set()
dev_sentences = []
for i, sent in enumerate(dev_lattice_sentences):
    sent_id = sent['id']
    sent_dup = [dup for dup in dups if sent_id in dup]
    tokens = [token_node[0]['misc']['token_str'] for token_node in sent['token_nodes']]
    lattice_lines = [format_lattice_line(j, node) for j, node in enumerate(sent['nodes'])]
    if sent_dup:
        if len(sent_dup[0].intersection(train_dedups)) > 0:
            print('dev dedup {}, {}: {}'.format('train',  sent_dup[0],  ' '.join(tokens)))
            continue
        if len(sent_dup[0].intersection(dev_dedups)) > 0:
            print('dev dedup {}, {}: {}'.format('dev',  sent_dup[0],  ' '.join(tokens)))
            continue
        dev_dedups.add(sent_id)
    dev_sentences.append((tokens, lattice_lines))

test_dedups = set()
test_sentences = []
for i, sent in enumerate(test_lattice_sentences):
    sent_id = sent['id']
    sent_dup = [dup for dup in dups if sent_id in dup]
    tokens = [token_node[0]['misc']['token_str'] for token_node in sent['token_nodes']]
    lattice_lines = [format_lattice_line(j, node) for j, node in enumerate(sent['nodes'])]
    if sent_dup:
        if len(sent_dup[0].intersection(train_dedups)) > 0:
            print('test dedup {}, {}: {}'.format('train',  sent_dup[0],  ' '.join(tokens)))
            continue
        if len(sent_dup[0].intersection(dev_dedups)) > 0:
            print('test dedup {}, {}: {}'.format('dev',  sent_dup[0],  ' '.join(tokens)))
            continue
        if len(sent_dup[0].intersection(test_dedups)) > 0:
            print('test dedup {}, {}: {}'.format('test',  sent_dup[0],  ' '.join(tokens)))
            continue
        test_dedups.add(sent_id)
    test_sentences.append((tokens, lattice_lines))


def save_tokens(f, sentences):
    for sentence in sentences:
        for line in sentence[0]:
            f.write(line)
            f.write('\n')
        f.write('\n')


def save_lattices(f, sentences):
    for sentence in sentences:
        for line in sentence[1]:
            line = line.replace('HebBinyan', 'binyan')
            f.write(line)
            f.write('\n')
        f.write('\n')


with open('data/clean/spmrl/hebtb/train-hebtb-01-tokens.txt', 'w') as f:
    save_tokens(f, train_sentences)
with open('data/clean/spmrl/hebtb/train-hebtb-01-gold.lattices', 'w') as f:
    save_lattices(f, train_sentences)
with open('data/clean/spmrl/hebtb/dev-hebtb-01-tokens.txt', 'w') as f:
    save_tokens(f, dev_sentences)
with open('data/clean/spmrl/hebtb/dev-hebtb-01-gold.lattices', 'w') as f:
    save_lattices(f, dev_sentences)
with open('data/clean/spmrl/hebtb/test-hebtb-01-tokens.txt', 'w') as f:
    save_tokens(f, test_sentences)
with open('data/clean/spmrl/hebtb/test-hebtb-01-gold.lattices', 'w') as f:
    save_lattices(f, test_sentences)
