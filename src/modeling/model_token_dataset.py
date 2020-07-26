import pickle
import torch
from torch.utils import data
from src.processing.token_dataset import *
from src.processing.token_vocab import TokenVocab
import multiprocessing as mp


# TensorDataset
class ModelTokenDataset(data.TensorDataset):

    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.inputs = [lattice[1:, lattice_token_column_pos] for lattice in tensors]
        self.lengths = [torch.tensor(lattice[1:, lattice_token_column_pos].nonzero().shape[0]) for lattice in tensors]
        self.outputs = [lattice[1:, lattice_pref_form_column_pos:lattice_suff_feats_column_pos+1] for lattice in tensors]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [(self.inputs[i], self.lengths[i], self.outputs[i]) for i in range(key.start, key.stop, key.step)]
        input_tokens = self.inputs[key]
        seq_length = self.lengths[key]
        output_analyses = self.outputs[key]
        return input_tokens, seq_length, output_analyses

    def __len__(self) -> int:
        return len(self.inputs)

    def save(self, path: Path):
        print(f"Saving token model dataset to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading token model dataset from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)


def to_sentences(inputs: torch.Tensor, outputs: torch.Tensor, vocab: TokenVocab) -> list:
    sent_idx = torch.zeros(inputs.shape, dtype=torch.long)
    token_idx = torch.stack([torch.arange(1, inputs.shape[1]+1, dtype=torch.long) for _ in range(inputs.shape[0])])
    analysis_idx = torch.zeros(inputs.shape, dtype=torch.long)
    is_gold = torch.ones(inputs.shape, dtype=torch.long)
    lattices = torch.cat((outputs, inputs.unsqueeze(dim=2), sent_idx.unsqueeze(dim=2), token_idx.unsqueeze(dim=2),
                          analysis_idx.unsqueeze(dim=2), is_gold.unsqueeze(dim=2)), dim=2)
    return [arr_to_sentence(lattice.numpy(), vocab)for lattice in lattices]


def to_tensor(sent: nlp.Sentence, vocab: TokenVocab) -> torch.Tensor:
    arr = sentence_to_arr(sent, vocab)
    return torch.tensor(arr)


def _arr_to_tensor(token_dataset: TokenDataset, i: int) -> torch.Tensor:
    return torch.tensor(token_dataset[i])


def get_model_token_dataset_partition(home_path: Path, token_dataset: TokenDataset) -> ModelTokenDataset:
    path = home_path / f'data/processed/spmrl/hebtb-token-model-dataset/{token_dataset.name}.pickle'
    if path.exists():
        return ModelTokenDataset.load(path)
    with mp.Pool() as p:
        lattices = p.starmap(_arr_to_tensor, [(token_dataset, i) for i in range(len(token_dataset))])
    mtds = ModelTokenDataset(*lattices)
    mtds.save(path)
    return mtds
