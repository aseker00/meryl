import pickle
import torch
from torch.utils import data
from src.processing.morph_dataset import *
from src.processing.morph_vocab import MorphVocab
import multiprocessing as mp


# TensorDataset
class ModelMorphDataset(data.Dataset):

    def __init__(self, tensors):
        # super().__init__(*tensors)
        self.lattices = tensors

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.lattices[i] for i in range(key.start, key.stop, key.step)]
        return self.lattices[key]

    def __len__(self) -> int:
        return len(self.lattices)

    def save(self, path: Path):
        print(f"Saving morph model dataset to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading morph model dataset from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)


def to_sentence(lattice: torch.Tensor, vocab: MorphVocab) -> nlp.Sentence:
    return arr_to_sentence(lattice.numpy(), vocab)


def to_tensor(sent: nlp.Sentence, vocab: MorphVocab) -> torch.Tensor:
    arr = sentence_to_arr(sent, vocab)
    return torch.tensor(arr)


def _arr_to_tensor(morph_dataset: MorphDataset, i: int) -> torch.Tensor:
    return torch.tensor(morph_dataset[i])


def get_model_morpheme_dataset_partition(home_path: Path, morph_dataset: MorphDataset) -> ModelMorphDataset:
    path = home_path / f'data/processed/spmrl/hebtb-morph-model-dataset/{morph_dataset.name}.pickle'
    if path.exists():
        return ModelMorphDataset.load(path)
    with mp.Pool() as p:
        lattices = p.starmap(_arr_to_tensor, [(morph_dataset, i) for i in range(len(morph_dataset))])
    mmds = ModelMorphDataset(lattices)
    mmds.save(path)
    return mmds
