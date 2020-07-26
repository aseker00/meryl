import torch
import torch.nn as nn
import torch.nn.functional as F
from src.processing.morph_vocab import MorphVocab
import src.modeling.model_morph_dataset as morph_dataset


class MorphEmbedding(nn.Module):

    def __init__(self, form_emb, lemma_emb, tag_emb, feats_emb):
        super(MorphEmbedding, self).__init__()
        self.form_emb = form_emb
        self.lemma_emb = lemma_emb
        self.tag_emb = tag_emb
        self.feats_emb = feats_emb
        self.embedding_dim = (self.form_emb.embedding_dim + self.lemma_emb.embedding_dim + self.tag_emb.embedding_dim + self.feats_emb.embedding_dim)

    def forward(self, lattice: torch.Tensor):
        forms = lattice[:, :, 0]
        lemmas = lattice[:, :, 1]
        tags = lattice[:, :, 2]
        feats = lattice[:, :, 3:9].contiguous().view(lattice.size(0), -1)
        embedded_forms = self.form_emb(forms)
        embedded_lemmas = self.lemma_emb(lemmas)
        embedded_tags = self.tag_emb(tags)
        embedded_feats = self.feats_emb(feats)
        embedded_feats = embedded_feats.view(lattice.shape[0], lattice.shape[1], -1, embedded_feats.shape[2])
        embedded_feats = embedded_feats.mean(2)
        embedded_lattice = torch.cat([embedded_forms, embedded_lemmas, embedded_tags, embedded_feats], dim=2)
        return embedded_lattice

    def update_form_emb_(self, form_matrix: torch.Tensor):
        cur_form_matrix = self.form_emb.weight
        new_form_matrix = torch.cat([cur_form_matrix, form_matrix], dim=0)
        self.form_emb = nn.Embedding.from_pretrained(new_form_matrix, freeze=False, padding_idx=0)

    def update_lemma_emb_(self, lemma_matrix: torch.Tensor):
        cur_lemma_matrix = self.lemma_emb.weight
        new_lemma_matrix = torch.cat([cur_lemma_matrix, lemma_matrix], dim=0)
        self.lemma_emb = nn.Embedding.from_pretrained(new_lemma_matrix, freeze=False, padding_idx=0)


class AnalysisEmbedding(MorphEmbedding):

    def __init__(self, form_emb, lemma_emb, tag_emb, feats_emb):
        super(AnalysisEmbedding, self).__init__(form_emb, lemma_emb, tag_emb, feats_emb)

    def forward(self, lattice: torch.Tensor):
        forms = lattice[:, :, 0]
        lemmas = lattice[:, :, 1]
        tags = lattice[:, :, 2]
        feats = lattice[:, :, 3:9].contiguous().view(lattice.size(0), -1)
        embedded_forms = self.form_emb(forms).mean(dim=1)
        embedded_lemmas = self.lemma_emb(lemmas).mean(dim=1)
        embedded_tags = self.tag_emb(tags).mean(dim=1)
        embedded_feats = self.feats_emb(feats).mean(dim=1)
        embedded_lattice = torch.cat([embedded_forms, embedded_lemmas, embedded_tags, embedded_feats], dim=1)
        return embedded_lattice


class AnalysisEncoder(nn.Module):

    def __init__(self, emb_dropout, input_size, hidden_size, num_layers, dropout):
        super(AnalysisEncoder, self, ).__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), bidirectional=True)

    def forward(self, embedded_lattice):
        embedded_lattice = self.dropout(embedded_lattice)
        # MorphEmbedding
        # if len(embedded_lattice.shape) == 3:
        #     encoded_lattice, hidden = self.rnn(embedded_lattice)
        #     return encoded_lattice, hidden
        encoded_lattice, hidden = self.rnn(embedded_lattice.unsqueeze(1))
        return encoded_lattice.squeeze(1), hidden


class AnalysisDecoder(nn.Module):

    def __init__(self, emb_dropout, input_size, hidden_size, num_layers, dropout):
        super(AnalysisDecoder, self).__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size*2, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0))

    def forward(self, embedded_analysis, hidden_state):
        embedded_analysis = self.dropout(embedded_analysis)
        # MorphEmbedding
        # if len(embedded_analysis.shape) == 3:
        #     decoded_analysis, hidden_state = self.rnn(embedded_analysis, hidden_state)
        #     return decoded_analysis, hidden_state
        decoded_analysis, hidden_state = self.rnn(embedded_analysis.unsqueeze(1), hidden_state)
        return decoded_analysis.squeeze(1), hidden_state


class Attention(nn.Module) :

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, context, mask):
        scores = torch.mm(context, query.transpose(0, 1)).squeeze(1)
        scores.masked_fill_(mask == 0, -1e10)
        return F.log_softmax(scores, dim=0)


class PtrNetModel(nn.Module):

    def __init__(self, vocab: MorphVocab, emb: MorphEmbedding, encoder: AnalysisEncoder, enc_dropout: float, decoder: AnalysisDecoder, dec_dropout: float, attn: Attention):
        super(PtrNetModel, self).__init__()
        self.vocab = vocab
        self.emb = emb
        self.encoder = encoder
        self.encoder_dropout = nn.Dropout(enc_dropout)
        self.decoder = decoder
        self.decoder_dropout = nn.Dropout(dec_dropout)
        self.attn = attn
        # self.nll = nn.NLLLoss()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, lattice, use_teacher_forcing=False):
        decoded_scores = []
        decoded_pointers = []
        gold_indices = lattice[:, 0, morph_dataset.lattice_is_gold_column_pos]
        missing_gold_indices = gold_indices == -1
        gold_indices = torch.nonzero(gold_indices).squeeze(1)
        decoded_pointers.append(gold_indices[0])
        embedded_lattice = self.emb(lattice)
        encoded_lattice, decoder_hidden_state = self._encode(embedded_lattice)
        embedded_analysis = embedded_lattice[0].unsqueeze(0)
        lattice_token_idx = lattice[:, 0, morph_dataset.lattice_token_idx_column_pos]
        for i in range(1, lattice_token_idx[-1].item() + 1):
            mask = lattice_token_idx == i
            decoded_analysis_weights = self._decode_step(encoded_lattice, embedded_analysis, decoder_hidden_state, mask)
            decoded_analysis_pointer_id = decoded_analysis_weights.argmax()
            decoded_scores.append(decoded_analysis_weights)
            decoded_pointers.append(decoded_analysis_pointer_id)
            if use_teacher_forcing:
                embedded_analysis = embedded_lattice[gold_indices[i].item()].unsqueeze(0)
            else:
                embedded_analysis = embedded_lattice[decoded_analysis_pointer_id.item()].unsqueeze(0)
        pred_score = torch.stack(decoded_scores)
        pred_indices = torch.stack(decoded_pointers)
        if gold_indices.nelement() == 0:
            return pred_indices
        if missing_gold_indices.any():
            missing_gold_indices = missing_gold_indices[gold_indices[1:]]
            # loss = self.nll(pred_score[~missing_gold_indices], gold_indices[1:][~missing_gold_indices])
            loss = self.cel(pred_score[~missing_gold_indices], gold_indices[1:][~missing_gold_indices])
        else:
            # loss = self.nll(pred_score, gold_indices[1:])
            loss = self.cel(pred_score, gold_indices[1:])
        return pred_indices, loss

    def decode(self, lattice):
        decoded_pointers = []
        embedded_lattice = self.emb(lattice)
        encoded_lattice, decoder_hidden_state = self._encode(embedded_lattice)
        embedded_analysis = embedded_lattice[0].unsqueeze(0)
        lattice_token_idx = lattice[:, 0, morph_dataset.lattice_token_idx_column_pos]
        for i in range(1, lattice_token_idx[-1].item() + 1):
            mask = lattice_token_idx == i
            decoded_analysis_weights = self._decode_step(encoded_lattice, embedded_analysis, decoder_hidden_state, mask)
            decoded_analysis_pointer_id = decoded_analysis_weights.argmax()
            decoded_pointers.append(decoded_analysis_pointer_id)
            embedded_analysis = embedded_lattice[decoded_analysis_pointer_id.item()].unsqueeze(0)
        return torch.stack(decoded_pointers)

    def _encode(self, embedded_lattice):
        encoded_lattice, encoder_hidden_state = self.encoder(embedded_lattice)
        encoded_lattice = self.encoder_dropout(encoded_lattice)
        encoder_h = encoder_hidden_state[0].view(self.encoder.rnn.num_layers, 2, -1, self.encoder.rnn.hidden_size)
        encoder_c = encoder_hidden_state[1].view(self.encoder.rnn.num_layers, 2, -1, self.encoder.rnn.hidden_size)
        decoder_h = encoder_h[-self.encoder.rnn.num_layers:, :, -1].contiguous().view(self.decoder.rnn.num_layers, 1, -1)
        decoder_c = encoder_c[-self.encoder.rnn.num_layers:, :, -1].contiguous().view(self.decoder.rnn.num_layers, 1, -1)
        decoder_hidden_state = (decoder_h, decoder_c)
        return encoded_lattice, decoder_hidden_state

    def _decode_step(self, encoded_lattice, embedded_analysis, decoder_hidden_state, mask):
        decoded_analysis, decoder_hidden_state = self.decoder(embedded_analysis, decoder_hidden_state)
        decoded_analysis = self.decoder_dropout(decoded_analysis)
        return self.attn(decoded_analysis, encoded_lattice, mask)
