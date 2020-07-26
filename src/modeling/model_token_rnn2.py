import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.processing.token_vocab import TokenVocab


class TagRNN2(nn.Module):

    def __init__(self, token_emb, emb_dropout, hidden_size, num_layers, hidden_dropout, rnn_dropout, class_dropout, vocab:  TokenVocab):
        super(TagRNN2, self).__init__()
        self.token_emb = token_emb
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(input_size=token_emb.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=(hidden_dropout if num_layers > 1 else 0), bidirectional=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.class_dropout = nn.Dropout(class_dropout)
        self.vocab = vocab
        self.cel = nn.CrossEntropyLoss()
        self.num_pref_forms = len(vocab.pref_forms)
        self.num_pref_tags = len(vocab.pref_tags)
        self.num_pref_feats = len(vocab.pref_feats)
        # self.num_host_forms = len(vocab.host_forms)
        self.num_host_tags = len(vocab.host_tags)
        self.num_host_feats = len(vocab.host_feats)
        self.num_suff_forms = len(vocab.suff_forms)
        self.num_suff_tags = len(vocab.suff_tags)
        self.num_suff_feats = len(vocab.suff_feats)
        self.pref_form_classifier = nn.Linear(2 * hidden_size, self.num_pref_forms)
        self.pref_tag_classifier = nn.Linear(2 * hidden_size, self.num_pref_tags)
        self.pref_feats_classifier = nn.Linear(2 * hidden_size, self.num_pref_feats)
        # self.host_form_classifier = nn.Linear(2 * hidden_size, self.num_host_forms)
        self.host_tag_classifier = nn.Linear(2 * hidden_size, self.num_host_tags)
        self.host_feats_classifier = nn.Linear(2 * hidden_size, self.num_host_feats)
        self.suff_form_classifier = nn.Linear(2 * hidden_size, self.num_suff_forms)
        self.suff_tag_classifier = nn.Linear(2 * hidden_size, self.num_suff_tags)
        self.suff_feats_classifier = nn.Linear(2 * hidden_size, self.num_suff_feats)

    def forward(self, token_seq: torch.Tensor, seq_lengths: torch.Tensor, gold_morph_seq: torch.Tensor):
        sorted_seq_lengths, sorted_seq_idx = seq_lengths.sort(0, descending=True)
        sorted_token_seq = token_seq[sorted_seq_idx]
        embedded_tokens = self.token_emb(sorted_token_seq)
        embedded_tokens = self.emb_dropout(embedded_tokens)
        packed_token_seq = pack_padded_sequence(embedded_tokens, sorted_seq_lengths, batch_first=True)
        packed_encoded_tokens, (_, _) = self.rnn(packed_token_seq)
        encoded_tokens, _ = pad_packed_sequence(packed_encoded_tokens, batch_first=True)
        encoded_tokens = self.rnn_dropout(encoded_tokens)

        pref_form_class_scores = self.pref_form_classifier(encoded_tokens)
        pref_tag_class_scores = self.pref_tag_classifier(encoded_tokens)
        pref_feats_class_scores = self.pref_feats_classifier(encoded_tokens)
        # host_form_class_scores = self.host_form_classifier(encoded_tokens)
        host_tag_class_scores = self.host_tag_classifier(encoded_tokens)
        host_feats_class_scores = self.host_feats_classifier(encoded_tokens)
        suff_form_class_scores = self.suff_form_classifier(encoded_tokens)
        suff_tag_class_scores = self.suff_tag_classifier(encoded_tokens)
        suff_feats_class_scores = self.suff_feats_classifier(encoded_tokens)

        pref_form_class_scores = self.class_dropout(pref_form_class_scores)
        pref_tag_class_scores = self.class_dropout(pref_tag_class_scores)
        pref_feats_class_scores = self.class_dropout(pref_feats_class_scores)
        # host_form_class_scores = self.class_dropout(host_form_class_scores)
        host_tag_class_scores = self.class_dropout(host_tag_class_scores)
        host_feats_class_scores = self.class_dropout(host_feats_class_scores)
        suff_form_class_scores = self.class_dropout(suff_form_class_scores)
        suff_tag_class_scores = self.class_dropout(suff_tag_class_scores)
        suff_feats_class_scores = self.class_dropout(suff_feats_class_scores)

        packed_pref_form_class_scores = pack_padded_sequence(pref_form_class_scores, sorted_seq_lengths, batch_first=True)
        packed_pref_tag_class_scores = pack_padded_sequence(pref_tag_class_scores, sorted_seq_lengths, batch_first=True)
        packed_pref_feats_class_scores = pack_padded_sequence(pref_feats_class_scores, sorted_seq_lengths, batch_first=True)
        # packed_host_form_class_scores = pack_padded_sequence(host_form_class_scores, sorted_seq_lengths, batch_first=True)
        packed_host_tag_class_scores = pack_padded_sequence(host_tag_class_scores, sorted_seq_lengths, batch_first=True)
        packed_host_feats_class_scores = pack_padded_sequence(host_feats_class_scores, sorted_seq_lengths, batch_first=True)
        packed_suff_form_class_scores = pack_padded_sequence(suff_form_class_scores, sorted_seq_lengths, batch_first=True)
        packed_suff_tag_class_scores = pack_padded_sequence(suff_tag_class_scores, sorted_seq_lengths, batch_first=True)
        packed_suff_feats_class_scores = pack_padded_sequence(suff_feats_class_scores, sorted_seq_lengths, batch_first=True)

        sorted_gold_pref_form_seq = gold_morph_seq[:, :, 0][sorted_seq_idx]
        sorted_gold_pref_tag_seq = gold_morph_seq[:, :, 1][sorted_seq_idx]
        sorted_gold_pref_feats_seq = gold_morph_seq[:, :, 2][sorted_seq_idx]
        sorted_gold_host_form_seq = gold_morph_seq[:, :, 3][sorted_seq_idx]
        sorted_gold_host_tag_seq = gold_morph_seq[:, :, 4][sorted_seq_idx]
        sorted_gold_host_feats_seq = gold_morph_seq[:, :, 5][sorted_seq_idx]
        sorted_gold_suff_form_seq = gold_morph_seq[:, :, 6][sorted_seq_idx]
        sorted_gold_suff_tag_seq = gold_morph_seq[:, :, 7][sorted_seq_idx]
        sorted_gold_suff_feats_seq = gold_morph_seq[:, :, 8][sorted_seq_idx]

        packed_gold_pref_form_seq = pack_padded_sequence(sorted_gold_pref_form_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_pref_tag_seq = pack_padded_sequence(sorted_gold_pref_tag_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_pref_feats_seq = pack_padded_sequence(sorted_gold_pref_feats_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_host_form_seq = pack_padded_sequence(sorted_gold_host_form_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_host_tag_seq = pack_padded_sequence(sorted_gold_host_tag_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_host_feats_seq = pack_padded_sequence(sorted_gold_host_feats_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_suff_form_seq = pack_padded_sequence(sorted_gold_suff_form_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_suff_tag_seq = pack_padded_sequence(sorted_gold_suff_tag_seq, sorted_seq_lengths, batch_first=True)
        packed_gold_suff_feats_seq = pack_padded_sequence(sorted_gold_suff_feats_seq, sorted_seq_lengths, batch_first=True)

        pref_form_loss = self.cel(packed_pref_form_class_scores.data, packed_gold_pref_form_seq.data)
        pref_tag_loss = self.cel(packed_pref_tag_class_scores.data, packed_gold_pref_tag_seq.data)
        pref_feats_loss = self.cel(packed_pref_feats_class_scores.data, packed_gold_pref_feats_seq.data)
        # host_form_loss = self.cel(packed_host_form_class_scores.data, packed_gold_host_form_seq.data)
        host_tag_loss = self.cel(packed_host_tag_class_scores.data, packed_gold_host_tag_seq.data)
        host_feats_loss = self.cel(packed_host_feats_class_scores.data, packed_gold_host_feats_seq.data)
        suff_form_loss = self.cel(packed_suff_form_class_scores.data, packed_gold_suff_form_seq.data)
        suff_tag_loss = self.cel(packed_suff_tag_class_scores.data, packed_gold_suff_tag_seq.data)
        suff_feats_loss = self.cel(packed_suff_feats_class_scores.data, packed_gold_suff_feats_seq.data)

        gold_pref_forms, _ = pad_packed_sequence(packed_gold_pref_form_seq, batch_first=True)
        gold_pref_tags, _ = pad_packed_sequence(packed_gold_pref_tag_seq, batch_first=True)
        gold_pref_feats, _ = pad_packed_sequence(packed_gold_pref_feats_seq, batch_first=True)
        gold_host_forms, _ = pad_packed_sequence(packed_gold_host_form_seq, batch_first=True)
        gold_host_tags, _ = pad_packed_sequence(packed_gold_host_tag_seq, batch_first=True)
        gold_host_feats, _ = pad_packed_sequence(packed_gold_host_feats_seq, batch_first=True)
        gold_suff_forms, _ = pad_packed_sequence(packed_gold_suff_form_seq, batch_first=True)
        gold_suff_tags, _ = pad_packed_sequence(packed_gold_suff_tag_seq, batch_first=True)
        gold_suff_feats, _ = pad_packed_sequence(packed_gold_suff_feats_seq, batch_first=True)

        pred_pref_forms = pref_form_class_scores.argmax(dim=2)
        pred_pref_tags = pref_tag_class_scores.argmax(dim=2)
        pred_pref_feats = pref_feats_class_scores.argmax(dim=2)
        # pred_host_forms = host_form_class_scores.argmax(dim=2)
        pred_host_tags = host_tag_class_scores.argmax(dim=2)
        pred_host_feats = host_feats_class_scores.argmax(dim=2)
        pred_suff_forms = suff_form_class_scores.argmax(dim=2)
        pred_suff_tags = suff_tag_class_scores.argmax(dim=2)
        pred_suff_feats = suff_feats_class_scores.argmax(dim=2)

        preds = torch.stack([pred_pref_forms, pred_pref_tags, pred_pref_feats, gold_host_forms, pred_host_tags, pred_host_feats, pred_suff_forms, pred_suff_tags, pred_suff_feats], dim=0)
        golds = torch.stack([gold_pref_forms, gold_pref_tags, gold_pref_feats, gold_host_forms, gold_host_tags, gold_host_feats, gold_suff_forms, gold_suff_tags, gold_suff_feats], dim=0)
        losses = torch.stack([pref_form_loss, pref_tag_loss, pref_feats_loss, host_tag_loss, host_feats_loss, suff_form_loss, suff_tag_loss, suff_feats_loss])
        return sorted_seq_lengths, sorted_seq_idx, sorted_token_seq[:, :sorted_seq_lengths[0]], preds, golds, losses

    def update_token_emb_(self, token_matrix: torch.Tensor):
        cur_token_matrix = self.token_emb.weight
        new_token_matrix = torch.cat([cur_token_matrix, token_matrix], dim=0)
        self.token_emb = nn.Embedding.from_pretrained(new_token_matrix, freeze=False, padding_idx=0)
