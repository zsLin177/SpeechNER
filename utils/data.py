import random
import math
import torch
import torchaudio as ta
import re
from utils.process import read_json, read_symbol_table, build_ner_vocab
from supar.utils.fn import pad
from supar.utils.field import SubwordField

per_regex = '\[.+?\]'
loc_regex = '\(.+?\)'
org_regex = '\<.+?\>'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file, char_dict_file, num_mel_bins=80, frame_length=25, frame_shift=10, max_frame_num=100000, speed_perturb=False, spec_aug=False, bert_tokenizer=None, e2ener=False, use_same_tokenizer=False, add_context=False) -> None:
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.max_frame_num = max_frame_num
        self.data = read_json(json_file)
        # <black> = 0, <unk> = 1, <sos/eos> = 4232
        self.char_dict = read_symbol_table(char_dict_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bert_tokenizer = bert_tokenizer
        self.e2ener = e2ener
        self.use_same_tokenizer = use_same_tokenizer
        self.add_context = add_context
        if self.bert_tokenizer is not None:
            self.word_field = SubwordField('bert',
                                    pad=self.bert_tokenizer.pad_token,
                                    unk=self.bert_tokenizer.unk_token,
                                    bos=self.bert_tokenizer.cls_token,
                                    eos=self.bert_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bert_tokenizer.tokenize)
            self.word_field.vocab = self.bert_tokenizer.get_vocab()

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)

        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]
        label, tokens = self.tokenize_for_asr(sentence)
        if not self.use_same_tokenizer:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = self.word_field.transform([tokens])[0][1:-1].squeeze(-1)
        
        key = self.data[index]["key"]

        if self.bert_tokenizer is not None:
            bert_input = self.word_field.transform([tokens])[0]
        else:
            bert_input = None

        if self.add_context:
            need_att_mask = self.build_need_att_mask(sentence)
        else:
            need_att_mask = None

        return y, label, key, bert_input, need_att_mask

    def build_need_att_mask(self, sentence, add_bos=True):
        if add_bos:
            mask = [0] * (len(sentence)+1)
            sentence = '#' + sentence
        else:
            mask = [0] * len(sentence)
        mask = torch.tensor(mask)

        per_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(per_regex, sentence)]
        loc_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(loc_regex, sentence)]
        org_intervals = [(item.span()[0], item.span()[1]-2) for item in re.finditer(org_regex, sentence)]

        for st, ed in per_intervals+loc_intervals+org_intervals:
            mask[st:ed+1] = 1
        
        return mask

    def tokenize_for_asr(self, sentence):
        label = []
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        for ch in tokens:
            if ch in self.char_dict:
                label.append(self.char_dict[ch])
            elif '<unk>' in self.char_dict:
                label.append(self.char_dict['<unk>'])
            else:
                raise KeyError
        return label, tokens

def collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    asr_target = pad([instance[1] for instance in batch], padding_value=-1)
    asr_target_length = torch.tensor([instance[1].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[2] for instance in batch]
    
    curr_bert = batch[0][3]

    if batch[0][4] is not None:
        need_att_mask = pad([instance[4] for instance in batch], padding_value=0)
        need_att_mask = need_att_mask.bool()
    else:
        need_att_mask = None

    if curr_bert is not None:
        bert_pad_idx = 0
        bert_input = pad([instance[3] for instance in batch], padding_value=bert_pad_idx)
        return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'bert_input': bert_input,
                'need_att_mask': need_att_mask}
    
    return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'need_att_mask': need_att_mask}


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, char_dict_file, if_flat, if_build_ner_vocab=True, ner_vocab=None, num_mel_bins=80, speed_perturb=False, spec_aug=False, bert_tokenizer=None, use_same_tokenizer=False) -> None:
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.data = read_json(json_file, if_flat)
        # <black> = 0, <unk> = 1, <sos/eos> = 4232
        self.char_dict = read_symbol_table(char_dict_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bert_tokenizer = bert_tokenizer
        self.use_same_tokenizer = use_same_tokenizer
        if self.bert_tokenizer is not None:
            self.word_field = SubwordField('bert',
                                    pad=self.bert_tokenizer.pad_token,
                                    unk=self.bert_tokenizer.unk_token,
                                    bos=self.bert_tokenizer.cls_token,
                                    eos=self.bert_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bert_tokenizer.tokenize)
            self.word_field.vocab = self.bert_tokenizer.get_vocab()

        if if_build_ner_vocab:
            self.ner_vocab = build_ner_vocab(self.data)
        else:
            self.ner_vocab = ner_vocab

    def __len__(self):
        return len(self.data)

    def get_ner(self, index):
        # based on words
        """
        sentence: 我是苏州人 -> [BOS] 我 是 苏 州 人 [EOS]
        ner_labels: [[None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, LOC, None],
                     [None, None, None, LOC, None, None],
                     [None, None, None, None, None, None],
                     [None, None, None, None, None, None],
                    ]
        """
        real_char_num = len(self.data[index]['sentence'])
        # plus bos and eos
        char_num = real_char_num + 2
        null_label_idx = len(self.ner_vocab)
        # row: end; column: start
        ner_labels = [[null_label_idx] * char_num for i in range(char_num)]

        dic = self.data[index]
        for ner in dic['entity']:
            st, ed, label = ner[0], ner[1], ner[3]
            ner_labels[ed][st+1] = self.ner_vocab[label]
        # lower triangular
        return ner_labels
    
    def tokenize_for_asr(self, sentence):
        label = []
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        for ch in tokens:
            if ch in self.char_dict:
                label.append(self.char_dict[ch])
            elif '<unk>' in self.char_dict:
                label.append(self.char_dict['<unk>'])
            else:
                raise KeyError
        return label, tokens

    def __getitem__(self, index):
        try:
            audio_path = self.data[index]["wav"]
        except KeyError:
            audio_path = self.data[index]["audio"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=25, frame_shift=10,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        try:
            sentence = self.data[index]["txt"]
        except KeyError:
            sentence = self.data[index]["sentence"]
        label, tokens = self.tokenize_for_asr(sentence)
        if not self.use_same_tokenizer:
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = self.word_field.transform([tokens])[0][1:-1].squeeze(-1)
        
        try:
            key = self.data[index]["key"]
        except KeyError:
            key = sentence

        if self.bert_tokenizer is not None:
            bert_input = self.word_field.transform([tokens])[0]
        else:
            bert_input = None
        
        # get ner labels
        ner_tensor = torch.tensor(self.get_ner(index))
        return y, label, key, bert_input, ner_tensor

def ner_collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    asr_target = pad([instance[1] for instance in batch], padding_value=-1)
    asr_target_length = torch.tensor([instance[1].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[2] for instance in batch]
    ner_labels = pad([instance[4] for instance in batch], padding_value=-1)
    
    curr_bert = batch[0][3]
    if curr_bert is not None:
        bert_pad_idx = 0
        bert_input = pad([instance[3] for instance in batch], padding_value=bert_pad_idx)
        return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'bert_input': bert_input,
                'ner': ner_labels}
    
    return {'audio_feat': audio_feat,
                'asr_target': asr_target,
                'audio_feat_length': audio_feat_length,
                'asr_target_length': asr_target_length,
                'keys': keys,
                'ner': ner_labels}
 
class BartSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                json_file,
                num_mel_bins=80, 
                frame_length=25,
                frame_shift=10,
                max_frame_num=512,
                speed_perturb=False, 
                spec_aug=False, 
                bart_tokenizer=None, 
                e2ener=False) -> None:
        """
        in fnlp/bart-base-chinese, the tokenizer is same as bert-base-chinese
        """
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.max_frame_num = max_frame_num
        self.data = read_json(json_file)
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.bart_tokenizer = bart_tokenizer
        self.e2ener = e2ener
        if self.bart_tokenizer is not None:
            self.word_field = SubwordField('bart',
                                    pad=self.bart_tokenizer.pad_token,
                                    unk=self.bart_tokenizer.unk_token,
                                    bos=self.bart_tokenizer.cls_token,
                                    eos=self.bart_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bart_tokenizer.tokenize)
            self.word_field.vocab = self.bart_tokenizer.get_vocab()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]["wav"]
        waveform, sample_frequency = ta.load(audio_path)

        # speed_pertrub
        if self.speed_perturb:
            speeds = [0.9, 1.0, 1.1]
            speed = random.choice(speeds)
            if speed != 1.0:
                waveform, _ = ta.sox_effects.apply_effects_tensor(
                waveform, sample_frequency,
                [['speed', str(speed)], ['rate', str(sample_frequency)]])
        
        # compute fbank
        waveform = waveform * (1 << 15)
        mat = ta.compliance.kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, frame_length=self.frame_length, frame_shift=self.frame_shift,
                                    dither=0.1, energy_floor=0.0, sample_frequency=sample_frequency)
        
        # spec_aug
        if self.spec_aug:
            num_t_mask, num_f_mask, max_t, max_f = 2, 2, 50, 10
            y = mat.clone().detach()
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            # that is turning some frame to zero
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0

            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
        else:
            y = mat
        
        # frame sample to max_frame_num
        raw_frame_num = y.size(0)
        if raw_frame_num > self.max_frame_num:
            sample_rate = raw_frame_num / self.max_frame_num
            selected_idx = torch.tensor([i for i in range(raw_frame_num) if math.floor(i % sample_rate) == 0], dtype=torch.long)
            y = torch.index_select(y, 0, selected_idx)
        
        if not self.e2ener:
            sentence = self.data[index]["txt"]
        else:
            sentence = self.data[index]["ner_txt"]

        tokens = self.tokenize(sentence)
        key = self.data[index]["key"]
        if self.bart_tokenizer is not None:
            bart_input = self.word_field.transform([tokens])[0]
        else:
            bart_input = None
        
        return y, key, bart_input

    def tokenize(self, sentence):
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        return tokens
        
def bartseq2seq_collate_fn(batch):
    audio_feat = pad([instance[0] for instance in batch])
    audio_feat_length = torch.tensor([instance[0].size(0) for instance in batch],
                                    dtype=torch.int32)
    keys = [instance[1] for instance in batch]
    curr_bart = batch[0][2]
    if curr_bart is not None:
        bart_pad_idx = 0
        bart_input = pad([instance[2] for instance in batch], padding_value=bart_pad_idx)
        return {'audio_feat': audio_feat,
                'audio_feat_length': audio_feat_length,
                'keys': keys,
                'bart_input': bart_input}
    else:
        raise KeyError("need bart input")

class BartTxtSeq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                json_file,
                bart_tokenizer=None, 
                e2ener=False,
                add_noise=False,
                mask_rate=0.35,
                poisson_avg=3) -> None:
        """
        in fnlp/bart-base-chinese, the tokenizer is same as bert-base-chinese
        """
        super().__init__()
        self.data = read_json(json_file)
        self.bart_tokenizer = bart_tokenizer
        self.e2ener = e2ener
        self.add_noise = add_noise
        self.mask_rate = mask_rate
        self.poisson_avg = poisson_avg
        self.poisson = torch.distributions.Poisson(poisson_avg)
        if self.bart_tokenizer is not None:
            self.word_field = SubwordField('bart',
                                    pad=self.bart_tokenizer.pad_token,
                                    unk=self.bart_tokenizer.unk_token,
                                    bos=self.bart_tokenizer.cls_token,
                                    eos=self.bart_tokenizer.sep_token,
                                    fix_len=10,
                                    tokenize=self.bart_tokenizer.tokenize)
            self.word_field.vocab = self.bart_tokenizer.get_vocab()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.e2ener:
            tgt = self.data[index]["txt"]
        else:
            tgt = self.data[index]["ner_txt"]
        tgt_tokens = self.my_tokenize(tgt)

        # src = self.data[index]["asrout"]
        src = self.data[index]["txt"]
        src_tokens = self.my_tokenize(src)
        if self.add_noise:
            # TODO
            pass
        key = self.data[index]["key"]

        return self.word_field.transform([src_tokens])[0].squeeze(-1), self.word_field.transform([tgt_tokens])[0].squeeze(-1), key

    def my_tokenize(self, sentence):
        tokens = []
        for ch in sentence:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)
        return tokens

    def add_noise_for_text_infilling(self, src_tokens_lst, max_span_len=10):
        src_token_num = len(src_tokens_lst)
        n_masked_num = math.ceil(src_token_num * self.mask_rate)
        if n_masked_num == 0:
            return src_tokens_lst
        
        span_masked_res = []
        # containing the character masked and the [MASK] added
        masked_num = 0
        masked_mask = torch.tensor([1] * src_token_num, dtype=torch.long)
        # the num of [MASK] added
        added_mask_num = 0

        while masked_num < n_masked_num:
            span_len = min(self.poisson.sample(sample_shape=(1,)).item(), max_span_len)
            if span_len > 0:
                span_len = min()
            else:
                added_mask_num += 1
                masked_num += 1

def barttxtseq2seq_collate_fn(batch):
    bart_pad_idx = 0
    keys = [instance[2] for instance in batch]
    src = pad([instance[0] for instance in batch], padding_value=bart_pad_idx)
    tgt = pad([instance[1] for instance in batch], padding_value=bart_pad_idx)
    return {'keys': keys,
            'src': src,
            'tgt': tgt}

