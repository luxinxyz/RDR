# -*- coding: utf-8 -*-
import os
import random
from itertools import chain

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DatasetBase(Dataset):
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files, self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line


class DoubanDataset(DatasetBase):
    def __init__(self, tokenizer, senti, valid=False, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(DoubanDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.senti = senti
        self.valid = valid

    def __getitem__(self, index):
        sent = self._get_line(index)
        sent = sent.strip()

        return self.process(sent)

    def process(self, sent, with_eos=True):

        if self.senti == "[NEU]":
            h_words = sent.split(" ")
            del_num = max(1, round(len(h_words) * 25 / 100))
            mask_id = random.sample(range(0, len(h_words)), del_num)
            new_sent = []
            for i in range(len(h_words)):
                if i not in mask_id:
                    new_sent.extend(list(h_words[i]))
            old_sent = list("".join(sent.split(" ")))
            sequence = ["[BOS]"] + new_sent + ["[SEP]"] + old_sent + ["[EOS]"]
        else:
            lines = sent.split("\t")
            senti = lines[0]
            sep_token = lines[1]
            new_sent = lines[-1].split(" ")
            sent = lines[2]
            old_sent = list("".join(sent.split(" ")))
            sequence = ["[BOS]"] + [sep_token] + [senti] + ["[SEP]"] + new_sent + ["[SEP]"] + old_sent + ["[EOS]"]

        # print(sequence)
        sequence = self.tokenizer.convert_tokens_to_ids(sequence)

        # print(sequence)

        instance = {}
        instance["input_ids"] = sequence
        instance["token_type_ids"] = [0] * len(sequence)
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            sep_token_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
            start_id_index = max(index for index, item in enumerate(sequence) if item == sep_token_id)  # find the last sep
            instance["lm_labels"][start_id_index + 1:len(sequence)] = sequence[start_id_index + 1:len(sequence)]
        return instance

    def collate(self, batch):
        input_ids = pad_sequence([torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch], batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence([torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch], batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence([torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch], batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))
