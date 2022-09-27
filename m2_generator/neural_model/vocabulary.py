import json
import re


class Vocabulary:
    def __init__(self, tokenizer, list_words, min_occ=3, unk_tok='<unk>', pad_tok='<pad>', vocab=None):
        self.tokenizer = tokenizer
        self.min_occ = min_occ
        self.unk_tok = unk_tok
        self.pad_tok = pad_tok
        if vocab is None:
            self.vocab = {self.pad_tok: 0,
                          self.unk_tok: 1}
            self.build_vocabulary(list_words)
        else:
            self.vocab = vocab

    def build_vocabulary(self, list_words):
        list_subwords = [sw for w in list_words for sw in self.tokenizer(w)]
        count = {sw: list_subwords.count(sw) for sw in set(list_subwords)}
        for sw in count:
            if count[sw] >= 3:
                self.vocab[sw] = len(self.vocab)

    def to_id(self, word):
        subwords = [sw for sw in self.tokenizer(word)]
        ids = []
        for sw in subwords:
            if sw not in self.vocab:
                ids.append(self.vocab[self.unk_tok])
            else:
                ids.append(self.vocab[sw])
        return ids

    def to_id_pad(self, word, max_len):
        ids = self.to_id(word)
        if len(ids) > max_len:
            return ids[0:max_len]
        else:
            pads = [self.vocab[self.pad_tok] for _ in range(max_len - len(ids))]
            return ids + pads

    def __len__(self):
        return len(self.vocab)


'''
    https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
'''


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def camel_case_tokenizer(word):
    return [sw.lower() for sw in camel_case_split(word)]
