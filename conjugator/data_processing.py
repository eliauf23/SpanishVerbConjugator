import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from verb_conjugation_dataset import VerbConjugationDataset


class DataProcessor:
    def __init__(self, path_to_data, cols_to_drop=None):
        self.max_len = None
        self.mood_map = {'indicativo': 0, 'subjuntivo': 1, 'imperativo afirmativo': 2, 'imperativo negativo': 3}
        self.tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
                          'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
                          'condicional perfecto': 8}
        self.person_map = {'1s': 0, '2s': 1, '3s': 2, '1p': 3, '2p': 4, '3p': 5}
        self.data = None
        self.path_to_data = path_to_data
        self.endings = None
        self.cols_to_drop = cols_to_drop
        self.input_vocab = None
        self.output_vocab = None

    def load_dataset(self):
        if self.data is None:
            self.read_data(self.path_to_data, self.cols_to_drop)
        if self.endings is None:
            self.create_dictionary_common_endings()
        return self.data.copy(), self.endings.copy()

    def convert_mood(self, mood):
        if mood in self.mood_map.values():
            return mood
        return self.mood_map.get(mood, "no mood")

    def convert_tense(self, tense):
        if tense in self.tense_map.values():
            return tense
        return self.tense_map.get(tense, "no tense")

    def convert_person(self, person):
        if person in self.person_map.values():
            return person
        return self.person_map.get(person, "no person")

    def read_data(self, path, cols_to_drop=None):
        if self is not None:
            df = pd.read_csv(path)
            # if df_cols has is_reflexive, drop all rows where is_reflexive is True
            if 'is_reflexive' in df.columns:
                df = df[df['is_reflexive'] == False]
            if 'stem' in df.columns:
                df = df.dropna(subset=['stem'])
            if 'conjugation' in df.columns:
                df = df.dropna(subset=['conjugation'])
            df = df.dropna(subset=['infinitive', 'tense', 'mood', 'person'])
            if cols_to_drop is not None:
                df = df.drop(cols_to_drop, axis=1)
            df['mood'] = df['mood'].apply(self.convert_mood)
            df['tense'] = df['tense'].apply(self.convert_tense)
            df['person'] = df['person'].apply(self.convert_person)
            df = df[(df['mood'] != "no mood") & (df['tense'] != "no tense") & (df['person'] != "no person")]
            df = df.sample(frac=1, random_state=42)
            df.reset_index(drop=True, inplace=True)
            self.data = df
            return
        raise FileNotFoundError("path supplied to read_data must be defined")

    def create_dictionary_common_endings(self):
        self.endings = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for _, row in self.data.iterrows():
            person = row['person']
            tense = row['tense']
            mood = row['mood']
            conjugation = row['conjugation']
            stem = row['stem']
            conj_tokens = conjugation.split(' ')
            if len(conj_tokens) == 2:
                ending = conj_tokens[1][len(stem):]
            else:
                ending = conjugation[len(stem):]
            self.endings[person][tense][mood].add(ending)

    def get_common_endings(self, person, tense, mood):
        person = self.convert_person(person)
        tense = self.convert_tense(tense)
        mood = self.convert_mood(mood)
        common_endings = self.endings[person][tense][mood]
        if '' in common_endings:
            common_endings.remove('')
        return common_endings if len(common_endings) > 0 else None

    def print_common_endings(self):
        MOODS = self.mood_map.keys()
        TENSES = self.tense_map.keys()
        SUBJECTS = self.person_map.keys()
        for m in MOODS:
            print(f"\n\nMood: {m}\n----------------------")
            for t in TENSES:
                print(f"  Tense: {t}")
                for s in SUBJECTS:
                    print(f"    Subject {s}: {self.get_common_endings(s, t, m)}")

    @staticmethod
    def encode(column, vocab):
        encoded = []
        for item in column:
            tokens = list(item)
            encoded_item = [vocab[token] for token in tokens]
            encoded_item.append(vocab['<PAD>'])
            encoded.append(encoded_item)
        return encoded

    @staticmethod
    def build_vocab(column, special_char='<PAD>'):
        vocab = defaultdict(lambda: len(vocab))
        spanish_characters = "abcdefghijklmnopqrstuvwxyzáéíóúñü "
        for char in spanish_characters:
            vocab[char]
        vocab[special_char]
        for item in column:
            tokens = list(item)
            tokens.append(special_char)
            for token in tokens:
                vocab[token]
        return vocab

    def pad_sequences(self, batch, only_x=False):
        inputs, targets, mood_tense_person = (batch, None) if only_x else zip(*batch)
        max_len_x = max(len(x) for x in inputs)
        max_len_y = max(len(y) for y in targets) if targets is not None else 0
        max_len = max(max_len_x, max_len_y)
        self.max_len = max_len
        padded_inputs = np.zeros((len(inputs), max_len), dtype=np.int64)
        for i, seq in enumerate(inputs):
            padded_inputs[i, :len(seq)] = seq
        if targets is not None:
            padded_targets = np.zeros((len(targets), max_len), dtype=np.int64)
            for i, seq in enumerate(targets):
                padded_targets[i, :len(seq)] = seq
            return torch.from_numpy(padded_inputs), torch.from_numpy(padded_targets), mood_tense_person
        return torch.from_numpy(padded_inputs), mood_tense_person

    def build_and_encode_dataset(self):
        if self.input_vocab is None or self.output_vocab is None:
            self.input_vocab = self.build_vocab(self.data['infinitive'])
            self.output_vocab = self.build_vocab(self.data['conjugation'])

        self.data['infinitive_encoded'] = self.encode(self.data['infinitive'], self.input_vocab)
        self.data['conjugation_encoded'] = self.encode(self.data['conjugation'], self.output_vocab)
        self.one_hot_encode_inputs()

    def one_hot_encode_inputs(self):
        assert self.input_vocab is not None
        assert self.data is not None
        one_hot_encoder = OneHotEncoder()
        one_hot_data = one_hot_encoder.fit_transform(self.data[['mood', 'tense', 'person']]).toarray()
        padded_one_hot_data = np.pad(one_hot_data, ((0, 0), (0, len(self.input_vocab) - 1)), 'constant')
        self.data['input'] = self.data.apply(
            lambda row: row['infinitive_encoded'] + [self.input_vocab['<PAD>']] + list(padded_one_hot_data[row.name]),
            axis=1)

    def split_data(self, frac_train=0.7, frac_valid=0.1, frac_test=0.2, random_state=42):
        if frac_valid is None or frac_valid == 0.0:
            train_data = self.data.sample(frac=frac_train, random_state=random_state)
            test_data = self.data.drop(train_data.index)
            return train_data, test_data, None

        if frac_train + frac_valid + frac_test != 1.0:
            raise ValueError("train, validation, test set fractions must add up to 1")
        train_data = self.data.sample(frac=frac_train, random_state=random_state)
        val_and_test_data = self.data.drop(train_data.index)
        val_data = val_and_test_data.sample(frac=(frac_valid / (frac_test + frac_valid)), random_state=random_state)
        test_data = val_and_test_data.drop(val_data.index)
        return train_data, test_data, val_data

    def split_data_and_create_dataloaders(self, frac_train=0.7, frac_test=0.2, frac_valid=0.1, random_state=42,
                                          batch_size=32):

        train_data, val_data, test_data = self.split_data(frac_train=frac_train, frac_valid=frac_valid,
                                                          frac_test=frac_test, random_state=random_state)
        # create datasets
        train_dataset = VerbConjugationDataset(train_data) if frac_train > 0.0 else None
        val_dataset = VerbConjugationDataset(val_data) if frac_valid > 0.0 else None
        test_dataset = VerbConjugationDataset(test_data) if frac_test > 0.0 else None
        # Create the data loaders
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=self.pad_sequences) if frac_train > 0.0 else None
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=self.pad_sequences) if frac_valid > 0.0 else None
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=self.pad_sequences) if frac_test > 0.0 else None
        return train_dl, val_dl, test_dl
