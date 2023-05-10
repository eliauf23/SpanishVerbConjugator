import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from verb_conjugation_dataset import VerbConjugationDataset


class DataProcessorSingleInput:
    def __init__(self, data, original_data_processor):
        self.original_data_processor = original_data_processor
        self.mood_map = {'indicativo': 0, 'subjuntivo': 1, 'imperativo afirmativo': 2, 'imperativo negativo': 3}
        self.tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
                          'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
                          'condicional perfecto': 8}
        self.person_map = {'1s': 0, '2s': 1, '3s': 2, '1p': 3, '2p': 4, '3p': 5}
        self.data = data
        self.data['mood'].apply(self.convert_mood)
        self.data['tense'].apply(self.convert_tense)
        self.data['person'].apply(self.convert_person)
        self.build_and_encode_dataset()

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

    @staticmethod
    def encode(column, vocab):
        encoded = []
        for item in column:
            tokens = list(item)
            encoded_item = [vocab[token] for token in tokens]
            encoded_item.append(vocab['<PAD>'])
            encoded.append(encoded_item)
        return encoded

    def pad_sequences_only_x(self, batch):
        inputs, mood_tense_person = batch
        max_len_x = self.original_data_processor.max_len
        padded_inputs = np.zeros((len(inputs), max_len_x), dtype=np.int64)
        for i, seq in enumerate(inputs):
            padded_inputs[i, :len(seq)] = seq
        return torch.from_numpy(padded_inputs), mood_tense_person

    def build_and_encode_dataset(self):
        input_vocab = self.original_data_processor.input_vocab
        output_vocab = self.original_data_processor.output_vocab
        self.data['infinitive_encoded'] = self.encode(self.data['infinitive'], input_vocab)
        self.data['conjugation_encoded'] = self.encode(self.data['conjugation'], output_vocab)
        one_hot_encoder = OneHotEncoder()
        one_hot_data = one_hot_encoder.fit_transform(self.data[['mood', 'tense', 'person']]).toarray()
        padded_one_hot_data = np.pad(one_hot_data, ((0, 0), (0, len(input_vocab) - 1)), 'constant')
        self.data['input'] = self.data.apply(
            lambda row: row['infinitive_encoded'] + [input_vocab['<PAD>']] + list(padded_one_hot_data[row.name]),
            axis=1)

    def create_dataloader(self, batch_size=1):
        dataset = VerbConjugationDataset(self.data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.pad_sequences_only_x
        )

