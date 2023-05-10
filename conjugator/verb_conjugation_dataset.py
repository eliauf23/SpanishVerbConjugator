
import numpy as np
import torch
from torch.utils.data import Dataset

class VerbConjugationDataset(Dataset):
    def __init__(self, _data, return_y=True):
        self.data = _data
        self.return_y = return_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mood_tense_person = {"mood": self.data.iloc[index]["mood"],
                             "tense": self.data.iloc[index]["tense"],
                             "person": self.data.iloc[index]["person"],
                             "stem": self.data.iloc[index]["stem"],
                             }
        x = np.array(self.data.iloc[index]['input']).astype(np.int64)
        if self.return_y:
            y = np.array(self.data.iloc[index]['conjugation_encoded']).astype(np.int64)
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), mood_tense_person
        else:
            return torch.tensor(x, dtype=torch.long), mood_tense_person
