import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from verb_conjugation_lstm import VerbConjugationLSTM
from utils import get_device
class VerbConjugation:
    def __init__(self, data_processor, model_save_path="/saved_models/conjugator_model.pth", frac_train=0.7,
                 frac_test=0.2,
                 frac_valid=0.1, batch_size=32):

        self.data = data_processor.data
        self.embedding_dim = 128
        self.hidden_size = 256
        self.n_layers = 2
        self.batch_size = batch_size
        self.num_dir_lstm = 2
        self.default_epochs_train = 30
        self.input_vocab = data_processor.input_vocab
        self.output_vocab = data_processor.output_vocab
        self.input_size = len(data_processor.input_vocab)
        self.output_size = len(data_processor.output_vocab)
        self.model_save_path = model_save_path if model_save_path is not None else "/saved_models/conjugator_model.pth"  # TODO: change for your device
        self.lr = 0.001
        self.frac_train = frac_train
        self.frac_test = frac_test
        self.frac_valid = frac_valid
        self.data_processor = data_processor
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.device = None
        self.criterion = None
        self.optimizer = None
        self.model = None
        self.idx2char  = {idx: char for char, idx in self.output_vocab.items()}

        # hyperparameters for custom loss function
        self.spelling_penalty_weight = 1.0  # can be fixed with spellcheck
        self.nonsense_endings_penalty_weight = 2.0  # can be fixed with regex detection + trim
        self.incorrect_ending_penalty_weight = 3.0  # can't really be fixed with post-processing


    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy of predicted conjugations."""
        self.idx2char = {idx: char for char, idx in self.output_vocab.items()}
        pred_indices = outputs.argmax(dim=-1)
        correct = 0
        total = 0
        target_conj, output_conj = [], []
        for i in range(len(targets)):
            target_conjugation = ''.join([self.idx2char[x.item()] for x in targets[i]])
            target_conj.append(target_conjugation.split('<PAD>')[0])
            pred_conjugation = ''.join([self.idx2char[x.item()] for x in pred_indices[i]])
            output_conj.append(pred_conjugation.split('<PAD>')[0])
            if pred_conjugation == target_conjugation:
                correct += 1
            total += 1
        return correct / total, target_conj, output_conj

    def calculate_custom_loss(self, outputs, targets, mood_tense_person):
        list_mtp = list(mood_tense_person)
        custom_loss = 0
        for i in range(len(outputs)):
            output = outputs[i]
            target = targets[i]
            mood = list_mtp[i]['mood']
            tense = list_mtp[i]['tense']
            person = list_mtp[i]['person']
            pred_indices = outputs.argmax(dim=-1)
            common_endings = self.data_processor.get_common_endings(mood=mood, tense=tense, person=person)
            pred_conjugation = (''.join([self.idx2char[x.item()] for x in pred_indices[i]])).split('<PAD>')[0]
            target_conjugation = (''.join([self.idx2char[x.item()] for x in targets[i]])).split('<PAD>')[0]
            if common_endings is not None:
                max_len_endings = max(len(item) for item in common_endings)
                # Check if endings of sizes 1 to max_len endings of the pred_conj are in list.
                # If not, penalize, else, do normal loss.
                for j in range(1, max_len_endings + 1):
                    if pred_conjugation[-j:] not in common_endings:
                        # Penalize for incorrect conjugation
                        custom_loss += self.incorrect_ending_penalty_weight * self.criterion(
                            output.view(-1, self.output_size), target.view(-1))
                        break
                else:
                    # Do normal loss calculation
                    custom_loss += self.criterion(output.view(-1, self.output_size), target.view(-1))
            else:
                # Do normal loss calculation
                custom_loss += self.criterion(output.view(-1, self.output_size), target.view(-1))

            # Penalize for spelling mistakes (if any)
            if pred_conjugation != target_conjugation:
                # TODO: check with regex for repeated characters on the end & apply penalty: nonsense_endings_penalty_weight
                custom_loss += self.spelling_penalty_weight * self.criterion(output.view(-1, self.output_size),
                                                                             target.view(-1))

        custom_loss /= len(outputs)
        return custom_loss

    def initialize(self, load_saved_model=False, path_to_saved_model="/content/drive/MyDrive/conjugator_model.pth", test_only=False):
        self.device = get_device()
        self.model = VerbConjugationLSTM(self.input_size, self.output_size, self.embedding_dim, self.hidden_size,
                                         self.n_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.data is not None:
            if test_only:
                self.test_dataloader = self.data_processor.create_dataloader(self.data, self.batch_size)
            else:
                self.train_dataloader, self.val_dataloader, self.test_dataloader = self.data_processor.split_data_and_create_dataloaders(
                frac_train=self.frac_train,
                frac_test=self.frac_test,
                frac_valid=self.frac_valid,
                random_state=42,
                batch_size=self.batch_size)

        if load_saved_model:
            path = path_to_saved_model if path_to_saved_model is not None else self.model_save_path
            print(path)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            # self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded in evaluation mode on device {self.device}")
        else:
            print(f"Model initialized on device {self.device}")

    def train_and_validate(self, epochs, validation=True):
        EPOCHS = epochs or self.default_epochs_train
        self.model.train()
        for epoch in range(EPOCHS):
            train_loss = 0
            train_accuracy = 0
            pbar_train = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")
            for inputs, targets, mood_tense_person in pbar_train:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.calculate_custom_loss(outputs, targets, mood_tense_person)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                acc, _, _ = self.calculate_accuracy(outputs, targets)
                train_accuracy += acc
                pbar_train.set_postfix(train_loss=train_loss / (len(self.train_dataloader)),
                                       train_accuracy=train_accuracy / (len(self.train_dataloader)))
            if validation:
                self.validate(epoch=epoch, epochs=EPOCHS)
                self.model.train()

        # Save the model
        torch.save(self.model.state_dict(), self.model_save_path)
        output_str = " Finished training" + (" and validation" if validation else "")
        print(f"\n\n{output_str}!\n\n")

    def validate(self, epoch, epochs):
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        pbar_val = tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}")
        with torch.no_grad():
            for inputs, targets, mood_tense_person in pbar_val:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.calculate_custom_loss(outputs, targets, mood_tense_person)
                val_loss += loss.item()
                acc, _, _ = self.calculate_accuracy(outputs, targets)
                val_accuracy += acc
                pbar_val.set_postfix(val_loss=val_loss / (len(self.val_dataloader)),
                                     val_accuracy=val_accuracy / (len(self.val_dataloader)))
        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(self.val_dataloader)}, Validation Accuracy: {val_accuracy / len(self.val_dataloader)}")

    def test(self):
        s, f = [], []
        self.model.eval()
        test_loss = 0
        test_acc = 0
        i = 0
        with torch.no_grad():
            for inputs, targets, mood_tense_person in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.calculate_custom_loss(outputs, targets, mood_tense_person)
                test_loss += loss.item()
                acc, target_conj, pred_conj = self.calculate_accuracy(outputs, targets)
                f.extend(
                    (mood_tense_person[x], pred_conj[x], target_conj[x])
                    for x in range(len(pred_conj))
                    if pred_conj[x] != target_conj[x]
                )
                s.extend(
                    (mood_tense_person[x], pred_conj[x], target_conj[x])
                    for x in range(len(pred_conj))
                    if pred_conj[x] == target_conj[x]
                )
                i = i + 1
                test_acc += acc
        print(
            f"Test Loss: {test_loss / len(self.test_dataloader)}, Validation Accuracy: {test_acc / len(self.test_dataloader)}")
        return s, f

    def predict(self, dataloader):
        # encode data -

        self.model.eval()
        results = []
        with torch.no_grad():
            for inputs, mood_tense_person in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                acc, targ_conj, pred_conj = self.calculate_accuracy(outputs, outputs)
                assert acc == 1
                print(pred_conj)
                print(targ_conj)
                results.extend(
                    (mood_tense_person[x], pred_conj[x])
                    for x in range(len(pred_conj))
                    if x == 0
                )
                print(results[0])
                return results
        return results



    def continue_training(self, epochs):
        # TODO: fix
        self.initialize(load_saved_model=True)
        self.train_and_validate(epochs=epochs)