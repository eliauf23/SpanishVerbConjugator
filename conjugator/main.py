import numpy as np
import pandas as pd
import torch

from convert_single_input import DataProcessorSingleInput
from data_processing import DataProcessor
from conjugator import VerbConjugation
from post_processing import post_process


def load_saved_model(path_to_saved_model, data_proc):
    conj = VerbConjugation(data_processor=data_proc)
    conj.initialize(load_saved_model=True, path_to_saved_model=path_to_saved_model)
    return conj



# def predict_conjugation(conjugator, infinitive, tense, person, mood):
    # # Convert mood, tense, and person to integer values
    # mood = data_processor.convert_mood(mood)
    # tense = data_processor.convert_tense(tense)
    # person = data_processor.convert_person(person)
    #
    # # Encode the input and one-hot encode the mood, tense, and person
    # input_seq = data_processor.encode([infinitive], conjugator.input_vocab)
    # mood_one_hot = np.zeros(len(data_processor.mood_map))
    # mood_one_hot[int(mood)] = 1
    # tense_one_hot = np.zeros(len(data_processor.tense_map))
    # tense_one_hot[int(tense)] = 1
    # person_one_hot = np.zeros(len(data_processor.person_map))
    # person_one_hot[int(person)] = 1
    # one_hot_input = np.concatenate((mood_one_hot, tense_one_hot, person_one_hot))
    # # Combine the encoded input with the one-hot encoded mood, tense, and person
    # input_data = np.concatenate((input_seq[0], [conjugator.input_vocab['<PAD>']], one_hot_input))
    # # Pad the input sequence to a fixed length (65 in this case)
    # fixed_length = 65
    # input_data = np.pad(input_data, (0, fixed_length - len(input_data)), 'constant',
    #                     constant_values=(conjugator.input_vocab['<PAD>'],))
    # # Convert the input data to a tensor and send it to the device
    # input_tensor = torch.tensor(input_data, dtype=torch.long).unsqueeze(0).to(conjugator.device)
    # # Make a prediction using the model
    # output_tensor = conjugator.model(input_tensor)
    # # print(output_tensor.shape)
    # pred_indices = output_tensor.argmax(dim=-1)
    # # Convert the predicted indices back to characters
    # idx2char = {idx: char for char, idx in conjugator.output_vocab.items()}
    # pred_conjugation = ''.join([idx2char[x.item()] for x in pred_indices[0]])
    # conj = pred_conjugation.split('<PAD>')[0]
    # # post_processed_conj = post_process(conj)
    # return conj


#
# path = '../data/cleaned_data.csv'
# data_processor = DataProcessor(path_to_data=path, cols_to_drop=['verb_ending', 'is_reflexive'])
# data_processor.load_dataset()
# data_processor.build_and_encode_dataset()
# conjugator = VerbConjugation(data_processor=data_processor)
# conjugator = load_saved_model(path_to_saved_model="../saved_models/model_v8.pth", data_proc=data_processor)


infinitives = ['hablar', 'comer', 'vivir', 'ser', 'estar',
               'ir', 'tener', 'poner', 'saber', 'querer',
               'hacer', 'decir', 'ver', 'dar', 'poder', 'deber',
                'venir', 'salir', 'conocer', 'creer', 'llevar',
                'oír', 'caer', 'leer', 'traer', 'conducir', 'dormir',
                'servir', 'pedir', 'repetir', 'seguir', 'sentir',
                'vestir', 'freír', 'sonreír', 'morir', 'caber',
                'construir', 'huir', 'incluir', 'traducir', 'producir',
                'concluir', 'deducir', 'reducir', 'introducir', 'traducir',
                'construir', 'huir', 'incluir', 'traducir', 'producir', 'conseguir'
               ]
# MOODS = list(data_processor.mood_map.keys())
# TENSES = list(data_processor.tense_map.keys())
# PERSONS = list(data_processor.person_map.keys())
# df_verbs = pd.read_csv('../data/cleaned_data.csv')
# filtered_df = df_verbs[df_verbs['infinitive'].isin(infinitives)]
# filtered_df.to_csv('../data/test.csv', index=False)

# TODO: port to flask app

def predict_conjugation(infinitive, tense, person, mood):
    path = '../data/cleaned_data.csv'
    data_processor = DataProcessor(path_to_data=path, cols_to_drop=['verb_ending', 'is_reflexive'])
    data_processor.load_dataset()
    data_processor.build_and_encode_dataset()
    conjugator = VerbConjugation(data_processor=data_processor)
    conjugator.initialize(load_saved_model=True, path_to_saved_model="../saved_models/model_v8.pth")
    df = pd.DataFrame({'infinitive': [infinitive], 'tense': [tense], 'person': [person], 'mood': [mood], 'conjugation': ['conjugation'], 'stem': [infinitive[:-2]]})
    proc_single_input = DataProcessorSingleInput(data=df, data_processor=data_processor)
    single_dl = proc_single_input.build_dataloader()
    conjugator.model.eval()
    conjugation = None
    with torch.no_grad():
        for batch in single_dl:
            input_tensor = batch[0].to(conjugator.device)
            output_tensor = conjugator.model(input_tensor)
            pred_indices = output_tensor.argmax(dim=-1)
            pred_conjugation = ''.join([conjugator.idx2char[x.item()] for x in pred_indices[0]])
            conjugation = pred_conjugation.split('<PAD>')[0]
            print(conjugation)
    return conjugation

#
# data_processor = DataProcessor(path_to_data='../data/test.csv')
# data_processor.load_dataset()
# data_processor.build_and_encode_dataset()
# conjugator = VerbConjugation(data_processor=data_processor)
# conjugator = load_saved_model(path_to_saved_model="../saved_models/model_v8.pth", data_proc=data_processor)
# successes, faliures = conjugator.test()

# for s in successes:
#     mood, tense, person = s[0], s[1], s[2]
#     pred_conj = s[3]
#     target_conj = s[4]
#     # convert mood, tense, and person to string values - I don't have a reverse map for this yet
#     # get key given value in mood_map
#     mood = list(data_processor.mood_map.keys())[list(data_processor.mood_map.values()).index(mood)]
#     tense = list(data_processor.tense_map.keys())[list(data_processor.tense_map.values()).index(tense)]
#     person = list(data_processor.person_map.keys())[list(data_processor.person_map.values()).index(person)]
#     print(f"Mood: {mood}, Tense: {tense}, Person: {person}")
#     print(f"Predicted Conjugation: {pred_conj}")
#     print(f"Target Conjugation: {target_conj}")
#     print()

# for f in faliures:
#     mood_tense_person, pred_conj, target_conj = f[0], f[1], f[2]
#     mood, tense, person = mood_tense_person['mood'], mood_tense_person['tense'], mood_tense_person['person']
#     mood_str = [key for key, value in data_processor.mood_map.items() if value == mood][0]
#     tense_str = [key for key, value in data_processor.tense_map.items() if value == tense][0]
#     person_str = [key for key, value in data_processor.person_map.items() if value == person][0]
#     print(f"Mood: {mood_str}, Tense: {tense_str}, Person: {person_str}")
#     print(f"Predicted Conjugation: {pred_conj}")
#     print(f"Target Conjugation: {target_conj}")
#     print()
predict_conjugation('hablar', 'presente', '1s', 'indicativo')

