import pandas as pd
import torch

from conjugator.data_processing import DataProcessor
from conjugator.conjugator import VerbConjugation
from mlconjug3 import Conjugator


def load_saved_model(path_to_saved_model, data_proc):
    conj = VerbConjugation(data_processor=data_proc)
    conj.initialize(load_saved_model=True, path_to_saved_model=path_to_saved_model)
    return conj


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


# TODO: port to flask app

def predict_single_conjugation(infinitive, tense, person, mood):
    path = '../data/cleaned_data.csv'
    data_processor = DataProcessor(path_to_data=path, cols_to_drop=['verb_ending', 'is_reflexive'])
    data_processor.load_dataset()
    data_processor.build_and_encode_dataset()
    print(data_processor.max_len)
    conjugator = VerbConjugation(data_processor=data_processor)
    conjugator.initialize(load_saved_model=True, path_to_saved_model="/Users/eliauf/PycharmProjects/VerbConjugator/saved_models/model_v8.pth")
    single_dl = data_processor.create_single_input_dataloader(infinitive=infinitive, mood=mood, person=person,
                                                              tense=tense)
    conjugator.model.eval()
    conjugation = None
    with torch.no_grad():
        for input, mood_tense_person in single_dl:
            input = input.to(conjugator.device)

            output_tensor = conjugator.model(input)
            pred_indices = output_tensor.argmax(dim=-1)
            pred_conjugation = ''.join([conjugator.idx2char[x.item()] for x in pred_indices[0]])
            conjugation = pred_conjugation.split('<PAD>')[0]
            print(input, mood_tense_person)
            print(conjugation)
    return conjugation


# def predict_all_persons_conj_for_tense_mood(infinitive, tense, mood):
#     path = '../data/cleaned_data.csv'
#     data_processor = DataProcessor(path_to_data=path, cols_to_drop=['verb_ending', 'is_reflexive'])
#     data_processor.load_dataset()
#     data_processor.build_and_encode_dataset()
#     print(data_processor.max_len)
#     conjugator = VerbConjugation(data_processor=data_processor)
#     conjugator.initialize(load_saved_model=True, path_to_saved_model="../saved_models/model_v8.pth")
#     single_dl = data_processor.create_single_input_dataloader(infinitive=infinitive, mood=mood, person=person, tense=tense)
#     conjugator.model.eval()
#     conjugation = None
#     with torch.no_grad():
#         for batch in single_dl:
#             input_tensor = batch[0].to(conjugator.device)
#             output_tensor = conjugator.model(input_tensor)
#             pred_indices = output_tensor.argmax(dim=-1)
#             pred_conjugation = ''.join([conjugator.idx2char[x.item()] for x in pred_indices[0]])
#             conjugation = pred_conjugation.split('<PAD>')[0]
#             print(conjugation)
#     return conjugation


def get_all_conjugations_for_infinitive(infinitive):
    tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
                 'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
                 'condicional perfecto': 8}
    TENSES = list(tense_map.keys())
    conjugator = Conjugator(language='es')
    verb = conjugator.conjugate(infinitive)
    data = {
        "infinitive": [],
        "is_reflexive": [],
        "verb_ending": [],
        "stem": [],
        "mood": [],
        "tense": [],
        "person": [],
        "conjugation": []
    }
    subj_map = {"yo": "1s",
                "tú": "2s",
                "él": "3s",
                "nosotros": "1p",
                "vosotros": "2p",
                "ellos": "3p",
                }
    for m in ['Indicativo', 'Subjuntivo', 'Condicional']:
        for t in TENSES:
            if t == 'imperfecto':
                tense_modif = 'pretérito imperfecto'
            elif t == 'pretérito':
                tense_modif = 'pretérito perfecto simple'
            else:
                tense_modif = t
            tense_modif = m + ' ' + tense_modif
            for p in subj_map.keys():
                try:
                    conjugation = verb[m][tense_modif][p]
                    data["infinitive"].append(infinitive)
                    data["is_reflexive"].append(False)
                    data["verb_ending"].append(infinitive[-2:])
                    data["stem"].append(infinitive[:-2])
                    if m == 'Condicional':
                        mood = 'indicativo'
                        tense = 'condicional'
                    else:
                        mood = m.lower()
                        tense = t.lower()
                    data["mood"].append(mood)
                    data["tense"].append(tense)
                    data["person"].append(subj_map[p])
                    data["conjugation"].append(conjugation)
                except KeyError:
                    continue

    data = pd.DataFrame(data)
    # print(len(data))
    # print(data.head(n=20))
    data_proc = DataProcessor(df=data)
    conjugator = VerbConjugation(data_processor=data_proc)
    conjugator.initialize(load_saved_model=True, path_to_saved_model="/Users/eliauf/PycharmProjects/VerbConjugator/saved_models/model_v8.pth", test_only=True)
    conjugations = conjugator.test()

    mood_map = {'indicativo': 0, 'subjuntivo': 1, 'imperativo afirmativo': 2, 'imperativo negativo': 3}
    tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
                 'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
                 'condicional perfecto': 8}
    person_map = {'1s': 0, '2s': 1, '3s': 2, '1p': 3, '2p': 4, '3p': 5}

    inverted_mood_map = {value: key for key, value in mood_map.items()}
    inverted_tense_map = {value: key for key, value in tense_map.items()}
    inverted_person_map = {value: key for key, value in person_map.items()}
    organized_conjugations = {}
    for x in conjugations:
        mtp, pred, target = x
        mood, tense, person = mtp["mood"], mtp["tense"], mtp["person"]
        # convert mood, tense, person to words
        mood = inverted_mood_map[mood]
        tense = inverted_tense_map[tense]
        person = inverted_person_map[person]
        print(mood, tense, person, pred, target)
        organized_conjugations[(mood, tense, person)] = (pred, target)

    # sort by mood, tense, person and make nested dict
    organized_conjugations = {k: v for k, v in sorted(organized_conjugations.items(), key=lambda item: item[0])}
    return organized_conjugations


organized_conjugations = get_all_conjugations_for_infinitive('hablar')

print(organized_conjugations)
