from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired
from conjugator.data_processing import DataProcessor
from conjugator.conjugator import VerbConjugation
import pandas as pd
from mlconjug3 import Conjugator


# def get_all_conjugations_for_infinitive(infinitive):
#     tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
#                  'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
#                  'condicional perfecto': 8}
#     TENSES = list(tense_map.keys())
#     conjugator = Conjugator(language='es')
#     verb = conjugator.conjugate(infinitive)
#     data = {
#         "infinitive": [],
#         "is_reflexive": [],
#         "verb_ending": [],
#         "stem": [],
#         "mood": [],
#         "tense": [],
#         "person": [],
#         "conjugation": []
#     }
#     subj_map = {"yo": "1s",
#                 "tú": "2s",
#                 "él": "3s",
#                 "nosotros": "1p",
#                 "vosotros": "2p",
#                 "ellos": "3p",
#                 }
#     for m in ['Indicativo', 'Subjuntivo', 'Condicional']:
#         for t in TENSES:
#             if t == 'imperfecto':
#                 tense_modif = 'pretérito imperfecto'
#             elif t == 'pretérito':
#                 tense_modif = 'pretérito perfecto simple'
#             else:
#                 tense_modif = t
#             tense_modif = m + ' ' + tense_modif
#             for p in subj_map.keys():
#                 try:
#                     conjugation = verb[m][tense_modif][p]
#                     data["infinitive"].append(infinitive)
#                     data["is_reflexive"].append(False)
#                     data["verb_ending"].append(infinitive[-2:])
#                     data["stem"].append(infinitive[:-2])
#                     if m == 'Condicional':
#                         mood = 'indicativo'
#                         tense = 'condicional'
#                     else:
#                         mood = m.lower()
#                         tense = t.lower()
#                     data["mood"].append(mood)
#                     data["tense"].append(tense)
#                     data["person"].append(subj_map[p])
#                     data["conjugation"].append(conjugation)
#                 except KeyError:
#                     continue
#
#     data = pd.DataFrame(data)
#     # drop tense = "futuro perfecto"
#     data = data[data["tense"] != "futuro perfecto"]
#     data_proc = DataProcessor(df=data)
#     conjugator = VerbConjugation(data_processor=data_proc)
#     conjugator.initialize(load_saved_model=True, path_to_saved_model="/Users/eliauf/PycharmProjects/VerbConjugator/saved_models/model_v8.pth", test_only=True)
#     conjugations = conjugator.test()
#
#     mood_map = {'indicativo': 0, 'subjuntivo': 1, 'imperativo afirmativo': 2, 'imperativo negativo': 3}
#     tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
#                  'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
#                  'condicional perfecto': 8}
#     person_map = {'1s': 0, '2s': 1, '3s': 2, '1p': 3, '2p': 4, '3p': 5}
#
#     inverted_mood_map = {value: key for key, value in mood_map.items()}
#     inverted_tense_map = {value: key for key, value in tense_map.items()}
#     inverted_person_map = {value: key for key, value in person_map.items()}
#     organized_conjugations = {}
#     for x in conjugations:
#         mtp, pred, target = x
#         mood, tense, person = mtp["mood"], mtp["tense"], mtp["person"]
#         # convert mood, tense, person to words
#         mood = inverted_mood_map[mood]
#         tense = inverted_tense_map[tense]
#         person = inverted_person_map[person]
#         print(mood, tense, person, pred, target)
#         organized_conjugations[(mood, tense, person)] = (pred, target)
#
#     # sort by mood, tense, person and make nested dict
#     organized_conjugations = {k: v for k, v in sorted(organized_conjugations.items(), key=lambda item: item[0])}
#     return organized_conjugations
#
#
#
#
#
#

import pandas as pd
from mlconjug3 import Conjugator


def get_all_conjugations_for_infinitive(infinitive):
    mood_map = {'indicativo': 0, 'subjuntivo': 1, 'imperativo afirmativo': 2, 'imperativo negativo': 3}
    tense_map = {'presente': 0, 'futuro': 1, 'imperfecto': 2, 'pretérito': 3, 'condicional': 4,
                 'presente perfecto': 5, 'futuro perfecto': 6, 'pluscuamperfecto': 7,
                 'condicional perfecto': 8}
    person_map = {'1s': 0, '2s': 1, '3s': 2, '1p': 3, '2p': 4, '3p': 5}

    mlconj3_mood_tense_mapping = {
        ('indicativo', 'presente'): 'Indicativo presente',
        ('indicativo', 'futuro'): 'Indicativo futuro',
        ('indicativo', 'imperfecto'): 'Indicativo pretérito imperfecto',
        ('indicativo', 'pretérito'): 'Indicativo pretérito perfecto simple',
        ('indicativo', 'condicional'): 'Condicional Condicional',
        ('subjuntivo', 'presente'): 'Subjuntivo presente',
        ('subjuntivo', 'futuro'): 'Subjuntivo futuro',
        # ('subjuntivo', 'imperfecto'): 'Subjuntivo Pretérito imperfecto 1',
        # ('subjuntivo', 'pretérito'): 'Subjuntivo pretérito perfecto',
    }

    subj_map = {"yo": "1s",
                "tú": "2s",
                "él": "3s",
                "nosotros": "1p",
                "vosotros": "2p",
                "ellos": "3p",
                }
    reverse_subj_map = {value: key for key, value in subj_map.items()}

    conjugator = Conjugator(language='es')
    verb = conjugator.conjugate(infinitive)
    # get all tenses from iterator
    all_tenses = []
    for x in verb.iterate():
        print(x)


    organized_conjugations = {}

    for m, mood_num in mood_map.items():
        for t, tense_num in tense_map.items():
            if (m, t) not in mlconj3_mood_tense_mapping:
                continue

            mlconj3_mood_tense = mlconj3_mood_tense_mapping[(m, t)]

            for p, person_num in person_map.items():
                # try:
                    mood_input = mlconj3_mood_tense.split(' ')[0]

                    tense_input = mood_input + ' ' + (' '.join(mlconj3_mood_tense.split(' ')[1:]))
                    person_input = reverse_subj_map[p]
                    print("mOOD:", mood_input)
                    print("tense:", tense_input)
                    conjugation = verb[mood_input][tense_input][person_input]
                    organized_conjugations[(m, t, p)] = (conjugation,)
                    # print(m, t, p, conjugation)
                # except KeyError as e:
                #     print(e)

                    continue

    organized_conjugations = {k: v for k, v in sorted(organized_conjugations.items(), key=lambda item: item[0])}
    print(organized_conjugations)
    return organized_conjugations


infinitive = "hablar"
conjugations = get_all_conjugations_for_infinitive(infinitive)
print(conjugations)



app = Flask(__name__)
app.config['SECRET_KEY'] = 'woooooooo'

class ConjugateForm(FlaskForm):
    infinitive = StringField('Infinitive', validators=[DataRequired()])
    tense = SelectField('Tense', choices=[('0', 'presente'), ('1', 'futuro'), ('2', 'imperfecto'), ('3', 'pretérito'), ('4', 'condicional'), ('5', 'presente perfecto'), ('6', 'futuro perfecto'), ('7', 'pluscuamperfecto'), ('8', 'condicional perfecto')], validators=[DataRequired()])
    mood = SelectField('Mood', choices=[('0', 'indicativo'), ('1', 'subjuntivo'), ('2', 'imperativo afirmativo'), ('3', 'imperativo negativo')], validators=[DataRequired()])
    person = SelectField('Person', choices=[('0', '1s'), ('1', '2s'), ('2', '3s'), ('3', '1p'), ('4', '2p'), ('5', '3p')], validators=[DataRequired()])
    submit = SubmitField('Conjugate')

class BulkConjugateForm(FlaskForm):
    infinitive = StringField('Infinitive', validators=[DataRequired()])
    submit = SubmitField('Conjugate')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ConjugateForm()
    if form.validate_on_submit():
        return conjugate(form)
    return render_template('conjugate.html', form=form)
@app.route('/bulk_conjugator', methods=['GET', 'POST'])
def bulk_conjugator():
    form = BulkConjugateForm()
    if form.validate_on_submit():
        return bulk_conjugate(form)
    return render_template('bulk_conjugate.html', form=form)

def bulk_conjugate(form):
    infinitive = form.infinitive.data
    conjugation_chart = get_all_conjugations_for_infinitive(infinitive)
    return render_template('conjugation_chart.html', conjugation_chart=conjugation_chart, infinitive=infinitive)


def conjugate(form):
    infinitive = form.infinitive.data
    tense = form.tense.data
    mood = form.mood.data
    person = form.person.data

    print(infinitive, tense, mood, person)

    conjugation = "hablo"
    is_in_dataset = True
    # todo: ^ conjugate with model!

    return render_template('conjugate.html', form=form, conjugation=conjugation, is_in_dataset=is_in_dataset)

if __name__ == '__main__':
    app.run(debug=True)
