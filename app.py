from flask import Flask, render_template, request
from conjugator.main
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired

class ConjugateForm(FlaskForm):
    infinitive = StringField('Infinitive', validators=[DataRequired()])
    tense = SelectField('Tense', choices=[('0', 'presente'), ('1', 'futuro'), ('2', 'imperfecto'), ('3', 'pretérito'), ('4', 'condicional'), ('5', 'presente perfecto'), ('6', 'futuro perfecto'), ('7', 'pluscuamperfecto'), ('8', 'condicional perfecto')], validators=[DataRequired()])
    mood = SelectField('Mood', choices=[('0', 'indicativo'), ('1', 'subjuntivo'), ('2', 'imperativo afirmativo'), ('3', 'imperativo negativo')], validators=[DataRequired()])
    person = SelectField('Person', choices=[('0', '1s'), ('1', '2s'), ('2', '3s'), ('3', '1p'), ('4', '2p'), ('5', '3p')], validators=[DataRequired()])
    submit = SubmitField('Conjugate')

app = Flask(__name__)
# conjugator = Conjugator()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/conjugate', methods=['POST'])
def conjugate():
    infinitive = request.form['infinitive']
    tense = request.form['tense']
    mood = request.form['mood']
    person = request.form['person']

    output, in_dataset = conjugator.conjugate_verb(infinitive, mood, tense, person)

    return render_template('conjugate.html', output=output, in_dataset=in_dataset)

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ConjugateForm()
    response = None
    if form.validate_on_submit():
        infinitive = form.infinitive.data
        tense = form.tense.data
        mood = form.mood.data
        person = form.person.data
        conjugation, in_vocab = conjugator.conjugate_verb(infinitive, mood, tense, person)
        response = {
            'infinitive': infinitive,
            'tense': tense,
            'mood': mood,
            'person': person,
            'conjugation': conjugation,
            'in_vocab': in_vocab
        }
    return render_template('index.html', form=form, response=response)



if __name__ == '__main__':
    app.run(debug=True)