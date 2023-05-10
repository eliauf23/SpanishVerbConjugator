from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'woooooooo'

class ConjugateForm(FlaskForm):
    infinitive = StringField('Infinitive', validators=[DataRequired()])
    tense = SelectField('Tense', choices=[('0', 'presente'), ('1', 'futuro'), ('2', 'imperfecto'), ('3', 'pretérito'), ('4', 'condicional'), ('5', 'presente perfecto'), ('6', 'futuro perfecto'), ('7', 'pluscuamperfecto'), ('8', 'condicional perfecto')], validators=[DataRequired()])
    mood = SelectField('Mood', choices=[('0', 'indicativo'), ('1', 'subjuntivo'), ('2', 'imperativo afirmativo'), ('3', 'imperativo negativo')], validators=[DataRequired()])
    person = SelectField('Person', choices=[('0', '1s'), ('1', '2s'), ('2', '3s'), ('3', '1p'), ('4', '2p'), ('5', '3p')], validators=[DataRequired()])
    submit = SubmitField('Conjugate')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ConjugateForm()
    if form.validate_on_submit():
        return conjugate(form)
    return render_template('conjugate.html', form=form)
# @app.route('/bulk_conjugator')
# def bulk_conjugator():
#     return render_template('conjugate.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

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
