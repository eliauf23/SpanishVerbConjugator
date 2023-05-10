from spellchecker import SpellChecker
import re

# def truncate_repeated_chars(text, min_repeats=3):
#     # Define a pattern that matches sequences of characters that repeat at least 'min_repeats' times
#     pattern = re.compile(r'(.)\1{%d,}' % (min_repeats - 1))
#     # Replace the repeated characters with a single instance of the character
#     truncated_text = pattern.sub(r'\1', text)
#
#     # Additionally, remove sequences of non-alphabetic characters
#     # truncated_text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+', ' ', truncated_text)
#
#     return truncated_text


def post_process(predictions):
    corrected_predictions = []
    spell_checker = SpellChecker(language='es')

    for prediction in predictions:
        # prediction = truncate_repeated_chars(prediction)
        # Tokenize the prediction into words
        words = prediction.split()
        # Correct each word using the spell checker
        corrected_words = [spell_checker.correction(word) for word in words]
        # Join the corrected words back into a sentence
        corrected_prediction = ' '.join(corrected_words)
        corrected_predictions.append(corrected_prediction)

    return corrected_predictions

# # HOW TO USE
# Apply post-processing to the model's predictions
# corrected_predictions = post_process(predictions)
#
# for x in range(len(corrected_predictions)):
#   if corrected_predictions[x] != predictions[x]:
#     print(f"CHANGED: (orig) {predictions[x]}, (new) {corrected_predictions[x] }")
