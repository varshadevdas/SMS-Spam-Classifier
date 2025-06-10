from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer once
model = load_model("spam_classifier.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

def predict_spam(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    text = text.strip()
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN)
    pred_prob = model.predict(padded_seq)[0][0]
    label = "Spam" if pred_prob > 0.5 else "Not Spam"
    return label, pred_prob

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    input_text = ''
    if request.method == 'POST':
        input_text = request.form['sms_text']
        prediction, confidence = predict_spam(input_text)
    return render_template('index.html', prediction=prediction, confidence=confidence, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
