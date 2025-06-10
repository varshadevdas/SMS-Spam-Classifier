# SMS Spam Classifier 📱🚫

This project uses an LSTM-based deep learning model to classify SMS messages as **Spam** or **Not Spam**.

## 🔧 Features

- Trained on UCI SMS dataset
- Cleaned, tokenized, and padded inputs
- Uses TensorFlow (LSTM)
- Model saved as `.h5` and tokenizer as `.pkl`
- Includes evaluation scripts
- Ready for deployment with Flask

## 📁 Structure

- `train_model.py`: Train the LSTM model
- `evaluate_model.py`: Accuracy, F1-score, Confusion Matrix
- `test_model.py`: Predict custom SMS text
- `spam_classifier.h5`: Trained model
- `tokenizer.pkl`: Tokenizer used for input processing

## 🧪 Test It

```bash
python test_model.py