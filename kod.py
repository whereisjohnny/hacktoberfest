# pip install transformers tensorflow  # lub pip install transformers torch, w zależności od wybranego frameworku

import pandas as pd

# Wczytaj dane z pliku CSV (zmień na swoje dane)
data = pd.read_csv('dane_wyborcze.csv')

# Podziel dane na cechy i etykiety
X = data['Wiadomość'].values
y = data['Preferencje'].values

from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification, TFAutoModelForSequenceClassification

# Wybierz model GPT-3.5 w języku polskim
model_name = "sberbank-ai/rugpt3.5-turbo"

# Wczytaj tokenizator i model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

# Tokenizacja danych
inputs = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors="tf")

# Trenowanie modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(inputs, y, epochs=5, batch_size=32, validation_split=0.2)

# Wczytaj nowe dane do klasyfikacji
nowe_dane = ["Jestem za zwiększeniem wydatków na służbę zdrowia.", "Popieram obniżenie podatków dla przedsiębiorców."]

# Tokenizacja nowych danych
nowe_inputs = tokenizer(nowe_dane, padding=True, truncation=True, return_tensors="tf")

# Przewidywanie preferencji
predictions = model.predict(nowe_inputs)

# Dekodowanie przewidywań
predicted_labels = [tokenizer.convert_ids_to_tokens(pred.argmax()) for pred in predictions]

print("Przewidywane preferencje wyborcze:")
for i in range(len(nowe_dane)):
    print(f"Wiadomość: {nowe_dane[i]}")
    print(f"Przewidziana preferencja: {predicted_labels[i]}\n")