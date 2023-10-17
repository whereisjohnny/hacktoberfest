# pip install transformers tensorflow

import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification

# Wczytaj dane z pliku CSV
data = pd.read_csv('dane_wyborcze.csv')

# Podziel dane na cechy i etykiety
X = data['Wiadomość'].values
y = data['Preferencje'].values

# Wybierz model PolBERT dla języka polskiego
model_name = "dkleczek/bert-base-polish-cased-v1"

# Wczytaj tokenizator i model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

# Tokenizacja danych
inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="tf")

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
predicted_labels = [tokenizer.decode(pred.argmax()) for pred in predictions.logits]

print("Przewidywane preferencje wyborcze:")
for i in range(len(nowe_dane)):
    print(f"Wiadomość: {nowe_dane[i]}")
    print(f"Przewidziana preferencja: {predicted_labels[i]}\n")
