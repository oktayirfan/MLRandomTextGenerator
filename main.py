import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSProp

filepath = tf.keras.utils.get_file("shakespeare.txt","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))

index_to_char = dict((i, c) for i, c in enumerate(characters))

max_length = 40
step = 3

sentences = []
next_characters = []

for i in range(0, len(text) - max_length, step):
  sentences.append(text[i:i+max_length])
  next_characters.append(text[i+max_length])

x = np.zeros((len(sentences), max_length, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i, t, char_to_index[char]] = 1
  y[i, char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSProp(lr=0.01))

model.fit(x, y, batch_size=256, epochs=4)

model.save('textgenerator.model')
def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)



def generate_text(length, temperature):
  start_index = random.randint(0, len(text) - max_length - 1)
  generated = ''
  sentence = text[start_index:start_index + max_length]
  generated += sentence
  
  for i in range(length):
    x_pred = np.zeros((1, max_length, len(characters)))
    for t, char in enumerate(sentence):
      x_pred[0, t, char_to_index[char]] = 1.

    predictions = model.predict(x_pred, verbose=0)[0]
    next_index = sample(predictions, temperature)
    next_characters = index_to_char[next_index]

    generated += next_characters
    sentence = sentence[1:]+next_characters
  return generated
print("Generating text..."")
print(generat_text(300, 0.5))

  



