import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take words and tokenize them
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('ai_response.pkl','wb'))


# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words and lemmatize each word, which create the base words
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #if word match found in current pattern, create bag of words array with 1
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle training data and convert to np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Add this code to reshape the input data to have a third dimension
train_x = np.array(train_x)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
# Determine the maximum length of a sequence
max_sequence_len = len(train_x[0])
for seq in train_x:
    if len(seq) > max_sequence_len:
        max_sequence_len

# Create model - 2 layers. First layer RNN, second layer output layer
model = Sequential()
model.add(LSTM(128, input_shape=(max_sequence_len, 1)))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile model with Stochastic gradient descent with Nesterov accelerated gradient (SGD is a good fit for chatbot)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#reshape the training data
train_x = sequence.pad_sequences(train_x, maxlen=max_sequence_len)

#fitting and saving the model
hist = model.fit(train_x, np.array(train_y), epochs=300, batch_size=8, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
