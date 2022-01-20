import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
# json file contains predefined patterns and responses

# Next Step: Tokenize the sentence
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # make a document containing word and the corresponding tag
        documents.append((w, intent['tag']))

        # add tag to classes array
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize the word, converting it into lower case and removing duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # remove duplicates
# sort classes
# documents = combination between patterns and intents
classes = sorted(list(set(classes)))
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words
print(len(words), "unique lemmatized words", words)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# creating training data
training = []
# empty array for output
output_empty = [0]*len(classes)
# in training set, we create a bag of words for each sentence
for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word
    # creating base word so that it can represent related words as well using lemmatizer
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        # we create bag of words array with 1 if word match found in current pattern
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

    # shuffle features and into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

#creating training and testing lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("training data created")

# First layer has 128 neurons, 2nd layer has 64 and 3rd layer has number of neurons equal to number of intents to predict
#output with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#training sgd model

sgd= SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov= True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics =['accuracy'])

#fitting and saving the model

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")

















