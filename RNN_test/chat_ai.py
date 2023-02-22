
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from numpy.lib.arraysetops import isin
import pickle
import numpy as np
from tensorflow import keras
import json
import random
from googlesearch import search
import urllib

model = keras.models.load_model('chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('ai_response.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag']== tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    msg_lower = msg.lower()
    if any(word in msg_lower for word in ['my name is', 'i am', "i'm", 'im']):
        if 'my name is' in msg_lower:
            name = msg[11:]
        elif 'i am' in msg_lower:
            name = msg[5:]
        elif "i'm" in msg_lower:
            name = msg[4:]
        else:
            name = msg[3:]
        ints = predict_class(msg, model)
        if not ints:
            res = "Sorry, I don't understand. Can you please rephrase your question?"
        else:
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}",name)
    elif any(word in msg_lower for word in ['find', 'search']):
        if 'find' in msg_lower:
            query = msg[5:]  # Extract the search query from the message
        else:
            query = msg[7:]
        results = search(query, num_results=5)  # Call the search() function from the googlesearch library
        if results:
            res = "Here are the top 5 search results for " + query + ":\n\n" + "\n\n".join(results)
        else:
            res = "Sorry, no results found for that query."
    else:
        ints = predict_class(msg, model)
        if not ints:
            res = "Sorry, I don't understand. Can you please rephrase your question?"
        else:
            res = getResponse(ints, intents)
    return res



#Creating GUI with tkinter
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#E1E1E1", font=("Helvetica", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
base.configure(bg="#212121")

#Create Chat window
ChatLog = Text(base, bd=0, bg="#212121", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Helvetica", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#21dede", activebackground="#309191", fg='#E1E1E1',
                    command=send)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="#E1E1E1", width="29", height="5", font="Helvetica")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
