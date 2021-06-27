from tkinter import *
import nltk
import speech_recognition as sr
import time
import pyttsx3
import webbrowser
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('luna_model.h5')
import json
import random
convos = json.loads(open('convo.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

r = sr.Recognizer()
def record_audio():
    with sr.Microphone() as source:
        audio = r.listen(source)
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)
        except sr.UnknownValueError:
            engine.say("Sorry, I did not get that")
        except sr.RequestError:
            engine.say("Sorry, my speech service is down")
        return voice_data

engine = pyttsx3.init()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

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
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"convo": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, convo_json):
    tag = ints[0]['convo']
    list_of_convos = convo_json['convos']
    for i in list_of_convos:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, convos)
    return res

def send_audio():
    msg = entry0.get("1.0",'end-1c').strip()
    entry0.delete("0.0",END)
    voice_data = record_audio()

    if voice_data != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + voice_data + '\n\n')
        ChatLog.config(foreground="#442265", font=("Helvetica", 10, "bold" ))

        res = chatbot_response(voice_data)
        ChatLog.insert(END, "Neo: " + res + '\n\n')
        engine.say(res)
        engine.runAndWait()

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

    def there_exists(terms):
        for term in terms:
            if term in voice_data:
                return True

    if there_exists(["search for"]) and 'youtube' not in voice_data:
        search_term = voice_data.split("for")[-1]
        url = f"https://google.com/search?q={search_term}"
        webbrowser.get().open(url)

    if there_exists(["youtube"]):
        search_term = voice_data.split("for")[-1]
        url = f"https://www.youtube.com/results?search_query={search_term}"
        webbrowser.get().open(url)

def send():
    msg = entry0.get("1.0",'end-1c').strip()
    entry0.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Helvetica", 10, "bold" ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Luna: " + res + '\n\n')
        engine.say(res)
        engine.runAndWait()

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

    def there_exists(terms):
        for term in terms:
            if term in msg:
                return True

    if there_exists(["search for"]) and 'youtube' not in msg:
        search_term = msg.split("for")[-1]
        url = f"https://google.com/search?q={search_term}"
        webbrowser.get().open(url)

    if there_exists(["youtube"]):
        search_term = msg.split("for")[-1]
        url = f"https://www.youtube.com/results?search_query={search_term}"
        webbrowser.get().open(url)



window = Tk()
window.title("Luna")
window.geometry("375x650")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 650,
    width = 375,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    141.5, 598.0,
    image = entry0_img)

#Create Chat window
ChatLog = Text(bd=0, bg= "#F3F3F3", height="435", width="321", font="Helvetica",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(window, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


entry0 = Text(
    window,
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry0.place(
    x = 37.0, y = 578,
    width = 209.0,
    height = 38)

img0 = PhotoImage(file = f"img0.png")
SendButton = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = send,
    relief = "flat")

SendButton.place(
    x = 262, y = 578,
    width = 40,
    height = 40)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = send_audio,
    relief = "flat")

b1.place(
    x = 308, y = 578,
    width = 40,
    height = 40)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    187.5, 294.5,
    image=background_img)

scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=27,y=103, height=435, width=321)

window.resizable(False, False)
window.mainloop()
