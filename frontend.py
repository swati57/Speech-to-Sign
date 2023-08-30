import streamlit as st
from audio_recorder_streamlit import audio_recorder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import wavio as wv
import sounddevice as sd
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
import wave
from audiorecorder import audiorecorder
from moviepy.editor import *
import os
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write

import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import speechToText


st.title("Speech to Sign")
st.caption("Introducing an application that allows for spoken English to be seamlessly translated into Indian Sign Language. With this app, users can effortlessly record their voice and, with just a click, see their words come to life in sign language through a captivating window on their screen. ")

col1, col2 = st.columns([1,4])

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1cm;
                    padding-right: 1cm;
                }
        </style>
        """, unsafe_allow_html=True)

with col1:
    def record():

        fs = 44100  # Sample rate
        seconds = 10  # Duration of recording
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        wv.write("recording1.wav", myrecording, fs, sampwidth=2)
        samplerate, data = wavfile.read('recording1.wav')

        write('output1.wav', fs, data.astype(np.int16))  # Save as WAV file

    st.caption("Click to record you voice")
    st.button("Record", on_click= record)

    
    


with col2:
    def TextToSign(sentence):
        
        text = sentence
        print("Text is:"+ str(text))
        #tokenizing the sentence
        text.lower()
        #tokenizing the sentence
        words = word_tokenize(text)

        tagged = nltk.pos_tag(words)
        tense = {}
        tense["future"] = len([word for word in tagged if word[1] == "MD"])
        tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
        tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
        tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])



        #stopwords that will be removed
        stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])



        # removing stopwords and applying lemmatizing nlp process to words
        lr = WordNetLemmatizer()
        filtered_text = []
        for w,p in zip(words,tagged):
            if w not in stop_words:
                if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
                    filtered_text.append(lr.lemmatize(w,pos='v'))
                elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
                    filtered_text.append(lr.lemmatize(w,pos='a'))

                else:
                    filtered_text.append(lr.lemmatize(w))


        # adding the specific word to specify tense
        words = filtered_text
        temp=[]
        for w in words:
            if w=='I':
                temp.append('Me')
            else:
                temp.append(w)
        words = temp
        probable_tense = max(tense,key=tense.get)

        if probable_tense == "past" and tense["past"]>=1:
            temp = ["Before"]
            temp = temp + words
            words = temp
        elif probable_tense == "future" and tense["future"]>=1:
            if "Will" not in words:
                    temp = ["Will"]
                    temp = temp + words
                    words = temp
            else:
                pass
        elif probable_tense == "present":
            if tense["present_continuous"]>=1:
                temp = ["Now"]
                temp = temp + words
                words = temp


        filtered_text = []
    
        database=[]
        directory="assets"
        for filename in os.listdir(directory):
                f = os.path.basename(filename)
                database.append(f[0:-4].lower())

        if len(words)>0:
            for w in words:
                path = "A2SL\TextToSign\assets"+ w + ".mp4"

                if w.lower() in database:
                    w=w[0].upper()+w[1:].lower()
                    filtered_text.append(w)
                else:
                    for c in w:
                        filtered_text.append(c.upper())
                #otherwise animation of word
        
                

        words = filtered_text
        print(words)
        if len(words)==0:
            final_clip = VideoFileClip("assets\Delay.mp4",audio=True)
        else:    
        # return render(request,'animation.html',{'words':words,'text':text})
            final_clip = VideoFileClip("assets\\"+ words[0] +  ".mp4",audio=True)
            for w in range(1,len(words)):
                if not words[w].isalpha():
                    final_clip = VideoFileClip("assets\Delay.mp4",audio=True)
        
                clip2 = VideoFileClip("assets\\"+ words[w] +  ".mp4",audio=True)
                final_clip = concatenate_videoclips([final_clip,clip2],method="compose")

        final_clip.write_videofile("merged.mp4")
        st.video("merged.mp4",start_time=0)



    def predict(audio_file):
        r = sr.Recognizer()                                                    
        with sr.AudioFile(audio_file) as source:               
            audio = r.record(source, duration=50)                              
            try:
                text_output = r.recognize_google(audio, language='en-IN')
            except Exception as e:
                print("could not understand audio")                             
        return text_output

    def process():
        # words= speechToText.predict("output1.wav")
        # print(words)
        sentence = predict("output1.wav")
        print(sentence)
        TextToSign(sentence)

    st.button("Submit", on_click=process)
    

    

   
      
