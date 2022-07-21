import tensorflow as tf
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import string
from string import digits
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input,LSTM,Dense

input_words=[]
target_words=[]
max_input_length=0
max_target_length=0
num_en_chars=0
num_dec_chars=0

cv=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char') 



#get all data from datafile and load the model.
datafile = pickle.load(open("training_data.pkl","rb"))
input_words = datafile['input_words']
target_words = datafile['target_words']
max_input_length = datafile['max_input_length']
max_target_length = datafile['max_target_length']
num_en_chars = datafile['num_en_chars']
num_dec_chars = datafile['num_dec_chars']


#Inference model
#load the model
model = models.load_model("s2s")
#construct encoder model from the output of second layer
#discard the encoder output and store only states.
enc_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
#add input object and state from the layer.
enc_model = Model(model.input[0], [state_h_enc, state_c_enc])

#create Input object for hidden and cell state for decoder
#shape of layer with hidden or latent dimension
dec_state_input_h = Input(shape=(300,), name="input_3")
dec_state_input_c = Input(shape=(300,), name="input_4")
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

#add input from the encoder output and initialize with 
#states.
dec_lstm = model.layers[3]
dec_outputs, state_h_dec, state_c_dec = dec_lstm(
    model.input[1], initial_state=dec_states_inputs
)
dec_states = [state_h_dec, state_c_dec]
dec_dense = model.layers[4]
dec_outputs = dec_dense(dec_outputs)
#create Model with the input of decoder state input and encoder input
#and decoder output with the decoder states.
dec_model = Model(
    [model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states
)

def decode_sequence(input_seq):
    #create dict object to get character from the index.
    target_token_index = dict(enumerate(input_words))
    reverse_target_char_index = dict(enumerate(target_words))

        #get the index and from dictionary get character from it.
        # char_index = np.argmax(output_chars[0, -1, :])
        # text_char = reverse_target_char_index[char_index]
        # decoded_sentence += text_char         
        # target_seq[0, 0, char_index] = 1.0
   
    # Encode the input as state vectors.
    states_value = enc_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.

    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_chars, h, c = dec_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_chars[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        if (sampled_char == '_END' or
           len(decoded_sentence) > max_target_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

def filtering(input_sentence):
    digit = list(range(10))
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
    for ele in input_sentence:
        if ele in punc:
           input_sentence = input_sentence.replace(ele, "")
        if digit in input_sentence: 
            input_sentence= input_sentence.replace(ele, "")
    

def filtering_the_input(input_sentence):
    input_sentence  = input_sentence.lower()
   
    input_sentence  = re.sub("'", '', input_sentence)
    input_sentence  = re.sub('”', '', input_sentence)
    input_sentence  = re.sub('“', '', input_sentence)
    input_sentence  = re.sub('"', '', input_sentence)
    input_sentence  = re.sub('"', '', input_sentence)
    exclude = set(string.punctuation) 
    input_sentence  = ''.join(ch for ch in input_sentence if ch not in exclude)
    remove_digits = str.maketrans('', '', digits)
    input_sentence  = input_sentence.translate(remove_digits)
    input_sentence  = input_sentence.strip()
    input_sentence  = re.sub(" +", " ", input_sentence)
    print("the input sentence is " +input_sentence)
    return input_sentence
    
def bag_of_words(input_t):
    cv=CountVectorizer(binary=True,tokenizer=lambda txt: txt.split(),stop_words=None,analyzer='char') 
    en_in_data=[]; 
    pad_en=[1]+[0]*(len(input_words)-1)

    cv_inp= cv.fit(input_words)
    en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

    if len(input_t)<max_target_length:
        for _ in range(max_target_length-len(input_t)):
            en_in_data[0].append(pad_en)
            

    return np.array(en_in_data,dtype="float32")

def predict(fr_in_data):
    en_in_data = bag_of_words(fr_in_data.lower()+".")
    print(input_words,target_words,max_input_length,max_target_length,num_en_chars,num_dec_chars)
    y_pred = decode_sequence(en_in_data)
    return y_pred    

    