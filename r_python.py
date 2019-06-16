#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
import pandas as pd
import time

from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import os

# global parameters1
global timestep
timestep = 5
global batch_size
batch_size = 1
global train_length
train_length = 1300
global epochs
epochs = 5

# global parameters2
global unit1
unit1 = 4
global unit2
unit2 = 4
global unit3
unit3 = 168

# global parameters
global adam_lr
adam_lr = 0.003
global loss_func
loss_func = 'mean_squared_error'
global file_path
file_path = "./dt1.txt"

global data_from
data_from = 0

global data_to
data_to = 2000

global data_from_column
data_from_column = 1

global data_to_column
data_to_column = 1

global dataset
dataset = []

# Windows
root = Tk()
root.title('Deep Learning Platform')
root.resizable(False, False)
windowWidth = 1400
windowHeight = 1400
screenWidth, screenHeight = root.maxsize()


# load data set
def load_dataset():
    global file_path
    file_path = filedialog.askopenfilename()
    if (os.path.splitext(file_path)[-1] == ".txt"):
        for line in open(file_path):
            line = line.strip()
            dataset.append(float(line))
    if (os.path.splitext(file_path)[-1] == ".csv"):
        cvs_reader = pd.read_csv(file_path)
        # label = ttk.Text(monty3, width=80, text=cvs_reader.head(22))
        label = Text(monty5, width=80, height=30)
        label.insert(INSERT, cvs_reader)
        label.grid(row=0, column=0)
        train_set = cvs_reader.values
        for line in train_set:
            # line = line.strip()
            dataset.append(float(line))
    if (os.path.splitext(file_path)[-1] == ".xlsx"):
        xls_reader = pd.read_excel(file_path)
        #  label = ttk.Text(monty3, width=80, text=xls_reader.head(220))
        label = Text(monty5, width=80, height=30)
        label.insert(INSERT, xls_reader)
        label.grid(row=0, column=0)
        train_set = xls_reader.values
        for line in train_set:
            #  line = line.strip()
            dataset.append(float(line))


# Pages
tabControl = ttk.Notebook(root)
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text='DATA LOAD')
tabControl.pack(expand=1, fill='both')
tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text='PARAMETER CONTROL')
tabControl.pack(expand=1, fill='both')

# tags
monty1 = ttk.LabelFrame(tab1, text='BASIC_PARA', width=500, height=100)
monty1.grid(column=0, row=0, padx=5, pady=20)
monty2 = ttk.LabelFrame(tab1, text='OPERATION', width=500, height=100)
monty2.grid(column=0, row=2, columnspan=2, padx=5, pady=20)
monty3 = ttk.LabelFrame(tab1, text='RESULT_SHOW', width=500, height=100)
monty3.grid(column=1, row=0, rowspan=2, padx=5, pady=20)

monty5 = ttk.LabelFrame(tab2, text='DATA_SHOW', width=650, height=550)
monty5.grid(column=1, row=0, rowspan=3, padx=5, pady=20)
monty4 = ttk.LabelFrame(tab2, text='DATA_CHOSEN_ROW', width=200, height=50)
monty4.grid(column=0, row=0, padx=5)
monty6 = ttk.LabelFrame(tab2, text='DATA_CHOSEN_COLUMN', width=200, height=50)
monty6.grid(column=0, row=1, padx=5)
monty7 = ttk.LabelFrame(tab2, text='DATA_LOAD', width=200, height=50)
monty7.grid(column=0, row=2, padx=5)

# Parameters1
var_steptime = StringVar()
var_steptime.set(timestep)
text_timestep_lable = StringVar()

var_batch_size = StringVar()
var_batch_size.set(batch_size)
text_batchsize_lable = StringVar()

var_train_length = StringVar()
var_train_length.set(train_length)
text_train_length_label = StringVar()

var_epochs = StringVar()
var_epochs.set(epochs)
text_epochs_label = StringVar()

# Parameters2
var_unit1 = StringVar()
var_unit1.set(unit1)
text_unit1_lable = StringVar()

var_unit2 = StringVar()
var_unit2.set(unit2)
text_unit2_lable = StringVar()

var_unit3 = StringVar()
var_unit3.set(unit3)
text_unit3_label = StringVar()

# Parameters3
var_adam_lr = StringVar()
var_adam_lr.set(adam_lr)
text_adam_lr_label = StringVar()

var_loss_func = StringVar()

# Parameters4
var_data_from = StringVar()
var_data_from.set(data_from)
text_data_from = StringVar()

var_data_to = StringVar()
var_data_to.set(data_to)
text_data_to = StringVar()

var_data_from_column = StringVar()
var_data_from_column.set(data_from_column)
text_data_from_column = StringVar()

var_data_to_column = StringVar()
var_data_to_column.set(data_to_column)
text_data_to_column = StringVar()

var_adam_lr = StringVar()
var_adam_lr.set(adam_lr)
text_adam_lr_label = StringVar()


def show(content):
    if content.isdigit():
        return True
    else:
        return False


show_com = root.register(show)

# Entry1
entry_steptime = Entry(monty1, textvariable=var_steptime, width=10, validate='key',
                       validatecommand=(show_com, '%P')).grid(row=0, column=2, pady=5)
lb1 = Label(monty1, text='timestep', width=13).grid(row=0, column=1, pady=5)
entry_batch_size = Entry(monty1, textvariable=var_batch_size, width=10, validate='key',
                         validatecommand=(show_com, '%P')).grid(row=1, column=2, pady=5)
lb2 = Label(monty1, width=13, text='batch_size').grid(row=1, column=1, pady=10)
entry_train_length = Entry(monty1, textvariable=var_train_length, width=10, validate='key',
                           validatecommand=(show_com, '%P')).grid(row=2, column=2, pady=5)
lb3 = Label(monty1, width=13, text='train_length').grid(row=2, column=1, pady=5)

entry_epochs = Entry(monty1, textvariable=var_epochs, width=10, validate='key',
                     validatecommand=(show_com, '%P')).grid(row=3, column=2, pady=5)
lb4 = Label(monty1, width=13, text='epochs').grid(row=3, column=1, pady=5)

entry_adam_lr = Entry(monty1, textvariable=var_adam_lr, width=10, validate='key').grid(row=4, column=2, pady=5)
lb3 = Label(monty1, width=13, text='adam_lr').grid(row=4, column=1, pady=5)

loss_funcChosen = ttk.Combobox(monty1, width=20, textvariable=var_loss_func)
loss_funcChosen['values'] = (
    'mean_squared_error', 'mean_absolute_error', 'squared_hinge', 'binary_crossentropy', 'mape', 'msle', 'hinge',
    'categorical_crossentropy')
loss_funcChosen.grid(row=5, column=1, columnspan=2, pady=5)
loss_funcChosen.current(0)
loss_funcChosen.config(state='readonly')
# Entry2
entry_unit1 = Entry(monty1, textvariable=var_unit1, width=10, validate='key',
                    validatecommand=(show_com, '%P')).grid(row=6, column=2, pady=5)
lb1 = Label(monty1, text='Embedding', width=13).grid(row=6, column=1, pady=5)
entry_unit2 = Entry(monty1, textvariable=var_unit2, width=10, validate='key',
                    validatecommand=(show_com, '%P')).grid(row=7, column=2, pady=5)
lb2 = Label(monty1, width=13, text='Hidden').grid(row=7, column=1, pady=5)
entry_unit3 = Entry(monty1, textvariable=var_unit3, width=10, validate='key',
                    validatecommand=(show_com, '%P')).grid(row=8, column=2, pady=5)
lb3 = Label(monty1, width=13, text='Dense').grid(row=8, column=1, pady=5)

# Entry 3
entry_data_from = Entry(monty4, textvariable=var_data_from, width=10, validate='key',
                        validatecommand=(show_com, '%P')).grid(row=0, column=2, pady=5)
lb1 = Label(monty4, text='From', width=13).grid(row=0, column=1, pady=5)

entry_data_to = Entry(monty4, textvariable=var_data_to, width=10, validate='key',
                      validatecommand=(show_com, '%P')).grid(row=1, column=2, pady=5)
lb1 = Label(monty4, text='To', width=13).grid(row=1, column=1, pady=5)

# Entry 4
entry_data_from_column = Entry(monty6, textvariable=var_data_from_column, width=10, validate='key',
                               validatecommand=(show_com, '%P')).grid(row=0, column=2, pady=5)
lb1 = Label(monty6, text='From', width=13).grid(row=0, column=1, pady=5)

entry_data_to_column = Entry(monty6, textvariable=var_data_to_column, width=10, validate='key',
                             validatecommand=(show_com, '%P')).grid(row=1, column=2, pady=5)
lb1 = Label(monty6, text='To', width=13).grid(row=1, column=1, pady=5)


# var_modify
def var_modify():
    global timestep
    timestep = int(var_steptime.get())
    global batch_size
    batch_size = int(var_batch_size.get())
    global train_length
    train_length = int(var_train_length.get())
    global epochs
    epochs = int(var_epochs.get())
    global unit1
    unit1 = int(var_unit1.get())
    global unit2
    unit2 = int(var_unit2.get())
    global unit3
    unit3 = int(var_unit3.get())


def conf_selected():
    global data_from
    data_from = int(var_data_from.get())
    global data_to
    data_to = int(var_data_to.get())

    # global adam_lr
    # adam_lr = float(var_unit3.get())
    # global loss_func
    # loss_func = var_loss_func.get()


# Figure Show
global IMG
IMG = []
global counter
counter = 0
global label
im = Image.open('ai.jpeg')
img = ImageTk.PhotoImage(im)
label = Label(monty3, image=img)
label.grid(row=0, column=0)


# RNN Training Function
def training():
    im = Image.open('ai.jpeg')
    img = ImageTk.PhotoImage(im)
    label = Label(monty3, image=img)
    label.grid(row=0, column=0)

    RNN_units = [unit1, unit2, unit3]  # 442
    data_length = train_length + timestep + RNN_units[2]
    test_length = timestep + RNN_units[2]

    global dataset
    dataset = dataset[data_from:data_to]
    date = []
    for line in open("./dt2.txt", "r"):
        line = line.strip()
        date.append(line)

    len1 = data_length + RNN_units[2] + timestep - 1
    len2 = len(dataset)
    dataset = dataset[(len2 - len1):len2]
    length = len(dataset)
    dataset = np.array(dataset)
    old_dataset = dataset.copy()
    dataset = (dataset - np.min(dataset)) / (max(dataset) - min(dataset))

    values = []
    labels = []
    for i in range(length - timestep - RNN_units[2] + 1):
        temp1 = dataset[i:i + timestep]
        temp2 = dataset[i + timestep:i + timestep + RNN_units[2]]
        values.append(temp1)
        labels.append(temp2)
    print(len(values))
    x_train = values[0:train_length]
    x_test = values[(train_length):train_length + timestep + RNN_units[2]]
    y_train = labels[0:train_length]
    y_test = labels[(train_length):train_length + timestep + RNN_units[2]]

    x_train = np.reshape(x_train, (train_length, timestep, 1))
    y_train = np.reshape(y_train, (train_length, RNN_units[2]))
    x_test = np.reshape(x_test, (timestep + RNN_units[2], timestep, 1))
    y_test = np.reshape(y_test, (timestep + RNN_units[2], RNN_units[2]))

    print(x_train.shape)
    model = Sequential()
    model.add(LSTM(RNN_units[0], input_shape=(timestep, 1), return_sequences=True))
    model.add(LSTM(RNN_units[1], return_sequences=False))
    model.add(Dense(RNN_units[2]))
    model.summary()
    adam = keras.optimizers.adam(lr=adam_lr)

    model.compile(loss=loss_func, optimizer=adam, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    trainpredict = model.predict(x_test, verbose=1, batch_size=batch_size)

    trainpredict = trainpredict * (max(old_dataset) - min(old_dataset)) + min(old_dataset)
    y_test = y_test * (max(old_dataset) - min(old_dataset)) + min(old_dataset)

    y = old_dataset[timestep + train_length:timestep + train_length + RNN_units[2]]
    print(trainpredict.shape)
    print(y_test.shape)
    print(len(y_test[0]))

    plt.clf()
    # plot prediction and real result
    fig = plt.figure(0)
    # plt.plot(range(168),y_test[0],'ro-')
    # plt.plot(range(168),trainpredict[0],'g*:')
    #

    plt.plot(range(unit3), y_test[0], 'ro-')
    plt.plot(range(unit3), trainpredict[0], 'g*:')
    plt.legend(('real', 'pred'), loc='upper right')
    plt.savefig('3.png')
    plt.clf()

    fig = plt.figure(1)
    plt.title("Intend Perception Accuracy", fontproperties="Times New Roman")
    plt.ylabel("Intend Encode", fontproperties="Times New Roman")
    plt.xlabel("Epoch", fontproperties="Times New Roman")
    plt.plot(range(timestep + unit3), y_test[:, 0], 'ro-')
    plt.plot(range(timestep + unit3), trainpredict[:, 0], 'g*:')
    plt.legend(('real', 'pred'), loc='upper right')
    plt.savefig('1.png')
    plt.clf()

    # plot train process
    fig = plt.figure(2)
    plt.plot(history.history['acc'], label='training acc')
    plt.plot(history.history['val_acc'], label='val acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.savefig('2.png')

    clearIMG()
    loadFigure()
    dataset = []


def clearIMG():
    global IMG
    IMG = []


def loadFigure():
    for i in range(1, 4):
        im = Image.open(str(i) + '.png')
        img = ImageTk.PhotoImage(im)
        IMG.append(img)
        counter = 0
        label = Label(monty3, image=IMG[counter])
    label.grid(row=0, column=0)


def chimg():
    global label
    global counter
    if counter < 1:
        counter += 1
    else:
        counter = 0
    label.destroy()
    label = Label(monty3, image=IMG[counter])
    label.grid(row=0, column=0)


def figure_config():
    counter = 2
    label = Label(monty3, image=IMG[counter])
    label.grid(row=0, column=0)


# Operation
Button(monty7, text='LOAD DATASET', width=15, command=load_dataset, activeforeground='white',
       activebackground='green').grid(row=0, column=0, padx=20, pady=10)
Button(monty7, text='CONF SELECT', width=15, command=conf_selected, activeforeground='white',
       activebackground='green').grid(row=1, column=0, padx=20, pady=10)

Button(monty2, text='MODIFY PARAMETERS', width=15, command=var_modify, activeforeground='white',
       activebackground='green').grid(row=0, column=1, padx=26, pady=10)
Button(monty2, text='START TRAINING', width=15, command=training, activeforeground='white',
       activebackground='green').grid(row=0, column=2, padx=26, pady=10)
Button(monty2, text='NEXT RESULT', width=15, command=chimg, activeforeground='white', activebackground='green').grid(
    row=0, column=3, padx=26, pady=10)
Button(monty2, text='FIGURE CONFIG', width=15, command=figure_config, activeforeground='white',
       activebackground='green').grid(row=0, column=4, padx=26, pady=10)
mainloop()
