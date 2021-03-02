# hello_world.py
# -*- coding: utf-8 -*-
import os
import PySimpleGUI as sg


def callback_function1():
    sg.popup('Capture Model')
    os.system('capture-model.sh')
    print('Abriendo Captura')

def callback_function2():
    sg.popup('Detect Model')
    print('Abriendo Script de Deteccion')
def callback_function3():
    sg.popup('Model Train')
    print('Abriendo script de entrenamiento')


sg.theme('dark grey 9')
window = sg.Window(title="FAWN Vision bot   ", layout=[ [sg.Image('fawn1.png'),sg.Button('Captura Imagenes - Manual',key='capture'),sg.Button('Entrenar Modelo',key="train"),sg.Button('Detectar Inferencia',key='detect')] ], margins=(300, 200))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'capture':
        callback_function1()        # call the "Callback" function
    elif event == 'detect':
        callback_function2()        # call the "Callback" function
    elif event == 'train':
        callback_function3()        # call the "Callback" function
window.close()
