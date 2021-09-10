from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.keras.preprocessing import image



# defininso o RNA
model = Sequential()

# função relu transforma tudo que é menor que zero pra zero
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# pega o mapa de caracteristicas e normaliza os valores entre 0 e 1
model.add(BatchNormalization())

# matriz do max pooling com as principais caracteristicas matrix 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# adicionando mais uma camada de convolução
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# altera a estrutura de dados para um vetor
model.add(Flatten())

# configuração da rede neural com 2 camadas ocultas
# adiciona a primeira camada oculta com 128 neuronios e 128 caracteristicas
# cada imagem tem 64x64 pixels de dimensão, cada pixel possui uma caracteristica
model.add(Dense(units=128, activation='relu'))

# o dropout para zerar algumas entradas da camada oculta, neste caso 20%
model.add(Dropout(0.2))

# criando mais uma camada oculta
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

# criando camada de saída da RNA
model.add(Dense(units=1, activation='sigmoid'))

# configurando atributos para o treinamento
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# carregando dados do treinamento
training_generate = ImageDataGenerator(rescale=1/255,
                                       rotation_range=7,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       height_shift_range=0.07,
                                       zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

training_base = training_generate.flow_from_directory('dataset/training_set',
                                                      target_size=(
                                                          64, 64),
                                                      batch_size=32,
                                                      class_mode='binary')

test_base = gerador_teste.flow_from_directory('dataset/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

model.fit_generator(training_base, steps_per_epoch=1800/32,
                    epochs=5, validation_data=test_base,
                    validation_steps=200/32)

model.save('./models')
model.save_weights('./checkpoint.h5')
