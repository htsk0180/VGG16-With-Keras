# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 07:01:15 2022

@author: siriu
"""
# Kütüphaneler.
import keras,os
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Verinin Yüklenmesi.
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="kediler-kopekler/train", target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="kediler-kopekler/test",target_size=(224,224))

# Modelin Oluşturulması. CNN üzerinde pooling işleminin yapılması.
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'] , optimizer=opt)
model.summary()

# ModelCheckpoint nesnesi ile val_acc parametresi takip edilir.ü
# Modelin en yüksek başarı değerine ulaştığı noktada model kaydedilir.
# Ve bir sonraki tahminlerde model tekrardan eğitilerek kaynak tüketilmemiş olur.
# EarlyStopping nesnesi ile modelimizin bir noktadan sonra kendini eğitemez durumda gelirse eğitime son verilir.
'''
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
'''

# Modelin Eğitilmesi.
hist = model.fit(steps_per_epoch=100,x=traindata, validation_data= testdata, validation_steps=1,epochs=1)
# Model kaydedilecekse ya da earlystopping uygulanacaksa fit içerisine parametre olarak callbacks=[checkpoint,early] verilir.

'''
# Sonuçların Çizilmesi.
import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
'''

# Modelin Test Edilmesi.
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
img = image.load_img("test4.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
# Modelimiz yukarıda kaydedilmiş ise 
# saved_model = load_model("vgg16_1.h5")
# output = saved_model.predict(img) kullanımı da mümkündür.
output = model.predict(img)
if output[0][0] > output[0][1]:
    print("kedi")
else:
    print('köpek')
