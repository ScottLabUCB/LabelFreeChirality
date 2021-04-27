import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adadelta
import numpy as np
from numpy.random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
import h5py
from sklearn import utils
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = None)
parser.add_argument('--n_epoch_net1', type=int, default=1)
parser.add_argument('--n_epoch_net2', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--version_number', default=None)
parser.add_argument('--repeats', type=int, default=3)

args = parser.parse_args()

wrong_frac = args.noise_rate
version = args.version_number
name_frac = str(wrong_frac).split('.')[-1]
if len(name_frac) == 1:
    name_frac = name_frac + '0'


right_images = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/data/split_Chiral_D_Large_TIFF_Cropped_front_only.h5','r')['right_front'][:]
right_images2 = h5py.File('/global/scratch/cgroschner/chiral_nanoparticles/data/split_Chiral_D_Large_TIFF_Cropped_right_front2.h5','r')['right_front'][:]

right_images = np.concatenate((right_images,right_images2),axis=0)
left_images = []

for img in right_images:
    left_images.append(np.fliplr(img))
left_images = np.array(left_images)

left_labels = [[0,1] for i in left_images]
right_labels = [[1,0] for i in right_images]

clean_train_X = np.concatenate((right_images[:10],left_images[10:20]),axis=0)
clean_train_X = clean_train_X/clean_train_X.max()
clean_train_X = np.expand_dims(clean_train_X,axis=3)
clean_train_X = np.concatenate((clean_train_X,clean_train_X,clean_train_X),axis=3)
clean_train_Y = np.concatenate((right_labels[:10],left_labels[10:20]),axis=0)

right_images = right_images[20:]
left_images = left_images[20:]
left_labels = left_labels[20:]
right_labels = right_labels[20:]

train_index = int(0.8*right_images.shape[0])
val_index = int(0.5*(right_images.shape[0]-train_index))+train_index

right_train = right_images[:train_index]
left_train = left_images[:train_index]
right_test = right_images[train_index:val_index]
left_test = left_images[train_index:val_index]
right_val = right_images[val_index:]
left_val = left_images[val_index:]
X_val = np.concatenate((left_val,right_val),axis=0)
Y_val = np.concatenate((left_labels[val_index:],right_labels[val_index:]),axis=0)
X_val, Y_val = utils.shuffle(X_val, Y_val,random_state=0)
X_val = X_val/X_val.max()
X_val = np.expand_dims(X_val,axis=3)

split_train = int(right_train.shape[0]*wrong_frac)
split_test = int(right_test.shape[0]*wrong_frac)

for idx in np.arange(0,split_train):
    right_train[idx] = np.fliplr(right_train[idx])
    left_train[idx] = np.fliplr(left_train[idx])

for idx in np.arange(0,split_test):
    right_test[idx] = np.fliplr(right_test[idx])
    left_test[idx] = np.fliplr(left_test[idx])

X_train = np.concatenate((left_train,right_train),axis=0)
Y_train = np.concatenate((left_labels[:train_index],right_labels[:train_index]),axis=0)
X_test = np.concatenate((left_test,right_test),axis=0)
Y_test = np.concatenate((left_labels[train_index:val_index],right_labels[train_index:val_index]),axis=0)

X_train, Y_train = utils.shuffle(X_train, Y_train,random_state=0)
X_test, Y_test = utils.shuffle(X_test, Y_test,random_state=0)


X_train = X_train/X_train.max()
X_test = X_test/X_test.max()
X_train = np.expand_dims(X_train,axis=3)
X_test = np.expand_dims(X_test,axis=3)

X_val = np.concatenate((X_val,X_val,X_val),axis=3)
X_train = np.concatenate((X_train,X_train,X_train),axis=3)
X_test = np.concatenate((X_test,X_test,X_test),axis=3)

batch_size = args.batch_size
seed = 42

def random_180(img):
    turns = randint(0,2)
    turns = turns*2
    return np.rot90(img,turns)

clean_train_datagen = ImageDataGenerator(
        rotation_range = 5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False,
        preprocessing_function=random_180)

clean_test_datagen = ImageDataGenerator(
        rotation_range = 5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False,
        preprocessing_function=random_180)

clean_train_generator = clean_train_datagen.flow(clean_train_X, y=clean_train_Y, batch_size=batch_size,seed=seed)

train_datagen = ImageDataGenerator(
        rotation_range = 5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False,
        preprocessing_function=random_180)

test_datagen = ImageDataGenerator(
        rotation_range = 5,
        zoom_range=0.3,
        horizontal_flip=False,
        vertical_flip = False,
        preprocessing_function=random_180)

train_generator = train_datagen.flow(X_train, y=Y_train, batch_size=batch_size,seed=seed)
val_generator = test_datagen.flow(X_test,y=Y_test,batch_size=batch_size,seed=seed)

repeat_trainings = args.repeats
for r in range(repeat_trainings):
    save_weights =  '/global/scratch/cgroschner/chiral_nanoparticles/clean_net_'+ name_frac +'perror_weights_v'+ version +'_'+str(r)+'.h5'
    save_history =  '/global/scratch/cgroschner/chiral_nanoparticles/clean_net_'+ name_frac +'perror_history_v'+ version +'_'+str(r)+'.h5'
    modelE = keras.models.Sequential()
    modelE.add(Conv2D(64, (3, 3), input_shape=(128, 128,3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(64, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    modelE.add(Dense(64))
    modelE.add(Activation('relu'))
    modelE.add(Dropout(0.5))
    modelE.add(Dense(2))
    modelE.add(Activation('softmax'))

    modelE.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(lr=args.lr),
                  metrics=['accuracy'])

    #clean net training


    modelCheckpoint = ModelCheckpoint(save_weights,
                                      monitor = 'loss',
                                      save_best_only = True,
                                      mode = 'min',
                                      verbose = 2,
                                      save_weights_only = True)

    callbacks_list = [modelCheckpoint]

    modelE.fit_generator(
            clean_train_generator,
            steps_per_epoch=9,
            epochs=args.n_epoch_net1,
            verbose = 0,
            callbacks=callbacks_list)

    keras.backend.clear_session()

    modelE = keras.models.Sequential()
    modelE.add(Conv2D(64, (3, 3), input_shape=(128, 128,3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Conv2D(64, (3, 3)))
    modelE.add(Activation('relu'))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))

    modelE.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    modelE.add(Dense(64))
    modelE.add(Activation('relu'))
    modelE.add(Dropout(0.5))
    modelE.add(Dense(2))
    modelE.add(Activation('softmax'))

    modelE.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(lr=args.lr),
                  metrics=['accuracy'])
    # earlyStopping = EarlyStopping(monitor='val_loss',
    #                               patience=2,
    #                               verbose=2,
    #                               min_delta = 0.001,
    #                               mode='min',)

    modelCheckpoint = ModelCheckpoint(save_weights,
                                      monitor = 'val_loss',
                                      save_best_only = True,
                                      mode = 'min',
                                      verbose = 2,
                                      save_weights_only = True)
    callbacks_list = [modelCheckpoint]

    modelE.load_weights(save_weights)
    history = modelE.fit_generator(
            train_generator,
            steps_per_epoch=2870,
            epochs=args.n_epoch_net2,
            validation_data=val_generator,
            validation_steps=358,
            verbose = 0,
            callbacks=callbacks_list)
    modelE.save_weights(save_weights)
    h = h5py.File(save_history,'w')
    h_keys = history.history.keys()
    print(h_keys)
    for k in h_keys:
        h.create_dataset(k,data=history.history[k])
    h.close()

    print(modelE.evaluate(X_val,Y_val))
    keras.backend.clear_session()
