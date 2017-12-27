## emotion recognition
for a given image of a face, predict emotion. 
## usage
```
git clone https://github.com/mynameisvinn/emotion
python emotion.py  # webcam
```
## model
### architecture
```python
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                        input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
```
### dataset
the dataset consists of 28,709 examples of 48x48 pixel grayscale images of faces, categorized as one of the following: angry, disgust, fear, happy, sad, surprise, neutral.

