# imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np

# path to images of with face mask and without.
# image names should be [your name]_with mask.jpg and [your name]_without_mask.jpg
name = 'Shreyas'
path = r'C:\Users\Administrator\PycharmProjects\Personal_Face_Mask_Detection_V3_RT'  # don't remove r

# data stores images as arrays
data = []
labels = []
count = 0
for imagePath in list(paths.list_images(path)):
    count += 1
    label = 'without-mask'
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image, mode='tf')  # mode is tf
    if str(name + '_without_mask.jpg') in str(imagePath):
        datagen = ImageDataGenerator(zoom_range=0.2, height_shift_range=40)
        i = 0
        lis = []
        img = load_img(imagePath, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        for batch in datagen.flow(x, batch_size=1):
            image = img_to_array(batch[0])
            image = preprocess_input(image, mode='tf')
            count += 1
            label = 'without-mask'
            data.append(image)
            labels.append(label)
            i += 1
            if i >= 100:
                break
    if str(name + '_with_mask.jpg') in str(imagePath):
        datagen = ImageDataGenerator(zoom_range=0.4, height_shift_range=40, brightness_range=[0.5, 1.5])
        i = 0
        lis = []
        img = load_img(imagePath, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        for batch in datagen.flow(x, batch_size=1):
            image = img_to_array(batch[0])
            image = preprocess_input(image, mode='tf')
            count += 1
            label = 'with-mask'
            data.append(image)
            labels.append(label)
            i += 1
            if i >= 500:
                break
    else:
        data.append(image)
        labels.append(label)
data = np.array(data, dtype="float32")
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print('Loading complete for ', count, ' images')
aug = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') > 1:
            print('\nLoss too high, try rerunning cell')
            self.model.stop_training = True
        if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
            print("\nStopping training with accuracy of >95%")
            self.model.stop_training = True


BS = 32
EPOCHS = 20
callback = myCallback()
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=[callback])
tf.keras.backend.clear_session()
path = r'C:\Users\Administrator\PycharmProjects\Personal_Face_Mask_Detection_V3_RT\model'
model.save(path)
# save model to be used in detector script
