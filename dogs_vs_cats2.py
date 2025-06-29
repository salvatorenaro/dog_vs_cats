
import os,shutil
from keras import models,layers,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
#Salvatore Naro 
#Per le spiegazioni visitate la repository dataset e andate sul file mnist.cnn

original_dataset_dir = '/Users/salvatore/Desktop/dataset_animali/train'
base_dir = '/Users/salva/Desktop/dogs_and_cats_smas' 
os.makedirs(base_dir,exist_ok=True)


train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)




train_cats_dir = os.path.join(train_dir,'cat')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dog')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cat')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir,'dog')
os.mkdir(validation_dogs_dir)

test_cats_dir  = os.path.join(test_dir,'cat')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dog')
os.mkdir(test_dogs_dir)



fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames =  ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames :
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames =  ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames :
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames =  ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames :
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    class_mode = 'binary',
    batch_size = 20
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode  = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    class_mode = 'binary',
    batch_size = 20
)

conv_base  = VGG16(weights = 'imagenet', include_top = False, input_shape=(150,150,3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])
history = model.fit(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)

conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-5),metrics=['acc'])
history_fine = model.fit(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)

test_loss,test_acc = model.evaluate(test_generator,steps=50)
print(f'test loss: {test_loss} test acc: {test_acc:.4f}')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

acc_fine = history_fine.history['acc']
val_acc_fine = history_fine.history['val_acc']
loss_fine = history_fine.history['loss']
val_loss_fine = history_fine.history['val_loss']

epochs = range(1, len(acc) + 1)
epochs_fine = range(len(acc) + 1, len(acc) + len(acc_fine) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs_fine, acc_fine, 'ro', label='Training acc ')
plt.plot(epochs_fine, val_acc_fine, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs_fine, loss_fine, 'ro', label='Training loss ')
plt.plot(epochs_fine, val_loss_fine, 'r', label='Validation loss ')
plt.title('Training and validation loss')
plt.legend()

plt.show()