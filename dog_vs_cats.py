#Importiamo le librierie
import os,shutil
from keras import models,layers,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import webbrowser


#Il Dataset scaricato dispone di 25000 imaggini,12500 per ogni classe.
#Potete scaricarlo da qua sotto basta rimuovere il canceletto davanti a webbrowser


#--------------------------------------------------------------------

#Scaricare il Dataset:

#webbrowser.open('www.kaggle.com/c/dogs-vs-cats/data') #->Apre il link per scaricare il dataset su Kaggle

#--------------------------------------------------------------------

#Definiamo il percorso originale e il percorso da creare
original_dataset_dir = '/Users/salva/OneDrive/Desktop/dog-vs-cats/train'#->Percorso originale scaricato da Kaggle
base_dir = '/Users/salva/OneDrive/Desktop/dogs_and_cats_small' #->Percorso in cui memorizzare il dataset compresso
os.makedirs(base_dir,exist_ok=True)

#-------------------------------------------------------------------
#Creiamo le directory per il dataset di addestramento, convalida e test
train_dir = os.path.join(base_dir,'train')                                                                                                                                                       
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')            #Directory separate per l'addestramento la convalida e il test    
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

#-------------------------------------------------------------------
#Directory con le imaggini dei cani e gatti per l'addestramento
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

#-----------------------------------------------------------------------
#Directory con le imaggini dei cani e gatti per la convalida

validation_cat_dir  = os.path.join(validation_dir,'cats')
os.mkdir(validation_cat_dir)
validation_dog_dir  = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dog_dir)
#------------------------------------------------------------------------
#Directory con le imaggini dei cani e gatti per il test
test_cats_dir = os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)
#-----------------------------------------------------------------------

#Copia le prime 1000 imaggini di gatti in train_cats_dir
fnames  = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
#-----------------------------------------------------------------------

#Copia la successione di 500 imaggini di gatti in validation_cat_dir
fnames  = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cat_dir,fname)
    shutil.copyfile(src,dst)
#-------------------------------------------------------------------------

#Copia la successione di 500 imaggini di gatti in test_cats_dir
fnames  = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)
#---------------------------------------------------------------------------

#Copia le prime 1000 imaggini di cani in train_dog_dir
fnames  = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
#-----------------------------------------------------------------------

#Copia la successione di 500 imaggini di cani in validation_dog_dir
fnames  = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dog_dir,fname)
    shutil.copyfile(src,dst)
#-------------------------------------------------------------------------

#Copia la successione di 500 imaggini di cani in test_dogs_dir
fnames  = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)
#---------------------------------------------------------------------------
#Creiamo il nostro modello

### Le varie descrizioni le trovate nel file dataset_mnist_cnn.py che si trova nella stessa repository di questo file ###

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu',padding='same', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#-------------------------------------------------------------------------------------
### visualizziamo l'architettura del modello ###

model.summary()
#-----------------------------------------------------------------------------------

#Compiliamo il nostro modello
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])
#----------------------------------------------------------------------------------------------------------

#Ri-scala tutte le imaggini di 1/255
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

for batch_data, labels_batch in train_generator:
    print(f"data batch: {batch_data.shape}")
    print(f"labels batch: {labels_batch.shape}")
    break
#----------------------------------------------------------------------------------------------

history = model.fit_generator(train_generator,steps_per_epoch= 100, epochs = 30, validation_data = validation_generator, validation_steps = 50)
model.save('cats_and_dogs_small_1.h5')
#-----------------------------------------------------------------------------------------------------

acc = history.history['acc']
val_Acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs,val_Acc,'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,'bo', label='Training loss')
plt.plot(epochs,val_loss,'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
