#!/usr/bin/env python
# coding: utf-8

# In[45]:


import keras                         #Important Imports
from keras.applications.vgg16 import VGG16
from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import glob as glob
import matplotlib.pyplot as plt


# In[2]:


image_size=224   #resize image to this 
vgg=VGG16(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))  #add preprocessing layer
#vgg.summary()


# In[3]:


for layer in vgg.layers[:-1]:             # freeze all layer except last layer (ie 5th pooling layer)
    layer.trainable=False
#vgg.summary()
    


# In[4]:


#for layer in model.layers:                #layer trainable observation
    #print(layer,layer.trainable)


# In[5]:


train_path='/home/gaurav/Downloads/data/train'
valid_path='/home/gaurav/Downloads/data/val'                    # path to train and test data 


# In[6]:


folders=glob.glob("/home/gaurav/Downloads/data/train/*")    #useful for getting no of classes


# In[7]:


x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)    # no of classes=19 ie 19 softmax layer


# In[8]:


model=Model(inputs=vgg.input,outputs=prediction)   #model object
model.summary()          # model summary 


# In[74]:


#cost and optimization method
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
categories=['Airport','Beach','Bridge','Commercial','Desert','Farmland','footballField','Forest','Indusrial','Meadow','Mountain','Park','Parking','Pond','Port','railwayStation','Resedential','River','Viaduct']


# In[10]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(rescale=1./255, 
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  rotation_range=20,
                                )

test_datagen= ImageDataGenerator(rescale=1./255)


training_set= train_datagen.flow_from_directory( '/home/gaurav/Downloads/data/train',
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode='categorical'     
                                                )
test_set=test_datagen.flow_from_directory( '/home/gaurav/Downloads/data/val',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical' 
                                        )


# In[11]:


#fit the model 

r= model.fit_generator(
                         training_set,
                         validation_data=test_set,
                         epochs=10,
                         steps_per_epoch=len(training_set),
                         validation_steps=len(test_set)      )


# In[12]:


# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[13]:


# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[76]:


import tensorflow as tf

from keras.models import load_model


# In[78]:



model.save('Whu_trained.h5')


# In[ ]:




