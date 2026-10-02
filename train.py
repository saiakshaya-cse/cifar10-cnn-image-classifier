import tensorflow as tf

#Loading the cifar10 dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()


#printing the shape of the training and testing data and labels
print("Training data shape:",x_train.shape)
print("Training labels shape:",y_train.shape)
print("Testing data shape:",x_test.shape)
print("Testing labels shape:",y_test.shape)

  
#Normalizing pixel values to be between 0 and 1
x_train=x_train/255.0
x_test=x_test/255.0


#Data augmentation to make the model more robust and make sure that the model does not overfit on the training data
data_augumentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


#Building the CNN model using  the Sequential API
model=tf.keras.Sequential([
      tf.keras. Input(shape=(32,32,3)),
    data_augumentation,
    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
    ),

#Below layers are added  to make the model to detect more complex features
    tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(
      32,
      (3,3),
      padding='same',
      activation='relu'
  ),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(
    64,
    (3,3),
    padding='same',
    activation='relu'
),
 tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(
    64,(3,3),
    padding='same',
    activation='relu'
),
tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10,activation='softmax')
])

# Model architecture summary
model.summary()

#setting compilation parameters for the model
model.compile(
optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']    
)


# Adding callbacks for making the training process efficient and making sure that the best model is saved
early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
checkpoint=tf.keras.callbacks.ModelCheckpoint("best_model.keras",monitor="val_loss",save_best_only=True)
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6)


#Training the model using the training data and labels
history=model.fit(x_train,y_train,epochs=10,validation_split=0.2,callbacks=[early_stop,checkpoint,reduce_lr])
print("final training accuracy:",history.history['accuracy'][-1])
print("final training loss:",history.history['loss'][-1])
print("final validation accuracy:",history.history['val_accuracy'][-1])
print("final validation loss:",history.history['val_loss'][-1])
print(history.history.keys())


#Model evaluation using the testing data and labels
test_loss,test_accuracy=model.evaluate(x_test,y_test)


#printing the test results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)