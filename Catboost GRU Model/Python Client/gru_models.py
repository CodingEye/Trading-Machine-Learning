import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tf2onnx



class GRUClassifier():
    def __init__(self, time_step, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.time_step = time_step
        self.classes_in_y = np.unique(self.y_train)


    def train(self, learning_rate=0.001, layers=2, neurons = 50, activation="relu", batch_size=32, epochs=100, loss="binary_crossentropy", verbose=0):

        self.model = Sequential()
        self.model.add(Input(shape=(self.time_step, self.X_train.shape[2]))) 
        self.model.add(GRU(units=neurons, activation=activation)) # input layer


        for layer in range(layers): # dynamically adjusting the number of hidden layers

            self.model.add(Dense(units=neurons, activation=activation))
            self.model.add(Dropout(0.5))

        self.model.add(Dense(units=len(self.classes_in_y), activation='softmax', name='output_layer')) # the output layer

        # Compile the model
        adam_optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=adam_optimizer, loss=loss, metrics=['accuracy'])
        

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[early_stopping], verbose=verbose)

        val_loss, val_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=verbose)

        print("Gru accuracy on validation sample = ",val_accuracy)

            
    def to_onnx(self, model_name, standard_scaler):

        # Convert the Keras model to ONNX
        spec = (tf.TensorSpec((None, self.time_step, self.X_train.shape[2]), tf.float16, name="input"),)
        self.model.output_names = ['outputs']

        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)

        # Save the ONNX model to a file
        with open(model_name, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Save the mean and scale parameters to binary files
        standard_scaler.mean_.tofile(f"{model_name.replace('.onnx','')}.standard_scaler_mean.bin")
        standard_scaler.scale_.tofile(f"{model_name.replace('.onnx','')}.standard_scaler_scale.bin")
