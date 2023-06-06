import tensorflow as tf
from models import CNN_2D_model, EEGNet, DeepConvNet, ShallowConvNet, CNN_1D_model
from data_loader import load_data

x, y = load_data(["data/robert_sajina", "data/zuza"],
                 labels=["left", "right", "jump", "none"],
                 shape=(60, 16))

model = CNN_1D_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x, y, epochs=10, batch_size=64, validation_split=0.3, shuffle=True)
