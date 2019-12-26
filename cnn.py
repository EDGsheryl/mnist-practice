import tensorflow as tf

epoch = 30
activation = 'softplus'

network = tf.keras.models.Sequential()

network.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation, input_shape=(28, 28, 1)))
network.add(tf.keras.layers.MaxPooling2D(2, 2))

network.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))
network.add(tf.keras.layers.MaxPooling2D(2, 2))

network.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))

network.add(tf.keras.layers.Flatten())

network.add(tf.keras.layers.Dense(64, activation=activation))
network.add(tf.keras.layers.Dense(10, activation='softmax'))

network.summary()

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x = x.reshape((60000, 28, 28, 1))
x_val = x_val.reshape((10000, 28, 28, 1))
x, x_val = x / 255.0, x_val / 255.0

network.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
network.fit(x, y, epochs=epoch)

test_loss, test_acc = network.evaluate(x_val, y_val)
print("准确率为 %.4f %%" % test_acc * 100)
