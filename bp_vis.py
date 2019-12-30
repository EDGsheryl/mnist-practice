import datetime

import tensorflow as tf

epoch = 30
activation = 'softplus'

network = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=activation),
    tf.keras.layers.Dense(128, activation=activation),
    tf.keras.layers.Dense(64, activation=activation),
    tf.keras.layers.Dense(32, activation=activation),
    tf.keras.layers.Dense(10, activation='softmax')
])
network.summary()

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

network.compile(optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象

network.fit(
    x,
    y,
    epochs=epoch,
    validation_data=(x_val, y_val),
    callbacks=[callback]
)

test_loss, test_acc = network.evaluate(x_val, y_val)
