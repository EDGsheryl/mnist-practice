import tensorflow as tf

batch_size = 200
epoch = 100
activation = 'softplus'


def preprocess(parm_x, parm_y):
    parm_x = tf.cast(parm_x, dtype=tf.int32)
    parm_y = tf.cast(parm_y, dtype=tf.int32)

    return parm_x, parm_y


(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batch_size).repeat(epoch)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batch_size, drop_remainder=True)

network = tf.keras.Sequential([tf.keras.layers.Dense(256, activation=activation),
                               tf.keras.layers.Dense(128, activation=activation),
                               tf.keras.layers.Dense(64, activation=activation),
                               tf.keras.layers.Dense(32, activation=activation),
                               tf.keras.layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-07)

for step, (x, y) in enumerate(db):
    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28 * 28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b]
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    # evaluate
    if step % 500 == 0:
        total, total_correct = 0., 0

        for _, (x_t, y_t) in enumerate(ds_val):
            # [b, 28, 28] => [b, 784]
            x_t = tf.reshape(x_t, (-1, 28 * 28))
            # [b, 784] => [b, 10]
            out = network(x_t)
            # [b, 10] => [b]
            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # bool type
            correct = tf.equal(pred, y_t)
            # bool tensor => int tensor => numpy
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x_t.shape[0]

        print(step * batch_size // epoch, 'Evaluate Acc:', total_correct / total)
