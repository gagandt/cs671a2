import tensorflow as tf
import numpy as np


tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 3), name='input_x')
y1 =  tf.placeholder(tf.float32, shape=(None, 1), name='length_y')
y2 =  tf.placeholder(tf.float32, shape=(None, 1), name='width_y')
y3 =  tf.placeholder(tf.float32, shape=(None, 1), name='color_y')
y4 =  tf.placeholder(tf.float32, shape=(None, 12), name='angle_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))

    # 1
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 2
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    flat = tf.contrib.layers.flatten(conv2_bn)

    length = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=16, activation_fn=tf.nn.relu)
    length = tf.nn.dropout(length, keep_prob)
    length = tf.layers.batch_normalization(length)
    length = tf.contrib.layers.fully_connected(inputs=length, num_outputs=1, activation_fn=tf.nn.sigmoid)
    

    width = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=16, activation_fn=tf.nn.relu)
    width = tf.nn.dropout(width, keep_prob)
    width = tf.layers.batch_normalization(width)
    width = tf.contrib.layers.fully_connected(inputs=width, num_outputs=1, activation_fn=tf.nn.sigmoid)
    

    color = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=16, activation_fn=tf.nn.relu)
    color = tf.nn.dropout(color, keep_prob)
    color = tf.layers.batch_normalization(color)
    color = tf.contrib.layers.fully_connected(inputs=color, num_outputs=1, activation_fn=tf.nn.sigmoid)
    

    angle = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.relu)
    angle = tf.nn.dropout(angle, keep_prob)
    angle = tf.layers.batch_normalization(angle)
    angle = tf.contrib.layers.fully_connected(inputs=angle, num_outputs=12, activation_fn=tf.nn.sigmoid)

    return length, width, color, angle


# Hyperparameters 

###############################
epochs = 10
batch_size = 128
no_of_batches = 96000/batch_size
keep_probability = 0.7
learning_rate = 0.001
##############################
## Vary these to give weights to some classifications
alpha = 1
beta = 1
gamma = 1
delta = 1
###############################

length_logits, width_logits, color_logits, angle_logits = conv_net(x, keep_prob)

# Confusion Here about the loss functions in tf
cost_length = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=length_logits, labels=y1))
cost_width = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=width_logits, labels=y2))
cost_color = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=color_logits, labels=y3))
cost_angle = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=angle_logits, labels=y4))

cost = alpha * cost_length + beta * cost_width + gamma * cost_color + delta * cost_angle

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred_len = tf.equal(tf.argmax(length_logits, 1), tf.argmax(y1, 1))
correct_pred_wid = tf.equal(tf.argmax(width_logits, 1), tf.argmax(y2, 1))
correct_pred_col = tf.equal(tf.argmax(color_logits, 1), tf.argmax(y3, 1))
correct_pred_ang = tf.equal(tf.argmax(angle_logits, 1), tf.argmax(y4, 1))


accuracy_length = tf.reduce_mean(tf.cast(correct_pred_len, tf.float32), name='length_accuracy')
accuracy_width = tf.reduce_mean(tf.cast(correct_pred_wid, tf.float32), name='width_accuracy')
accuracy_color = tf.reduce_mean(tf.cast(correct_pred_col, tf.float32), name='color_accuracy')
accuracy_angle = tf.reduce_mean(tf.cast(correct_pred_ang, tf.float32), name='angle_accuracy')

accuracy = accuracy_angle + accuracy_color + accuracy_length + accuracy_width
accuracy = accuracy / 4

def train_neural_network(session, optimizer, keep_probability, feature_batch, length_batch, width_batch, color_batch, angle_batch):
    session.run(
        optimizer, 
        feed_dict={
            x: feature_batch,
            y1: length_batch,
            y2: width_batch,
            y3: color_batch,
            y4: angle_batch,
            keep_prob: keep_probability
        }
    )

def print_stats(session, feature_batch, length_batch, width_batch, color_batch, angle_batch, cost, accuracy):
    loss = sess.run(
        cost, 
        feed_dict={
            x: feature_batch,
            y1: length_batch,
            y2: width_batch,
            y3: color_batch,
            y4: angle_batch,
            keep_prob: 1.
        }
    )
    valid_acc = sess.run(
        accuracy, 
        feed_dict={
            x: feature_batch,
            y1: length_batch,
            y2: width_batch,
            y3: color_batch,
            y4: angle_batch,
            keep_prob: 1.
        }
    )
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

from cus_load import data
dataset = data("output")
X_train, y_leng, y_widt, y_colo, y_angl = dataset.load()

from keras.utils import to_categorical
X_train = X_train.reshape(len(X_train),28,28,3)

y_angl = to_categorical(y_angl)

y_leng.reshape(96000,1)

print(y_leng.shape)

X_train = np.split(X_train, no_of_batches)
y_leng = np.split(y_leng, no_of_batches)
y_widt = np.split(y_widt, no_of_batches)
y_colo = np.split(y_colo, no_of_batches)
y_angl = np.split(y_angl, no_of_batches)


def load_preprocess_training_batch(batch_size, batch_number, total_batches):
    
    
    return X_train[batch_number], y_leng[batch_number], y_widt[batch_number], y_colo[batch_number], y_angl[batch_number]
    # Each of size 128 (Or anything)


with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        # Calculate this: 
        total_X_train = len(X_train) #Iska kuch karo. Ye value sahi aani chahiye
        n_batches = int(no_of_batches)

        for batch_number in range(1, n_batches + 1):
            #for batch_features, length_labels, width_labels, color_labels, angle_labels in load_preprocess_training_batch(batch_size):
            train_neural_network(sess, optimizer, keep_probability, X_train[batch_number], y_leng[batch_number], y_widt[batch_number], y_colo[batch_number], y_angl[batch_number])

            print('Epoch {:>2}, Line Dataset Batch {}:  '.format(epoch + 1, batch_i), end='')

            print_stats(sess, batch_features, length_labels, width_labels, color_labels, angle_labels, cost, accuracy)



##### load_preprocess_training_batch ==== Ye Tomar ya kisiko likhna hai"""