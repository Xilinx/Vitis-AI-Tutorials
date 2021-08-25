'''
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


from tensorflow import layers, nn


def customCNN(cnn_in, is_training, drop_rate):

    net = layers.conv2d(inputs=cnn_in, filters=32, kernel_size=3, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.conv2d(inputs=net, filters=32, kernel_size=3, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.conv2d(inputs=net, filters=32, kernel_size=3, strides=2, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.conv2d(inputs=net, filters=32, kernel_size=3, strides=2, padding='same')
    net = layers.batch_normalization(inputs=net, training=is_training)
    net = nn.relu(net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)

    net = layers.flatten(inputs=net)
    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)
    net = layers.dense(inputs=net, units=512)

    net = layers.dropout(inputs=net, rate=drop_rate, training=is_training)
    logits = layers.dense(inputs=net, units=10, activation=None)
    return logits

