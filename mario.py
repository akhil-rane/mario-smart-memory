import gym
import ppaquette_gym_super_mario
import mss
import os
import mss.tools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
#from resizeimage import resizeimage
from scipy.misc import imread
import skimage.transform

from collections import deque
import random

session = tf.Session()

REPLAY_MEMORY = 50000
BATCH = 32

with tf.device('/cpu:0'):
    with tf.Session() as session:
        def preprocess(img):
            return np.mean(img[::1, ::1], axis=2).astype(np.uint8)

        def create_weights(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        def create_biases(size):
            return tf.Variable(tf.constant(0.05, shape=[size]))


        sct = mss.mss()
        monitor = {
            'top': 173,
            'left': 428,
            'width': 508,
            'height': 442,
        }


        actions = [ [0,0,0,0,0,0],
                    [1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1],
                    [0,1,0,0,1,0],
                    [0,1,0,0,0,1],
                    [0,0,0,1,1,0],
                    [0,0,0,1,0,1],
                    [0,1,0,0,1,1],
                    [0,0,0,1,1,1],
                    [0,0,1,0,0,1],]

        img_height = 224
        img_width = 256
        number_action = len(actions)
        batch_size = 32
        discount = .99
        memory_capacity = 10000
        learning_rate_initial = 0.00025
        decay = 0.9999
        decay_steps = 5

        epsilon = 1.0
        epsilon_min = 0.015
        epsilon_decay = 0.9999


        clip_grad = False
        min_grad = -10.0
        max_grad = 10.0


        #replay_memory = SarstReplayMemory(memory_capacity, [img_height, img_width, 1])

        replay_memory = deque()

        # input_prediction = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1], name='input_prediction')
        # actions = tf.placeholder(tf.uint8, shape=[None, number_action], name='actions')
        # # y_true_cls = tf.argmax(y_true, dimension=1)
        #
        filter_size_conv1 = 7
        num_filters_conv1 = 8

        filter_size_conv2 = 5
        num_filters_conv2 = 16

        filter_size_conv3 = 3
        num_filters_conv3 = 32
        #
        #
        # n_hidden_1 = 256
        #
        def create_convolutional_layer(input, conv_filter_size, num_filters, strides, num_channels):
            weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_channels,  num_filters])
            biases = create_biases(num_filters)
            layer = tf.nn.conv2d(input = input, filter = weights, strides = strides, padding='SAME')
            layer += biases
            layer = tf.nn.relu(layer)
            return layer
        #
        # def create_flatten_layer(layer):
        #     layer_shape = layer.get_shape()
        #     num_features = layer_shape[1:4].num_elements()
        #     layer = tf.reshape(layer, [-1, num_features])
        #     return layer
        #
        def create_fc_layer(input,
                     num_inputs,
                     num_outputs,
                     use_relu=True):

            weights = create_weights(shape=[num_inputs, num_outputs])
            biases = create_biases(num_outputs)
            layer = tf.matmul(input, weights) + biases
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

        net_input = {}


        # def create_target_network():
        #     net_input['target'] = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1], name='target')
        #     conv_layer_1 = create_convolutional_layer(input = net_input['target'],
        #                                               conv_filter_size = filter_size_conv1,
        #                                               num_filters = num_filters_conv1,
        #                                               strides = [1, 2, 2, 1],
        #                                               num_channels = 1)
        #     conv_pool_1 = tf.nn.max_pool(conv_layer_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        #
        #
        #     conv_layer_2 = create_convolutional_layer(input = conv_pool_1,
        #                                               conv_filter_size = filter_size_conv2,
        #                                               num_filters = num_filters_conv2,
        #                                               strides = [1, 2, 2, 1],
        #                                               num_channels = num_filters_conv1)
        #
        #     conv_pool_2 = tf.nn.max_pool(conv_layer_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        #
        #
        #     conv_layer_3 = create_convolutional_layer(input = conv_pool_2,
        #                                               conv_filter_size = filter_size_conv3,
        #                                               num_filters = num_filters_conv3,
        #                                               strides = [1, 2, 2, 1],
        #                                               num_channels = num_filters_conv2)
        #     conv_layer_3 = tf.contrib.layers.flatten(conv_layer_3)
        #
        #     fully_conn_1 = create_fc_layer(input = conv_layer_3,
        #                                    num_inputs = int(conv_layer_3.get_shape()[1]),
        #                                    num_outputs = 256)
        #     weights = create_weights(shape=[256, number_action])
        #     biases = create_biases(number_action)
        #     output_targets = tf.nn.bias_add(tf.matmul(fully_conn_1, weights), biases, name='output_targets',)
        #
        #     return output_targets

        def create_pediction_network():
            net_input['input_prediction'] = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1], name='input_prediction')
            conv_layer_1 = create_convolutional_layer(input = net_input['input_prediction'],
                                                      conv_filter_size = filter_size_conv1,
                                                      num_filters = num_filters_conv1,
                                                      strides = [1, 2, 2, 1],
                                                      num_channels = 1)
            conv_pool_1 = tf.nn.max_pool(conv_layer_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


            conv_layer_2 = create_convolutional_layer(input = conv_pool_1,
                                                      conv_filter_size = filter_size_conv2,
                                                      num_filters = num_filters_conv2,
                                                      strides = [1, 2, 2, 1],
                                                      num_channels = num_filters_conv1)

            conv_pool_2 = tf.nn.max_pool(conv_layer_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


            conv_layer_3 = create_convolutional_layer(input = conv_pool_2,
                                                      conv_filter_size = filter_size_conv3,
                                                      num_filters = num_filters_conv3,
                                                      strides = [1, 2, 2, 1],
                                                      num_channels = num_filters_conv2)

            conv_layer_3 = tf.contrib.layers.flatten(conv_layer_3)

            fully_conn_1 = create_fc_layer(input = conv_layer_3,
                                           num_inputs = int(conv_layer_3.get_shape()[1]),
                                           num_outputs = 256)

            weights = create_weights(shape=[256, number_action])
            biases = create_biases(number_action)


            output_prediction = tf.nn.bias_add(tf.matmul(fully_conn_1, weights), biases, name='output_prediction',)
            max_q_action = tf.argmax(output_prediction, axis=1)

            return output_prediction, max_q_action

        output_prediction, max_q_action = create_pediction_network()
        # output_targets = create_target_network()

        global_step = tf.placeholder(tf.int32, name="global_step")


        target_y = tf.placeholder(tf.float32, shape=[None], name="target_y")
        chosen_actions = tf.placeholder(tf.int32, shape=[None], name="chosen_actions")
        chosen_actions_one_hot = tf.one_hot(chosen_actions, number_action, name="chosen_actions_one_hot")
        predict_y = tf.reduce_sum(chosen_actions_one_hot * output_prediction, axis=1, name="predict_y")
        loss = tf.reduce_mean(tf.square(tf.subtract(target_y, predict_y)), name="loss")



        learning_rate = tf.train.exponential_decay(learning_rate_initial,
                                                       global_step,
                                                       decay_steps,
                                                       decay)

        #and pass it to the optimizer to train on this defined loss function
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam")

        optimizer = opt.minimize(loss)


        init_op = tf.global_variables_initializer()
        session.run(init_op)



        def choose_action(state):

            def _choose_to_explore():
                #returns true with a probability equal to that of epsilon
                return True if np.random.rand() < epsilon else False

            def _choose_random_action():
                r = np.random.randint(low=0, high=number_action) #high is 1 above what can be picked
                return actions[r]

            if _choose_to_explore():
                return _choose_random_action()
            else:
                #returns a tensor with the single best q action evaluated from prediction network
                #[0] returns this single value from the tensor
                a = max_q_action.eval({net_input['input_prediction'] : [state]})[0]
                return actions[a]

        env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
        observation = env.reset()

        done = False
        t = 0
        features = []
        f = []
        total_iterations = 0
        minibatches_run = 0
        state = None

        saver = tf.train.Saver()

        if os.path.isfile("/Users/akhil/game_artificial_intelligence/final_project/final_mario/mario-bot-sarst-replay/tmp/model.ckpt"):
            saver.restore(session, "/Users/akhil/game_artificial_intelligence/final_project/final_mario/mario-bot-sarst-replay/tmp/model.ckpt")


        while not done:
            env.render()

            if(state is not None):
                action =choose_action(state)

            else:
                action = actions[0]


            frame, reward, is_done, state_info = env.step(action)
            # frame = frame[: , :, 0:3]

            data = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            data = np.expand_dims(data, axis=2)

            # data = preprocess(frame)

            if state is not None:
                # replay_memory.add_to_memory(state,
                #                             actions.index(action),
                #                             reward,
                #                             data,
                #                             is_done)
                replay_memory.append((state,actions.index(action),reward,data,is_done))
                if len(replay_memory) > REPLAY_MEMORY:
                    replay_memory.popleft()


            state = data

            if len(replay_memory) >= batch_size:
                minibatch = random.sample(replay_memory, BATCH)
                mem_state, mem_action, mem_reward, mem_data, mem_is_terminal = zip(*minibatch)
                mem_q_value = output_prediction.eval({net_input['input_prediction'] : mem_data})
                max_q_value_future = np.max(mem_q_value, axis=1)
                mem_is_terminal = np.asarray(mem_is_terminal)
                mem_target_y = mem_reward + (discount * max_q_value_future * (1 - (mem_is_terminal*1)))

                _, report_predictions, lr, one_hot_actions, final_predictions, report_loss = session.run([optimizer, output_prediction, learning_rate, chosen_actions_one_hot, predict_y, loss], {
                    net_input['input_prediction'] : mem_state,
                    chosen_actions : mem_action, #and definitely the actions
                    target_y : mem_target_y,     #and the targets in the optimizer
                    global_step : total_iterations #and update our global step. TODO, maybe this should be self.global_step. make sure isn't incremented twice with minimize() call
                })
                minibatches_run = minibatches_run + 1


            total_iterations = total_iterations + 1
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if total_iterations % 2000 == 0:
                print(total_iterations)
                # self.copy_prediction_parameters_to_target_network()

            if total_iterations % 1000 == 0:
                print 'saving model'
                saver.save(session, "/Users/akhil/game_artificial_intelligence/final_project/final_mario/mario-bot-sarst-replay/tmp/model.ckpt")

            if is_done:
                env.reset()


session.close()
