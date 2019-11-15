#!/usr/bin/env python3
from datetime import datetime
import os
import tensorflow as tf
import advattack.filesystem.filesystem as fs
from advattack.attack import generate_attack

import seaborn as sns
import pandas as pd

directories = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house',
               'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
               'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'zero']
data_dir = 'resources/audio/'
output_dir = 'resources/results/'
target_label = 'yes'
graph_path = 'resources/tf_ckpts/conv_actions_frozen.pb'
max_query = 5000
input_node = 'wav_data:0'
output_node_name = 'labels_softmax:0'
population_size = 2
parameters = [2, 5, 10]
output_graphic_name = 'TablePopulation'
labels = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


def run():
    target_idx = [idx for idx in range(len(labels)) if labels[idx]==target_label]
    target_idx = target_idx[0]
    fs.load_graph(graph_path)
    with tf.Session() as sess:

        output_node = sess.graph.get_tensor_by_name(output_node_name)
        for directory in directories:
            output_directory = output_dir + directory + '/'
            data_directory = data_dir + directory + '/'
            wav_files_list = \
                [f for f in os.listdir(data_directory) if f.endswith(".wav")]
            for parameter in parameters:
                print('Current children: ' + str(parameter))
                data = []
                counter = 1
                for input_file in wav_files_list:
                    print(str(counter) + '/' + str(len(wav_files_list)))
                    x_orig = fs.load_audio_file(data_directory + input_file)
                    pbs = int(x_orig[34])
                    chunk_id = chr(int(x_orig[0])) + chr(int(x_orig[1])) + chr(int(x_orig[2])) + chr(int(x_orig[3]))
                    assert chunk_id == 'RIFF', 'ONLY RIIF format is supported'
                    assert pbs == 16, "Only PBS=16 is supported now"

                    attack_output, attack_fitness = generate_attack(input_file, x_orig, target_idx, target_label, sess,
                                                                    input_node, output_node, max_query,
                                                                    population_size, parameter, data, pbs)
#### save manipulated audio_files
                    #fs.save_audio_file(attack_output, output_dir + str(attack_fitness) + '---' + input_file)
                    counter = counter + 1

                df = pd.DataFrame(data, columns=['index', 'fitness', 'best', 'category', 'mu', 'input_file'])
                parameter_string = str(population_size) + '+' + str(parameter)
                path = output_directory + output_graphic_name + parameter_string + '.csv'
                df.to_csv(path)