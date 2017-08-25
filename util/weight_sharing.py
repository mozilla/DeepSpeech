#!/usr/bin/env/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.cluster import MiniBatchKMeans

class CodeBook():
    def __init__(self, num_cluster, learning_rate, feed_dict,  session=None):
        self.session = session
        self.num_clusters = {'h1': num_cluster, 'h2': int(num_cluster/2),  'h3': num_cluster, 'h5': int(num_cluster/2), 'h6': int(num_cluster/2),
                             'b1': num_cluster, 'b2': int(num_cluster/2), 'b3': num_cluster, 'b5': int(num_cluster/2),
                             'b6': 29,
                             'bidirectional_rnn/fw/basic_lstm_cell/weights': num_cluster,
                             'bidirectional_rnn/bw/basic_lstm_cell/weights': num_cluster,
                             'bidirectional_rnn/fw/basic_lstm_cell/biases': num_cluster,
                             'bidirectional_rnn/bw/basic_lstm_cell/biases': num_cluster}

        self.param_name = {'h1': 'h1:0', 'h2': 'h2:0', 'h3': 'h3:0', 'h5': 'h5:0', 'h6': 'h6:0',
                            'b1': 'b1:0', 'b2': 'b2:0', 'b3': 'b3:0', 'b5': 'b5:0', 'b6': 'b6:0',
                            'bidirectional_rnn/fw/basic_lstm_cell/weights': 'bidirectional_rnn/fw/basic_lstm_cell/weights:0',
                            'bidirectional_rnn/bw/basic_lstm_cell/weights': 'bidirectional_rnn/bw/basic_lstm_cell/weights:0',
                            'bidirectional_rnn/fw/basic_lstm_cell/biases': 'bidirectional_rnn/fw/basic_lstm_cell/biases:0',
                            'bidirectional_rnn/bw/basic_lstm_cell/biases': 'bidirectional_rnn/bw/basic_lstm_cell/biases:0'}

        self.codebook = {}

        self.batch_bh_factor = 200
        self.batch_lstm_factor = 500
        self.batch_b6_factor = 4

        self.batch_factor = {'h1': self.batch_bh_factor, 'h2': self.batch_bh_factor, 'h3': self.batch_bh_factor,
                             'h5': self.batch_bh_factor, 'h6': self.batch_bh_factor, 'b1': self.batch_bh_factor,
                             'b2': self.batch_bh_factor, 'b3': self.batch_bh_factor, 'b5': self.batch_bh_factor,
                             'b6': self.batch_b6_factor,
                             'bidirectional_rnn/fw/basic_lstm_cell/weights': self.batch_lstm_factor,
                             'bidirectional_rnn/bw/basic_lstm_cell/weights': self.batch_lstm_factor,
                             'bidirectional_rnn/fw/basic_lstm_cell/biases': self.batch_lstm_factor,
                             'bidirectional_rnn/bw/basic_lstm_cell/biases': self.batch_lstm_factor}
        self.learning_rate = learning_rate
        self.feed_dict = feed_dict

    def get_parameters(self):

        params = {}
        param_name = self.param_name
        if not self.session is None:
            # Retrieve parameters
            for key in param_name.keys():
                params[key] = self.session.graph.get_tensor_by_name(param_name[key])
        else:
            log_warning('Session should not be None here')

        return params


    def get_centroids(self, input_tensor, param_name):

        dim = 0
        biases = ['b1', 'b2', 'b3', 'b5', 'b6', 'bidirectional_rnn/fw/basic_lstm_cell/biases', 'bidirectional_rnn/bw/basic_lstm_cell/biases']
        if param_name in biases:
            dim = input_tensor.get_shape()[0]
        else:
            dim = input_tensor.get_shape()[1]

        # Reshape the tensor value to [num_samples, num_features=1] to compute the clusters
        x = np.array(input_tensor.eval(session=self.session), dtype=np.float64).reshape([-1, 1])

        # Define the batch size to nearest integer
        batch_size = int(len(x)/self.batch_factor[param_name]);

        # In case the sample set is less that num_clusters
        sample_set = 3 * batch_size
        if sample_set < self.num_clusters[param_name]:
            sample_set = 3 * self.num_clusters[param_name]

        # Perform Linear Initialization for the centroids
        initial_centroids = np.linspace(np.amin(x), np.amax(x), num=self.num_clusters[param_name]).reshape([-1, 1])

        # Compute centroids using MiniBatch KMeans cluster
        km = MiniBatchKMeans(init=initial_centroids, n_clusters=self.num_clusters[param_name], batch_size=batch_size, init_size=sample_set, n_init=1, max_no_improvement=10, verbose=0)

        # Reshape the predicted matrices back
        pred = np.array(km.fit_predict(x).reshape([-1, dim]), dtype=np.uint8)

        # Retrieve the cluster centers
        cent = km.cluster_centers_
        return cent, pred


    def generate_codebook_for_weight(self, param):

        # For each weight matrix compute the centroids
        codebk = {}
        for p in param.keys():
            cent, pred = self.get_centroids(param[p], p)
            codebk[p] = [cent, pred]

        return codebk

    def generate_codebook_for_gradient(self, param, weight):
        # For each parameter group the gradients by weights according to cluster index
        codebk = {}
        for p in param.keys():
            key = p[:-2]
            codebk[key] = np.array([np.sum(k) for k in [np.take(param[p].flatten(), y) for y in [np.where(weight[key][1].flatten() == x) for x in range(self.num_clusters[key])]]]).reshape([-1, 1])

        return codebk

    def get_gradients(self, avg_tower_gradients):

        grad = {}
        if not self.session is None:
            for gradient in avg_tower_gradients:
                if gradient[1].name in self.param_name.values():

                    # Retrieve the gradients from grad_and_var variables
                    grad[gradient[1].name] = self.session.run(gradient[0], **self.feed_dict)
        return grad

    def update_codebook(self, avg_tower_gradients):

        # Get the parameters and gradients
        params = self.get_parameters()
        grad = self.get_gradients(avg_tower_gradients)

        # Generate the codebook for current epoch weight and gradient
        codebook_weight = self.generate_codebook_for_weight(params)
        codebook_grad = self.generate_codebook_for_gradient(grad, codebook_weight)

        if not self.codebook.keys():
            self.codebook = codebook_weight

        # Fine tune the centroids in codebook
        for p in params.keys():
            self.codebook[p][0] = np.subtract(self.codebook[p][0], self.learning_rate * codebook_grad[p])


