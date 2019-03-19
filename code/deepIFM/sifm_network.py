'''
Tensorflow implementation of
Interaction-aware Factorization Machines for Recommender Systems (DeepIFM)
https://arxiv.org/abs/1902.09757
@author: 
cstur4@zju.edu.cn

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.layers import base
import collections
import copy
import re
import weakref

import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

@tf_export('estimator.SIFMNetwork')
class SIFMNetwork(estimator.Estimator):
    """Interaction-aware Factorization Machine with sampling"""

    def __init__(
            self,
            model_dir,
            model_params,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params):     
            num_ps_replicas = config.num_ps_replicas if config else 0
            input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas,
                min_slice_size=64 << 20)

            dnn_partitioner = (
                partitioned_variables.min_max_variable_partitioner(
                    max_partitions=num_ps_replicas))
            with variable_scope.variable_scope(
                    "IFM",
                    values=tuple(six.itervalues(features)),
                    partitioner=dnn_partitioner):

                with variable_scope.variable_scope(
                        'input_from_feature_columns',
                        values=tuple(six.itervalues(features)),
                        partitioner=input_layer_partitioner):

                    raw_embeddings = tf.feature_column.input_layer(
                        features=features,
                        feature_columns=model_params['deep_columns'])

                    feature_bias = tf.feature_column.linear_model(features, model_params['wide_columns'])

                embeddings = tf.reshape(raw_embeddings, [-1, model_params['num_field'], model_params['embedding_size']])
                embedding_size = embeddings.get_shape().as_list()[2]
                l2_reg = model_params["l2_reg"]
                learning_rate = model_params["learning_rate"]
                # batch_norm_decay = params["batch_norm_decay"]
                # optimizer = params["optimizer"]
                deep_layers = map(int, model_params["deep_layers"].split(','))

                # ------bulid weights------

                dim_to_field = model_params['dim_to_field']

                all_weights = dict()
                # if freeze_fm, set all other params untrainable

                # all_weights['feature_embeddings'] = tf.get_variable(name='feature_embeddings', shape=[model_params["num_feature"], embeddings.get_shape().as_list()[1]], initializer=tf.glorot_normal_initializer())
                #             all_weights['feature_bias'] = tf.get_variable(name='feature_bias', shape=[model_params["num_feature"]], initializer=tf.zeros_initializer())
                #             all_weights['bias'] = tf.get_variable(name='bias', shape=[1], initializer=tf.zeros_initializer())  # 1 * 1

                # attention
                interaction_factor_hidden = model_params['interaction_factor_hidden']
                all_weights['attention_W'] = tf.get_variable(name='attention_W',
                                                             shape=[embedding_size, model_params['attention_size']],
                                                             initializer=tf.glorot_normal_initializer())

                all_weights['attention_b'] = tf.get_variable(name='attention_b',
                                                             shape=[1, model_params['attention_size']],
                                                             initializer=tf.glorot_normal_initializer())

                all_weights['attention_p'] = tf.get_variable(name='attention_p', shape=[model_params['attention_size']],
                                                             initializer=tf.ones_initializer())

                self.valid_dimension = model_params['num_field']

                all_weights['interaction'] = tf.get_variable(name='interaction',
                                                             shape=[self.valid_dimension, interaction_factor_hidden],
                                                             initializer=tf.glorot_normal_initializer())
                all_weights['factor'] = tf.get_variable(name='factor',
                                                        shape=[interaction_factor_hidden, embedding_size],
                                                        initializer=tf.glorot_normal_initializer())
                all_weights['global_step'] = tf.Variable(0, trainable=False)

                self.weights = all_weights
                self.nonzero_embeddings = embeddings

                element_wise_product_list = []
                interactions = []
                count = 0
                for i in range(0, self.valid_dimension):
                    for j in range(i + 1, self.valid_dimension):
                        if dim_to_field[i] != dim_to_field[j]:
                            element_wise_product_list.append(
                                tf.multiply(self.nonzero_embeddings[:, i, :], self.nonzero_embeddings[:, j, :]))

                            interactions.append(tf.multiply(tf.gather(self.weights['interaction'], dim_to_field[i]),
                                                            tf.gather(self.weights['interaction'], dim_to_field[j])))
                            count += 1
                num_interaction = count
                self.element_wise_product = tf.stack(element_wise_product_list)  # (M'*(M'-1)) * None * K
                self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                         name="element_wise_product")  # None * (M'*(M'-1)) * K
                self.interactions = tf.reduce_sum(self.element_wise_product, 2, name="interactions")

                # _________ MLP Layer / attention part _____________
                self.num_interaction = num_interaction

                self.field_interactions = tf.stack(interactions)

                self.attention_interaction = tf.matmul(self.field_interactions, self.weights['factor'])

                num_samples = int(num_interaction * model_params.get('num_samples', 0.8))  # tf.where(tf.greater(tf.train.get_global_step(), 2000), 45, 25)

                analysis = tf.square(self.attention_interaction)
                norm = tf.reduce_sum(analysis, axis=1)
             
                # gumbel-max trick
                uniform = tf.random_uniform(norm.shape, minval=np.finfo(np.float32).tiny, maxval=1., dtype=np.float32)
                gumbel = -tf.log(-tf.log(uniform))

                noisy_logits = gumbel + tf.log(norm)

                top_k = tf.nn.top_k(noisy_logits, num_samples)

                # samples = tf.multinomial(tf.log([norm]), num_samples)
                samples = tf.reshape(top_k.indices, [num_samples, 1])
                self.samples = samples

                def construct_out(attention_interaction, element_wise_product, num_samples):
                    self.attention_mul = tf.reshape(
                        tf.matmul(tf.reshape(element_wise_product, shape=[-1, embedding_size]), \
                                  self.weights['attention_W']), shape=[-1, num_samples, model_params['attention_size']])
                    self.attention_mul = self.attention_mul / model_params['temperature']
                    self.attention_exp = tf.exp(
                        tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                                                                                          self.weights['attention_b'])),
                                      2, keep_dims=True))  # None * (M'*(M'-1)) * 1
                    self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keep_dims=True)  # None * 1 * 1
                    self.attention_out = tf.div(self.attention_exp, self.attention_sum,
                                                name="attention_out")  # None * (M'*(M'-1)) * 1

                    predict = self.attention_out * element_wise_product
                    predict = tf.nn.dropout(predict, model_params['dropout'])  # dropout

                    predict = tf.reshape(predict, [-1, num_samples * embedding_size])
                    attention_interaction = tf.reshape(attention_interaction, [num_samples * embedding_size])

                    predict = tf.tensordot(predict, attention_interaction,
                                           axes=1)  # self.AFM * self.attention_f_out#tf.tensordot(self.AFM, self.weights['attention_interaction'], axes=1)

                    self.out = tf.expand_dims(predict, -1) + feature_bias

                    pre = raw_embeddings
                    for sz in deep_layers:
                        pre = tf.contrib.layers.fully_connected(inputs=pre, num_outputs=sz, \
                                                                weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                    l2_reg), scope="%d_layer" % sz, reuse=tf.AUTO_REUSE)

                    y = tf.contrib.layers.fully_connected(inputs=pre, num_outputs=1, activation_fn=tf.identity,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                          scope='out_layer', reuse=tf.AUTO_REUSE) + self.out
                    y = tf.squeeze(y, -1)
                    return y

                self.prediction = construct_out(self.attention_interaction, self.element_wise_product,
                                                self.num_interaction)
                self.attention_interaction = tf.gather(self.attention_interaction, top_k.indices)

                self.element_wise_product = tf.gather(self.element_wise_product, top_k.indices, axis=1)


                self.sampled = construct_out(self.attention_interaction, self.element_wise_product, num_samples)
                self.out = tf.cond(self.weights['global_step'] < tf.constant(50000), lambda: self.prediction,
                                   lambda: self.sampled)
                pred = tf.sigmoid(self.out)

                predictions = {"prob": pred}
                export_outputs = {
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                        predictions)}
                # Provide an estimator spec for `ModeKeys.PREDICT`
                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=predictions,
                        export_outputs=export_outputs)

                # ------bulid loss------

                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=labels))
                # + tf.contrib.layers.l2_regularizer(model_params['lamda_attention'])(self.weights['attention_W']) \
                #        +tf.contrib.layers.l2_regularizer(model_params['lamda_factorization'])(self.weights['interaction']) \
                #               + tf.contrib.layers.l2_regularizer(model_params['lamda_factorization'])(self.weights['factor'])

                # Provide an estimator spec for `ModeKeys.EVAL`
                eval_metric_ops = {
                    "auc": tf.metrics.auc(labels, pred)
                }
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=predictions,
                        loss=loss,
                        eval_metric_ops=eval_metric_ops)

                # ------bulid optimizer------
                if model_params["optimizer"] == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                                       epsilon=1e-8)
                elif model_params["optimizer"] == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                elif model_params["optimizer"] == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
                elif model_params["optimizer"] == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
                elif model_params["optimizer"] == 'ftrl':
                    optimizer = tf.train.FtrlOptimizer(learning_rate)
                elif model_params["optimizer"] == "dcsgd":
                    optimizer = tf.contrib.opt.DelayCompensatedGradientDescentOptimizer(learning_rate=learning_rate,
                                                                                        variance_parameter=2.0)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 50.0)
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
            # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            # Provide an estimator spec for `ModeKeys.TRAIN` modes
            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'pred': pred,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(SIFMNetwork, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
