# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides function to build an structured melody RNN model's graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import six
import tensorflow as tf
import magenta

from tensorflow.python.util import nest as tf_nest


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells)

  return cell


def encode_rnn_outputs(rnn_outputs, batch_size, input_size, encoding_size):
  """Encodes a sequence of RNN outputs using shared weights.

  Args:
    rnn_states: A tensor of input RNN states with shape
        `[batch_size, num_steps, num_units]`.
    batch_size: The number of sequences per batch.
    input_size: The number of "input" RNN outputs.
    encoding_size: The size of the final encoding, used to compute self-
        similarities.

  Returns:
    A tensor with shape `[batch_size, num_steps, encoding_size]` containing the
    encoded RNN output at each step.
  """
  rnn_outputs_flat = tf.expand_dims(rnn_outputs, -1)

  # This isn't really a 2D convolution, but a fully-connected layer operating on
  # RNN outputs.
  encodings = tf.contrib.layers.conv2d(
      rnn_outputs_flat, encoding_size, [1, input_size], padding='VALID',
      activation_fn=tf.nn.relu)

  return tf.squeeze(encodings, axis=2)


def similarity_weighted_attention(targets, self_similarity):
  """Computes similarity-weighted softmax attention over past inputs.

  For each step, computes an attention-weighted sum of the input at prior steps,
  where attention is determined by self-similarity.

  Args:
    targets: A tensor of input sequences with shape
        `[batch_size, num_target_steps, target_size]`.
    self_similarity: A tensor of input self-similarities based on encoded
        windows, with shape `[batch_size, num_input_steps, num_target_steps]`.

  Returns:
    A tensor with shape `[batch_size, num_input_steps, target_size]` containing
    the similarity-weighted attention over targets for each step.
  """
  num_input_steps = tf.shape(self_similarity)[1]
  num_target_steps = tf.shape(self_similarity)[2]

  steps = tf.range(num_target_steps - num_input_steps + 1, num_target_steps + 1)
  transposed_self_similarity = tf.transpose(self_similarity, [1, 0, 2])

  def similarity_to_attention(enumerated_similarity):
    step, sim = enumerated_similarity
    return tf.concat(
        [tf.nn.softmax(sim[:, :step]), tf.zeros_like(sim[:, step:])], axis=-1)

  transposed_attention = tf.map_fn(
      similarity_to_attention, (steps, transposed_self_similarity),
      dtype=tf.float32)
  attention = tf.transpose(transposed_attention, [1, 0, 2])

  return tf.matmul(attention, targets)


def self_similarity_layer(inputs, lengths, past_targets, past_encodings,
                          rnn_layer_sizes, dropout_keep_prob, batch_size,
                          encoding_size):
  """SKDJFLKSJDLFKSJLK"""
  # Run an RNN over the inputs.
  cell = make_rnn_cell(rnn_layer_sizes, dropout_keep_prob)
  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, final_state = tf.nn.dynamic_rnn(
      cell, inputs, sequence_length=lengths, initial_state=initial_state,
      swap_memory=True)

  # Encode the RNN outputs.
  encodings = encode_rnn_outputs(
      outputs, batch_size, rnn_layer_sizes[-1], encoding_size)

  # Compute similarity between current encodings and all past and current
  # encodings except the most recent.
  target_encodings = tf.concat([past_encodings, encodings[:, :-1, :]], axis=1)
  self_similarity = tf.matmul(encodings, target_encodings, transpose_b=True)

  # Compute and append similarity-weighted attention on all targets.
  targets = tf.concat([past_targets, inputs[:, 1:, :]], axis=1)
  attention_outputs = similarity_weighted_attention(targets, self_similarity)
  combined_outputs = tf.concat([outputs, attention_outputs], axis=2)

  return (
      combined_outputs, initial_state, final_state, encodings, self_similarity)


def build_graph(mode, config, sequence_example_file_paths=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  if len(hparams.rnn_layer_sizes) != len(hparams.encoding_sizes):
    raise ValueError(
        'inconsistent number of RNN and encoding layers: %d vs %d' % (
            len(hparams.rnn_layer_sizes), len(hparams.encoding_sizes)))

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  num_layers = len(hparams.rnn_layer_sizes)

  layer_input_sizes = [input_size]
  for layer in range(num_layers - 1):
    layer_input_sizes.append(
        layer_input_sizes[-1] + hparams.rnn_layer_sizes[layer][-1])

  with tf.Graph().as_default() as graph:
    inputs, labels, lengths = None, None, None

    past_targets = []
    past_encodings = []

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size,
          shuffle=mode == 'train')
      # When training, we get the entire input sequence with no history.
      for layer in range(num_layers):
        past_targets.append(
            tf.zeros([hparams.batch_size, 0, layer_input_sizes[layer]]))
        past_encodings.append(
            tf.zeros([hparams.batch_size, 0, hparams.encoding_sizes[layer]]))

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])
      # When generating, we need to attend over all past targets and encodings
      # at each level.
      for layer in range(num_layers):
        past_targets.append(
            tf.placeholder(
                tf.float32, [hparams.batch_size, None,
                             layer_input_sizes[layer]]))
        past_encodings.append(
            tf.placeholder(
                tf.float32, [hparams.batch_size, None,
                             hparams.encoding_sizes[layer]]))

    initial_state = []
    final_state = []
    encodings = []
    self_similarity = []

    for layer in range(num_layers):
      with tf.variable_scope('similarity_layer_%d' % (layer + 1)):
        outputs, layer_initial_state, layer_final_state, layer_encodings, layer_self_similarity = (
            self_similarity_layer(
                inputs, lengths, past_targets[layer], past_encodings[layer],
                rnn_layer_sizes=hparams.rnn_layer_sizes[layer],
                dropout_keep_prob=(
                    1.0 if mode == 'generate' else hparams.dropout_keep_prob),
                batch_size=hparams.batch_size,
                encoding_size=hparams.encoding_sizes[layer]))
        inputs = outputs
        initial_state.append(layer_initial_state)
        final_state.append(layer_final_state)
        encodings.append(layer_encodings)
        self_similarity.append(layer_self_similarity)

    outputs_flat = magenta.common.flatten_maybe_padded_sequences(
        outputs, lengths)
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      labels_flat = magenta.common.flatten_maybe_padded_sequences(
          labels, lengths)

      softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_flat, logits=logits_flat)

      predictions_flat = tf.argmax(logits_flat, axis=1)
      correct_predictions = tf.to_float(
          tf.equal(labels_flat, predictions_flat))
      event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
      no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))

      if mode == 'train':
        loss = tf.reduce_mean(softmax_cross_entropy)
        perplexity = tf.exp(loss)
        accuracy = tf.reduce_mean(correct_predictions)
        event_accuracy = (
            tf.reduce_sum(correct_predictions * event_positions) /
            tf.reduce_sum(event_positions))
        no_event_accuracy = (
            tf.reduce_sum(correct_predictions * no_event_positions) /
            tf.reduce_sum(no_event_positions))

        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparams.clip_norm)
        tf.add_to_collection('train_op', train_op)

        vars_to_summarize = {
            'loss': loss,
            'metrics/perplexity': perplexity,
            'metrics/accuracy': accuracy,
            'metrics/event_accuracy': event_accuracy,
            'metrics/no_event_accuracy': no_event_accuracy,
        }

        for layer in range(num_layers):
          tf.summary.image('self_similarity_%d' % (layer + 1),
                           tf.expand_dims(self_similarity[layer], -1),
                           max_outputs=1)

      elif mode == 'eval':
        vars_to_summarize, update_ops = tf.contrib.metrics.aggregate_metric_map(
            {
                'loss': tf.metrics.mean(softmax_cross_entropy),
                'metrics/accuracy': tf.metrics.accuracy(
                    labels_flat, predictions_flat),
                'metrics/per_class_accuracy':
                    tf.metrics.mean_per_class_accuracy(
                        labels_flat, predictions_flat, num_classes),
                'metrics/event_accuracy': tf.metrics.recall(
                    event_positions, correct_predictions),
                'metrics/no_event_accuracy': tf.metrics.recall(
                    no_event_positions, correct_predictions),
            })

        for updates_op in update_ops.values():
          tf.add_to_collection('eval_ops', updates_op)

        # Perplexity is just exp(loss) and doesn't need its own update op.
        vars_to_summarize['metrics/perplexity'] = tf.exp(
            vars_to_summarize['loss'])

      for var_name, var_value in six.iteritems(vars_to_summarize):
        tf.summary.scalar(var_name, var_value)
        tf.add_to_collection(var_name, var_value)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)

      for layer in range(num_layers):
        tf.add_to_collection('past_targets_%d' % layer, past_targets[layer])
        tf.add_to_collection('past_encodings_%d' % layer, past_encodings[layer])
        tf.add_to_collection('encodings_%d' % layer, encodings[layer])

        # Flatten state tuples for metagraph compatibility.
        for state in tf_nest.flatten(initial_state[layer]):
          tf.add_to_collection('initial_state_%d' % layer, state)
        for state in tf_nest.flatten(final_state[layer]):
          tf.add_to_collection('final_state_%d' % layer, state)

  return graph
