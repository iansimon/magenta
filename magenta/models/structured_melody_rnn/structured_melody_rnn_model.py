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
"""Structured Melody RNN model."""

import collections
import copy
import functools

# internal imports
import numpy as np
import tensorflow as tf

import magenta
from magenta.common import beam_search
from magenta.common import state_util
from magenta.models.shared import events_rnn_model
from magenta.models.structured_melody_rnn import structured_melody_rnn_graph
import magenta.music as mm

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0


# Model state when generating event sequences with self-similarity.
ModelState = collections.namedtuple(
    'ModelState',
    ['inputs', 'input_buffer', 'labels', 'past_encodings', 'rnn_state'])


class StructuredMelodyRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN melody generation models with self-similarity."""

  def _build_graph_for_generation(self):
    return structured_melody_rnn_graph.build_graph('generate', self._config)

  def _generate_step_for_batch(self, event_sequences, inputs, input_buffer,
                               labels, past_encodings, initial_state,
                               temperature):
    """Extends a batch of event sequences by a single step each.

    This method modifies the event sequences in place.

    Args:
      event_sequences: A list of event sequences, each of which is a Python
          list-like object. The list of event sequences should have length equal
          to `self._batch_size()`. These are extended by this method.
      inputs: A Python list of model inputs, with length equal to
          `self._batch_size()`.
      input_buffer: ???
      labels: ???
      past_encodings: A numpy array of past encodings, with shape
          `[batch_size, num_labels, encoding_size]`.
      initial_state: A numpy array containing the initial RNN state, where
          `initial_state.shape[0]` is equal to `self._batch_size()`.
      temperature: The softmax temperature.

    Returns:
      labels: A list of
      encodings: A numpy array of encodings, with shape
          `[batch_size, num_inputs, encoding_size]`.
      final_state: The final RNN state, a numpy array the same size as
          `initial_state`.
      loglik: The log-likelihood of the chosen softmax value for each event
          sequence, a 1-D numpy array of length
          `self._batch_size()`. If `inputs` is a full-length inputs batch, the
          log-likelihood of each entire sequence up to and including the
          generated step will be computed and returned.
    """
    assert len(event_sequences) == self._batch_size()

    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_input_buffer = self._session.graph.get_collection('input_buffer')[0]
    graph_labels = self._session.graph.get_collection('labels')[0]
    graph_past_encodings = self._session.graph.get_collection(
        'past_encodings')[0]
    graph_encodings = self._session.graph.get_collection('encodings')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')
    graph_final_state = self._session.graph.get_collection('final_state')
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')

    feed_dict = {graph_inputs: inputs, graph_input_buffer: input_buffer,
                 graph_labels: labels, graph_past_encodings: past_encodings,
                 tuple(graph_initial_state): initial_state}
    # For backwards compatibility, we only try to pass temperature if the
    # placeholder exists in the graph.
    if graph_temperature:
      feed_dict[graph_temperature[0]] = temperature
    encodings, final_state, softmax = self._session.run(
        [graph_encodings, graph_final_state, graph_softmax], feed_dict)

    if softmax.shape[1] > 1:
      # The inputs batch is longer than a single step, so we also want to
      # compute the log-likelihood of the event sequences up until the step
      # we're generating.
      loglik = self._config.encoder_decoder.evaluate_log_likelihood(
          event_sequences, softmax[:, :-1, :])
    else:
      loglik = np.zeros(len(event_sequences))

    indices = self._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax)
    p = softmax[range(len(event_sequences)), -1, indices]

    for i in range(len(event_sequences)):
      labels[i].append(indices[i])

    return labels, encodings, final_state, loglik + np.log(p)

  def _generate_step(self, event_sequences, model_states, logliks, temperature):
    """Extends a list of event sequences by a single step each.

    This method modifies the event sequences in place. It also returns the
    modified event sequences and updated model states and log-likelihoods.

    Args:
      event_sequences: A list of event sequence objects, which are extended by
          this method.
      model_states: A list of model states, each of which contains model inputs,
          input buffers, past encodings, and initial RNN states.
      logliks: A list containing the current log-likelihood for each event
          sequence.
      temperature: The softmax temperature.

    Returns:
      event_sequences: A list of extended event sequences. These are modified in
          place but also returned.
      final_states: A list of resulting model states, containing model inputs
          for the next step, input buffers, encodings, and RNN states for each
          event sequence.
      logliks: A list containing the updated log-likelihood for each event
          sequence.
    """
    # Split the sequences to extend into batches matching the model batch size.
    batch_size = self._batch_size()
    num_seqs = len(event_sequences)
    num_batches = int(np.ceil(num_seqs / float(batch_size)))

    # Unpack the model states.
    inputs = [model_state.inputs for model_state in model_states]
    input_buffers = [model_state.input_buffer for model_state in model_states]
    labels = [model_state.labels for model_state in model_states]
    past_encodings = [model_state.past_encodings
                      for model_state in model_states]
    initial_states = [model_state.rnn_state for model_state in model_states]

    encodings = []
    final_states = []
    logliks = np.array(logliks, dtype=np.float32)

    # Add padding to fill the final batch.
    pad_amt = -len(event_sequences) % batch_size
    padded_event_sequences = event_sequences + [
        copy.deepcopy(event_sequences[-1]) for _ in range(pad_amt)]
    padded_inputs = inputs + [inputs[-1]] * pad_amt
    padded_input_buffers = input_buffers + [input_buffers[-1]] * pad_amt
    padded_labels = labels + [copy.deepcopy(labels[-1]) for _ in range(pad_amt)]
    padded_past_encodings = past_encodings + [past_encodings[-1]] * pad_amt
    padded_initial_states = initial_states + [initial_states[-1]] * pad_amt

    for b in range(num_batches):
      i, j = b * batch_size, (b + 1) * batch_size
      pad_amt = max(0, j - num_seqs)
      # Generate a single step for one batch of event sequences.
      batch_labels, batch_encodings, batch_final_state, batch_loglik = (
          self._generate_step_for_batch(
              padded_event_sequences[i:j],
              padded_inputs[i:j],
              padded_input_buffers[i:j],
              padded_labels[i:j],
              padded_past_encodings[i:j],
              state_util.batch(padded_initial_states[i:j], batch_size),
              temperature))
      encodings += [np.concatenate([past_encoding, encoding], axis=0)
                    for past_encoding, encoding
                    in zip(padded_past_encodings[i:j], batch_encodings)]
      final_states += state_util.unbatch(
          batch_final_state, batch_size)[:j - i - pad_amt]
      logliks[i:j - pad_amt] += batch_loglik[:j - i - pad_amt]

    # Construct input buffers for next step.
    for prev_inputs, input_buffer in zip(inputs, input_buffers):
      input_buffer = input_buffer[1:] + [prev_inputs[-1]]

    # Construct inputs for next step.
    next_inputs = self._config.encoder_decoder.get_inputs_batch(
        event_sequences)

    model_states = [
        ModelState(inputs=inputs, input_buffer=input_buffer, labels=past_labels,
                   past_encodings=past_encodings, rnn_state=final_state)
        for inputs, input_buffer, past_labels, past_encodings, final_state
        in zip(next_inputs, input_buffers, labels, encodings, final_states)]

    return event_sequences, model_states, logliks

  def _generate_events(self, num_steps, primer_events, temperature=1.0,
                       beam_size=1, branch_factor=1, steps_per_iteration=1):
    """Generate an event sequence from a primer sequence.

    Args:
      num_steps: The integer length in steps of the final event sequence, after
          generation. Includes the primer.
      primer_events: The primer event sequence, a Python list-like object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes events more
         random, less than 1.0 makes events less random.
      beam_size: An integer, beam size to use when generating event sequences
          via beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.

    Returns:
      The generated event sequence (which begins with the provided primer).

    Raises:
      EventSequenceRnnModelException: If the primer sequence has zero length or
          is not shorter than num_steps.
    """
    if not primer_events:
      raise EventSequenceRnnModelException(
          'primer sequence must have non-zero length')
    if len(primer_events) >= num_steps:
      raise EventSequenceRnnModelException(
          'primer sequence must be shorter than `num_steps`')

    if len(primer_events) >= num_steps:
      # Sequence is already long enough, no need to generate.
      return primer_events

    event_sequences = [copy.deepcopy(primer_events)]

    # Construct inputs for first step after primer, along with initial input
    # buffer (all zeroes) and past encodings (empty).
    inputs = self._config.encoder_decoder.get_inputs_batch(
        event_sequences, full_length=True)
    input_buffer = [[0.0] * self._config.encoder_decoder.input_size
                    for _ in range(self._config.hparams.window_size - 1)]
    labels = [self._config.encoder_decoder.events_to_label(primer_events, i)
              for i in range(1, len(primer_events))]
    past_encodings = np.zeros([0, self._config.hparams.encoding_size],
                              dtype=np.float32)

    graph_initial_state = self._session.graph.get_collection('initial_state')
    initial_states = state_util.unbatch(self._session.run(graph_initial_state))

    # Beam search will maintain a state for each sequence consisting of the next
    # inputs to feed the model, and the current RNN state. We start out with the
    # initial full inputs batch and the zero state.
    initial_state = ModelState(
        inputs=inputs[0], input_buffer=input_buffer, labels=labels,
        past_encodings=past_encodings, rnn_state=initial_states[0])

    events, _, loglik = beam_search(
        initial_sequence=event_sequences[0],
        initial_state=initial_state,
        generate_step_fn=functools.partial(
            self._generate_step,
            temperature=temperature),
        num_steps=num_steps - len(primer_events),
        beam_size=beam_size,
        branch_factor=branch_factor,
        steps_per_iteration=steps_per_iteration)

    tf.logging.info('Beam search yields sequence with log-likelihood: %f ',
                    loglik)

    return events

  def generate_melody(self, num_steps, primer_melody, temperature=1.0,
                      beam_size=1, branch_factor=1, steps_per_iteration=1):
    """Generate a melody from a primer melody.

    Args:
      num_steps: The integer length in steps of the final melody, after
          generation. Includes the primer.
      primer_melody: The primer melody, a Melody object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes melodies more
         random, less than 1.0 makes melodies less random.
      beam_size: An integer, beam size to use when generating melodies via beam
          search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of melody steps to take per beam
          search iteration.

    Returns:
      The generated Melody object (which begins with the provided primer
          melody).
    """
    melody = copy.deepcopy(primer_melody)

    transpose_amount = melody.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)

    melody = self._generate_events(num_steps, melody, temperature, beam_size,
                                   branch_factor, steps_per_iteration)

    melody.transpose(-transpose_amount)

    return melody

  def melody_log_likelihood(self, melody):
    """Evaluate the log likelihood of a melody under the model.

    Args:
      melody: The Melody object for which to evaluate the log likelihood.

    Returns:
      The log likelihood of `melody` under this model.
    """
    melody_copy = copy.deepcopy(melody)

    melody_copy.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)

    return self._evaluate_log_likelihood([melody_copy])[0]


class StructuredMelodyRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  """Stores a configuration for a StructuredMelodyRnn.

  You can change `min_note` and `max_note` to increase/decrease the melody
  range. Since melodies are transposed into this range to be run through
  the model and then transposed back into their original range after the
  melodies have been extended, the location of the range is somewhat
  arbitrary, but the size of the range determines the possible size of the
  generated melodies range. `transpose_to_key` should be set to the key
  that if melodies were transposed into that key, they would best sit
  between `min_note` and `max_note` with having as few notes outside that
  range.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use.
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies will be transposed into, or
        None if it should not be transposed.
  """

  def __init__(self, details, encoder_decoder, hparams,
               min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE,
               transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):
    super(StructuredMelodyRnnConfig, self).__init__(details, encoder_decoder, hparams)

    if min_note < mm.MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > mm.MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < mm.NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or transpose_to_key > mm.NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key


# Default configurations.
default_configs = {
    'self_similarity_rnn': StructuredMelodyRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='self_similarity_rnn',
            description='Melody RNN with learned self-similarity attention.'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
        tf.contrib.training.HParams(
            batch_size=128,
            window_size=16,
            encoding_size=64,
            rnn_layer_sizes=[128, 128],
            dropout_keep_prob=0.5,
            clip_norm=5,
            learning_rate=0.001)),
}
