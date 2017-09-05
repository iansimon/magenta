# Copyright 2017 Google Inc. All Rights Reserved.
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
"""MIDI RNN model."""

import collections
import functools

# internal imports

import tensorflow as tf
import magenta

from magenta.models.midi_rnn import midi_encoder_decoder
from magenta.models.midi_rnn import midi_lib
from magenta.models.midi_rnn.midi_lib import MidiEvent
from magenta.models.shared import events_rnn_model


class MidiRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN MIDI sequence generation models."""

  def generate_midi(
      self, num_steps, primer_sequence, temperature=1.0, beam_size=1,
      branch_factor=1, steps_per_iteration=1):
    """Generate a MIDI track from a primer track.

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a MidiSequence object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes tracks more
         random, less than 1.0 makes tracks less random.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.

    Returns:
      The generated MidiSequence object (which begins with the provided primer
      track).
    """
    if self._config.use_control_state:
      control_events = []
      control_state = midi_lib.MidiControlState(idx=-1, step=0,
          active_notes=collections.defaultdict(int), current_program=0)
      extend_control_events_callback = midi_lib.extend_control_events
    else:
      control_events = None
      control_state = None
      extend_control_events_callback = None

    return self._generate_events(
        num_steps, primer_sequence, temperature, beam_size, branch_factor,
        steps_per_iteration, control_events=control_events,
        control_state=control_state,
        extend_control_events_callback=extend_control_events_callback)

  def midi_log_likelihood(self, sequence):
    """Evaluate the log likelihood of a MIDI sequence.

    Args:
      sequence: The MidiSequence object for which to evaluate the log
          likelihood.

    Returns:
      The log likelihood of `sequence` under this model.
    """
    return self._evaluate_log_likelihood([sequence])[0]


class MidiRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  """Stores a configuration for a Midi RNN.

  Attributes:
    use_control_state: Whether or not to condition on control state (notes
        active, current program, position in bar).
  """

  def __init__(self, details, encoder_decoder, hparams, steps_per_quarter,
               use_control_state=False):
    super(MidiRnnConfig, self).__init__(
        details, encoder_decoder, hparams, steps_per_quarter=steps_per_quarter)
    self.use_control_state = use_control_state


default_configs = {
    'midi': MidiRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='midi',
            description='MIDI RNN'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            midi_encoder_decoder.MidiOneHotEncoding()),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        steps_per_quarter=24),

    'midi_with_state': MidiRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='midi_with_state',
            description='MIDI RNN with control state'),
        magenta.music.ConditionalEventSequenceEncoderDecoder(
            midi_encoder_decoder.MidiControlSequenceEncoderDecoder(
                steps_per_bar=96),
            magenta.music.OneHotEventSequenceEncoderDecoder(
                midi_encoder_decoder.MidiOneHotEncoding())),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        steps_per_quarter=24,
        use_control_state=True),
}
