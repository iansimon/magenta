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
"""Time Signature RNN model."""

import collections
import functools

# internal imports

import tensorflow as tf
import magenta

from magenta.models.time_signature_rnn import time_signature_encoder_decoder
from magenta.models.time_signature_rnn.time_signature_lib import RhythmEvent
from magenta.models.shared import events_rnn_model


class TimeSignatureRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN performance generation models."""

  def predict_time_signature(
      self, num_steps, rhythm_sequence, temperature=1.0, beam_size=1,
      branch_factor=1, steps_per_iteration=1):
    """Generate a sequence of time signature predictions.

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a Performance object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes tracks more
         random, less than 1.0 makes tracks less random.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.

    Returns:
      The generated Performance object (which begins with the provided primer
      track).
    """
    control_events = None
    control_state = None
    extend_control_events_callback = None

    return self._generate_events(
        num_steps, primer_sequence, temperature, beam_size, branch_factor,
        steps_per_iteration, control_events=control_events,
        control_state=control_state,
        extend_control_events_callback=extend_control_events_callback)


default_configs = {
    'basic_timesig': events_rnn_model.EventSequenceRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='basic_timesig',
            description='Time signature prediction'),
        magenta.music.ConditionalEventSequenceEncoderDecoder(
            magenta.music.OneHotEventSequenceEncoderDecoder(
                time_signature_encoder_decoder.RhythmOneHotEncoding()),
            magenta.music.OneHotEventSequenceEncoderDecoder(
                time_signature_encoder_decoder.TimeSignatureOneHotEncoding())),
        tf.contrib.training.HParams(
            batch_size=256,
            rnn_layer_sizes=[128, 128, 128],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.0003)),
}
