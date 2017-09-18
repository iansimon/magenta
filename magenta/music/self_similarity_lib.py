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
"""Functions for working with self-similarity matrices."""

from __future__ import division

# internal imports
import six
from six.moves import range  # pylint: disable=redefined-builtin

from magenta.common import sequence_example_lib
from magenta.music import encoder_decoder

def event_sequence_self_similarity(events, window_width=8):
  n = len(events)
  self_similarity = []

  for i in range(n):
    ss = []
    for j in range(i):
      matches = 0
      total = 0
      for k in range(-window_width, window_width + 1):
        if (i + k >= 0) and (i + k < n) and (j + k >= 0) and (j + k < n):
          if events[i + k] == events[j + k]:
            matches += 1
      ss.append(matches / (2 * window_width + 1))
    self_similarity.append(ss)

  return self_similarity


class SelfSimilarityEncoderDecoder(
    encoder_decoder.ConditionalEventSequenceEncoderDecoder):
  def __init__(self, base_encoder_decoder):
    self._base_encoder_decoder = base_encoder_decoder

  @property
  def input_size(self):
    return (self._base_encoder_decoder.input_size +
            self._base_encoder_decoder.num_classes)

  @property
  def num_classes(self):
    return self._base_encoder_decoder.num_classes

  @property
  def default_event_label(self):
    return self._base_encoder_decoder.default_event_label

  def events_to_input(self, self_similarity, base_events, position):
    # Each element of events is a (self-sim vector, event) tuple.
    similarities = self_similarity[position + 1]
    base_input = self._base_encoder_decoder.events_to_input(
        base_events, position)

    self_similarity_input = [0.0] * self._base_encoder_decoder.num_classes
    total_similarity = 0.0
    for i in range(position):
      idx = self._base_encoder_decoder.events_to_label(base_events, i)
      self_similarity_input[idx] += similarities[i]
      total_similarity += similarities[i]

    if total_similarity > 0.0:
      for idx in range(self._base_encoder_decoder.num_classes):
        self_similarity_input[idx] /= total_similarity

    return base_input + self_similarity_input

  def events_to_label(self, base_events, position):
    return self._base_encoder_decoder.events_to_label(base_events, position)

  def class_index_to_event(self, class_index, base_events):
    return self._base_encoder_decoder.class_index_to_event(
        class_index, base_events)

  def encode(self, self_similarity, base_events):
    if len(self_similarity) != len(base_events):
      raise ValueError('must have the same number of self-similarity and base '
                       'events (%d self-similarities but %d base events)' % (
                           len(self_similarity), len(base_events)))

    inputs = []
    labels = []
    for i in range(len(base_events) - 1):
      inputs.append(self.events_to_input(self_similarity, base_events, i))
      labels.append(self.events_to_label(base_events, i + 1))
    return sequence_example_lib.make_sequence_example(inputs, labels)

  def get_inputs_batch(self, self_similarity_sequences, base_event_sequences,
                       full_length=False):
    if len(self_similarity_sequences) != len(base_event_sequences):
      raise ValueError(
          '%d self-similarity sequences but %d base event sequences' %
          (len(self_similarity_sequences, len(base_event_sequences))))

    inputs_batch = []
    for self_similarity, base_events in zip(
        self_similarity_sequences, base_event_sequences):
      if len(self_similarity) <= len(base_events):
        raise ValueError('self-similarity sequence must be longer than base '
                         'event sequence (%d self-similarities but %d base '
                         'events)' % (len(self_similarity), len(base_events)))
      inputs = []
      if full_length:
        for i in range(len(base_events)):
          inputs.append(self.events_to_input(self_similarity, base_events, i))
      else:
        inputs.append(self.events_to_input(
            self_similarity, base_events, len(base_events) - 1))
      inputs_batch.append(inputs)
    return inputs_batch

  def extend_event_sequences(self, base_event_sequences, softmax):
    return self._base_encoder_decoder.extend_event_sequences(
        base_event_sequences, softmax)

  def evaluate_log_likelihood(self, base_event_sequences, softmax):
    return self._base_encoder_decoder.evaluate_log_likelihood(
        base_event_sequences, softmax)
