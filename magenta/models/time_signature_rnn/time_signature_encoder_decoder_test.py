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
"""Tests for performance_encoder_decoder."""

# internal imports

import tensorflow as tf

from magenta.models.time_signature_rnn import time_signature_encoder_decoder
from magenta.models.time_signature_rnn.time_signature_lib import RhythmEvent


class RhythmOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = time_signature_encoder_decoder.RhythmOneHotEncoding()

  def testEncodeDecode(self):
    expected_pairs = [
        (RhythmEvent(
            event_type=RhythmEvent.ONSET, event_value=4), 3),
        (RhythmEvent(
            event_type=RhythmEvent.ONSET, event_value=1), 0),
        (RhythmEvent(
            event_type=RhythmEvent.ONSET, event_value=8), 7),
        (RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=10), 17),
        (RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=1), 8),
        (RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=100), 107),
    ]

    for expected_event, expected_index in expected_pairs:
      index = self.enc.encode_event(expected_event)
      self.assertEqual(expected_index, index)
      event = self.enc.decode_event(expected_index)
      self.assertEqual(expected_event, event)

  def testEventToNumSteps(self):
    self.assertEqual(0, self.enc.event_to_num_steps(
        RhythmEvent(event_type=RhythmEvent.ONSET, event_value=5)))
    self.assertEqual(1, self.enc.event_to_num_steps(
        RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=1)))
    self.assertEqual(45, self.enc.event_to_num_steps(
        RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=45)))
    self.assertEqual(100, self.enc.event_to_num_steps(
        RhythmEvent(
            event_type=RhythmEvent.TIME_SHIFT, event_value=100)))


class TimeSignatureOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = time_signature_encoder_decoder.TimeSignatureOneHotEncoding()

  def testEncodeDecode(self):
    self.assertEqual(0, self.enc.encode_event((2, 2)))
    self.assertEqual(1, self.enc.encode_event((2, 4)))
    self.assertEqual(2, self.enc.encode_event((3, 4)))
    self.assertEqual(3, self.enc.encode_event((4, 4)))
    self.assertEqual(4, self.enc.encode_event((6, 8)))

    self.assertEqual((2, 2), self.enc.decode_event(0))
    self.assertEqual((2, 4), self.enc.decode_event(1))
    self.assertEqual((3, 4), self.enc.decode_event(2))
    self.assertEqual((4, 4), self.enc.decode_event(3))
    self.assertEqual((6, 8), self.enc.decode_event(4))


if __name__ == '__main__':
  tf.test.main()
