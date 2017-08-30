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
"""Tests for midi_encoder_decoder."""

# internal imports

import tensorflow as tf

from magenta.models.midi_rnn import midi_encoder_decoder
from magenta.models.midi_rnn.midi_lib import MidiEvent


class MidiOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = midi_encoder_decoder.MidiOneHotEncoding()

  def testEncodeDecode(self):
    expected_pairs = [
        (MidiEvent(
            event_type=MidiEvent.NOTE_ON, event_value=60), 60),
        (MidiEvent(
            event_type=MidiEvent.NOTE_ON, event_value=0), 0),
        (MidiEvent(
            event_type=MidiEvent.NOTE_ON, event_value=127), 127),
        (MidiEvent(
            event_type=MidiEvent.NOTE_OFF, event_value=72), 200),
        (MidiEvent(
            event_type=MidiEvent.NOTE_OFF, event_value=0), 128),
        (MidiEvent(
            event_type=MidiEvent.NOTE_OFF, event_value=127), 255),
        (MidiEvent(
            event_type=MidiEvent.DRUM, event_value=60), 316),
        (MidiEvent(
            event_type=MidiEvent.DRUM, event_value=0), 256),
        (MidiEvent(
            event_type=MidiEvent.DRUM, event_value=127), 383),
        (MidiEvent(
            event_type=MidiEvent.TIME_SHIFT, event_value=10), 393),
        (MidiEvent(
            event_type=MidiEvent.TIME_SHIFT, event_value=1), 384),
        (MidiEvent(
            event_type=MidiEvent.TIME_SHIFT, event_value=96), 479),
        (MidiEvent(
            event_type=MidiEvent.PROGRAM, event_value=5), 484),
        (MidiEvent(
            event_type=MidiEvent.PROGRAM, event_value=1), 480),
        (MidiEvent(
            event_type=MidiEvent.PROGRAM, event_value=16), 495)
    ]

    for expected_event, expected_index in expected_pairs:
      index = self.enc.encode_event(expected_event)
      self.assertEqual(expected_index, index)
      event = self.enc.decode_event(expected_index)
      self.assertEqual(expected_event, event)


if __name__ == '__main__':
  tf.test.main()
