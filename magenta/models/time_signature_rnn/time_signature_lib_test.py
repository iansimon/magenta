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
"""Tests for performance_lib."""

# internal imports
import tensorflow as tf

from magenta.models.time_signature_rnn import time_signature_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class TimeSignatureLibTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None

    self.note_sequence = music_pb2.NoteSequence()
    self.note_sequence.ticks_per_quarter = 220

    ts = self.note_sequence.time_signatures.add()
    ts.numerator = 4
    ts.denominator = 4

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 127, 0.0, 4.0), (64, 127, 0.0, 3.0), (67, 127, 1.0, 2.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
        self.note_sequence, steps_per_second=100)
    rhythm_sequence = list(
        time_signature_lib.RhythmSequence(quantized_sequence))

    pe = time_signature_lib.RhythmEvent
    expected_rhythm_sequence = [
        pe(pe.ONSET, 8),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 4),
    ]
    self.assertEqual(expected_rhythm_sequence, rhythm_sequence)

  def testSetLengthAddSteps(self):
    rhythm_sequence = time_signature_lib.RhythmSequence(steps_per_second=100)

    rhythm_sequence.set_length(50)
    self.assertEqual(50, rhythm_sequence.num_steps)

    pe = time_signature_lib.RhythmEvent
    rhythm_events = [pe(pe.TIME_SHIFT, 50)]
    self.assertEqual(rhythm_events, list(rhythm_sequence))

    rhythm_sequence.set_length(150)
    self.assertEqual(150, rhythm_sequence.num_steps)

    pe = time_signature_lib.RhythmEvent
    rhythm_events = [
        pe(pe.TIME_SHIFT, 100),
        pe(pe.TIME_SHIFT, 50),
    ]
    self.assertEqual(rhythm_events, list(rhythm_sequence))

    rhythm_sequence.set_length(200)
    self.assertEqual(200, rhythm_sequence.num_steps)

    pe = time_signature_lib.RhythmEvent
    rhythm_events = [
        pe(pe.TIME_SHIFT, 100),
        pe(pe.TIME_SHIFT, 100),
    ]
    self.assertEqual(rhythm_events, list(rhythm_sequence))

  def testSetLengthRemoveSteps(self):
    rhythm_sequence = time_signature_lib.RhythmSequence(steps_per_second=100)

    pe = time_signature_lib.RhythmEvent
    rhythm_events = [
        pe(pe.ONSET, 4),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 3),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 4),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 5),
    ]
    for event in rhythm_events:
      rhythm_sequence.append(event)

    rhythm_sequence.set_length(200)
    rhythm_events = [
        pe(pe.ONSET, 4),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 3),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 4),
    ]
    self.assertEqual(rhythm_events, list(rhythm_sequence))

    rhythm_sequence.set_length(50)
    rhythm_events = [
        pe(pe.ONSET, 4),
        pe(pe.TIME_SHIFT, 50),
    ]
    self.assertEqual(rhythm_events, list(rhythm_sequence))

  def testNumSteps(self):
    rhythm_sequence = time_signature_lib.RhythmSequence(steps_per_second=100)

    pe = time_signature_lib.RhythmEvent
    rhythm_events = [
        pe(pe.ONSET, 5),
        pe(pe.TIME_SHIFT, 100),
        pe(pe.ONSET, 3),
    ]
    for event in rhythm_events:
      rhythm_sequence.append(event)

    self.assertEqual(100, rhythm_sequence.num_steps)

  def testExtractRhythmSequences(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0), (60, 100, 0.5, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
        self.note_sequence, steps_per_second=100)

    rhythms, _ = time_signature_lib.extract_rhythm_sequences(quantized_sequence)
    self.assertEqual(1, len(rhythms))

    rhythms, _ = time_signature_lib.extract_rhythm_sequences(
        quantized_sequence, min_events_discard=1, max_events_truncate=10)
    self.assertEqual(1, len(rhythms))

    rhythms, _ = time_signature_lib.extract_rhythm_sequences(
        quantized_sequence, min_events_discard=4, max_events_truncate=10)
    self.assertEqual(0, len(rhythms))

    rhythms, _ = time_signature_lib.extract_rhythm_sequences(
        quantized_sequence, min_events_discard=1, max_events_truncate=2)
    self.assertEqual(1, len(rhythms))
    self.assertEqual(2, len(rhythms[0]))


if __name__ == '__main__':
  tf.test.main()
