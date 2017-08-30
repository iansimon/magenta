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
"""Tests for midi_lib."""

# internal imports
import tensorflow as tf

from magenta.models.midi_rnn import midi_lib
from magenta.music import sequences_lib
from magenta.music import testing_lib
from magenta.protobuf import music_pb2


class MidiLibTest(tf.test.TestCase):

  def setUp(self):
    self.maxDiff = None

    self.note_sequence = music_pb2.NoteSequence()
    self.note_sequence.tempos.add().qpm = 60.0
    self.note_sequence.ticks_per_quarter = 220

    ts = self.note_sequence.time_signatures.add()
    ts.numerator = 4
    ts.denominator = 4

  def testFromQuantizedNoteSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)],
        program=0)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(60, 100, 0.0, 3.0)],
        program=8)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    midi_seq = list(midi_lib.MidiSequence(quantized_sequence))

    me = midi_lib.MidiEvent
    expected_midi_seq = [
        me(me.PROGRAM, 1),
        me(me.NOTE_ON, 60),
        me(me.NOTE_ON, 64),
        me(me.PROGRAM, 2),
        me(me.NOTE_ON, 60),
        me(me.TIME_SHIFT, 1),
        me(me.PROGRAM, 1),
        me(me.NOTE_ON, 67),
        me(me.TIME_SHIFT, 1),
        me(me.NOTE_OFF, 67),
        me(me.TIME_SHIFT, 1),
        me(me.NOTE_OFF, 64),
        me(me.PROGRAM, 2),
        me(me.NOTE_OFF, 60),
        me(me.TIME_SHIFT, 1),
        me(me.PROGRAM, 1),
        me(me.NOTE_OFF, 60),
    ]
    self.assertEqual(expected_midi_seq, midi_seq)

  def testToSequence(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0,
        [(60, 100, 0.0, 4.0), (64, 100, 0.0, 3.0), (67, 100, 1.0, 2.0)],
        program=0)
    testing_lib.add_track_to_sequence(
        self.note_sequence, 1,
        [(60, 100, 0.0, 3.0)],
        program=8)
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=1)
    midi_seq = midi_lib.MidiSequence(quantized_sequence)
    midi_seq_ns = midi_seq.to_sequence(qpm=60.0)

    # Make comparison easier by sorting.
    midi_seq_ns.notes.sort(key=lambda n: (n.start_time, n.program, n.pitch))
    self.note_sequence.notes.sort(
        key=lambda n: (n.start_time, n.program, n.pitch))

    self.assertEqual(self.note_sequence, midi_seq_ns)

  def testExtractMidiSequences(self):
    testing_lib.add_track_to_sequence(
        self.note_sequence, 0, [(60, 100, 0.0, 4.0)])
    quantized_sequence = sequences_lib.quantize_note_sequence(
        self.note_sequence, steps_per_quarter=24)

    midi_seqs, _ = midi_lib.extract_midi_sequences(quantized_sequence)
    self.assertEqual(1, len(midi_seqs))

    midi_seqs, _ = midi_lib.extract_midi_sequences(
        quantized_sequence, min_events_discard=1, max_events_truncate=10)
    self.assertEqual(1, len(midi_seqs))

    midi_seqs, _ = midi_lib.extract_midi_sequences(
        quantized_sequence, min_events_discard=8, max_events_truncate=10)
    self.assertEqual(0, len(midi_seqs))

    midi_seqs, _ = midi_lib.extract_midi_sequences(
        quantized_sequence, min_events_discard=1, max_events_truncate=3)
    self.assertEqual(1, len(midi_seqs))
    self.assertEqual(3, len(midi_seqs[0]))


if __name__ == '__main__':
  tf.test.main()
