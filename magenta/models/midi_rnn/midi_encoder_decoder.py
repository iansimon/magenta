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
"""Classes for converting between MIDI input and model input/output."""

from __future__ import division

# internal imports

from magenta.models.midi_rnn import midi_lib
from magenta.models.midi_rnn.midi_lib import MidiEvent
from magenta.music import constants
from magenta.music import encoder_decoder


# Value ranges for event types, as (event_type, min_value, max_value) tuples.
EVENT_RANGES = [
    (MidiEvent.NOTE_ON, midi_lib.MIN_MIDI_PITCH, midi_lib.MAX_MIDI_PITCH),
    (MidiEvent.NOTE_OFF, midi_lib.MIN_MIDI_PITCH, midi_lib.MAX_MIDI_PITCH),
    (MidiEvent.DRUM, midi_lib.MIN_MIDI_PITCH, midi_lib.MAX_MIDI_PITCH),
    (MidiEvent.TIME_SHIFT, 1, midi_lib.MAX_SHIFT_STEPS),
    (MidiEvent.PROGRAM, 1, midi_lib.NUM_PROGRAM_BINS),
]


class MidiOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for MIDI events."""

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for event_type, min_value, max_value in EVENT_RANGES)

  @property
  def default_event(self):
    return MidiEvent(
        event_type=MidiEvent.TIME_SHIFT,
        event_value=midi_lib.MAX_SHIFT_STEPS)

  def encode_event(self, event):
    offset = 0
    for event_type, min_value, max_value in EVENT_RANGES:
      if event.event_type == event_type:
        return offset + event.event_value - min_value
      offset += max_value - min_value + 1

    raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    offset = 0
    for event_type, min_value, max_value in EVENT_RANGES:
      if offset <= index <= offset + max_value - min_value:
        return MidiEvent(
            event_type=event_type, event_value=min_value + index - offset)
      offset += max_value - min_value + 1

    raise ValueError('Unknown event index: %s' % index)


class MidiControlSequenceEncoderDecoder(
    encoder_decoder.EventSequenceEncoderDecoder):
  """An encoder/decoder for MIDI control sequences."""

  def __init__(self, steps_per_bar):
    self._steps_per_bar = steps_per_bar

  @property
  def input_size(self):
    return (
        midi_lib.NUM_PROGRAM_BINS * (midi_lib.NUM_MIDI_PITCHES + 1) +
        self._steps_per_bar)

  @property
  def num_classes(self):
    raise NotImplementedError

  @property
  def default_event_label(self):
    raise NotImplementedError

  def events_to_input(self, events, position):
    return events[position]

  def events_to_label(self, events, position):
    raise NotImplementedError

  def class_index_to_event(self, class_index, events):
    raise NotImplementedError
