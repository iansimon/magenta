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
"""Utility functions for working with polyphonic MIDI."""

from __future__ import division

import collections
import copy
import math

# internal imports
import tensorflow as tf

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH

MAX_MIDI_PROGRAM = 127
MIN_MIDI_PROGRAM = 0
NUM_PROGRAM_BINS = 16
PROGRAMS_PER_BIN = 8

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_STEPS_PER_QUARTER = 24
MAX_SHIFT_STEPS = 96


class MidiEvent(object):
  """Class for storing events in a polyphonic MIDI."""

  # Start of a new note.
  NOTE_ON = 1
  # End of a note.
  NOTE_OFF = 2
  # Drum hit.
  DRUM = 3
  # Shift time forward.
  TIME_SHIFT = 4
  # Change current program.
  PROGRAM = 5

  def __init__(self, event_type, event_value):
    if not MidiEvent.NOTE_ON <= event_type <= MidiEvent.PROGRAM:
      raise ValueError('Invalid event type: %s' % event_type)

    if (event_type == MidiEvent.NOTE_ON or
        event_type == MidiEvent.NOTE_OFF or
        event_type == MidiEvent.DRUM):
      if not MIN_MIDI_PITCH <= event_value <= MAX_MIDI_PITCH:
        raise ValueError('Invalid pitch value: %s' % event_value)
    elif event_type == MidiEvent.TIME_SHIFT:
      if not 1 <= event_value <= MAX_SHIFT_STEPS:
        raise ValueError('Invalid time shift value: %s' % event_value)
    elif event_type == MidiEvent.PROGRAM:
      if not 1 <= event_value <= NUM_PROGRAM_BINS:
        raise ValueError('Invalid program value: %s' % event_value)

    self.event_type = event_type
    self.event_value = event_value

  def __repr__(self):
    return 'MidiEvent(%r, %r)' % (self.event_type, self.event_value)

  def __eq__(self, other):
    if not isinstance(other, MidiEvent):
      return False
    return (self.event_type == other.event_type and
            self.event_value == other.event_value)


class MidiSequence(events_lib.EventSequence):
  """Stores a polyphonic sequence as a stream of MIDI events.

  Events are MidiEvent objects that encode event type and value.
  """

  def __init__(self, quantized_sequence=None, steps_per_quarter=None,
               start_step=0):
    """Construct a MidiSequence.

    Either quantized_sequence or steps_per_second should be supplied.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      steps_per_quarter: Number of quantized time steps per quarter note.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.
    """
    if (quantized_sequence, steps_per_quarter).count(None) != 1:
      raise ValueError(
          'Must specify exactly one of quantized_sequence or steps_per_quarter')

    if quantized_sequence:
      sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
      self._events = self._from_quantized_sequence(
          quantized_sequence, start_step)
      self._steps_per_quarter = (
          quantized_sequence.quantization_info.steps_per_quarter)
    else:
      self._events = []
      self._steps_per_quarter = steps_per_quarter

    self._start_step = start_step

  @property
  def start_step(self):
    return self._start_step

  @property
  def steps_per_second(self):
    return self._steps_per_second

  def _append_steps(self, num_steps):
    """Adds steps to the end of the sequence."""
    if (self._events and
        self._events[-1].event_type == MidiEvent.TIME_SHIFT and
        self._events[-1].event_value < MAX_SHIFT_STEPS):
      # Last event is already non-maximal time shift. Increase its duration.
      added_steps = min(num_steps,
                        MAX_SHIFT_STEPS - self._events[-1].event_value)
      self._events[-1] = MidiEvent(
          MidiEvent.TIME_SHIFT,
          self._events[-1].event_value + added_steps)
      num_steps -= added_steps

    while num_steps >= MAX_SHIFT_STEPS:
      self._events.append(
          MidiEvent(event_type=MidiEvent.TIME_SHIFT,
                    event_value=MAX_SHIFT_STEPS))
      num_steps -= MAX_SHIFT_STEPS

    if num_steps > 0:
      self._events.append(
          MidiEvent(event_type=MidiEvent.TIME_SHIFT,
                    event_value=num_steps))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    while self._events and steps_trimmed < num_steps:
      if self._events[-1].event_type == MidiEvent.TIME_SHIFT:
        if steps_trimmed + self._events[-1].event_value > num_steps:
          self._events[-1] = MidiEvent(
              event_type=MidiEvent.TIME_SHIFT,
              event_value=(self._events[-1].event_value -
                           num_steps + steps_trimmed))
          steps_trimmed = num_steps
        else:
          steps_trimmed += self._events[-1].event_value
          self._events.pop()
      else:
        self._events.pop()

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with time shifts to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if from_left:
      raise NotImplementedError('from_left is not supported')

    if self.num_steps < steps:
      self._append_steps(steps - self.num_steps)
    elif self.num_steps > steps:
      self._trim_steps(self.num_steps - steps)

    assert self.num_steps == steps

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The MIDI event to append to the end.

    Raises:
      ValueError: If `event` is not a valid MIDI event.
    """
    if not isinstance(event, MidiEvent):
      raise ValueError('Invalid MIDI event: %s' % event)
    self._events.append(event)

  def truncate(self, num_events):
    """Truncates this MidiSequence to the specified number of events.

    Args:
      num_events: The number of events to which this MIDI sequence will be
          truncated.
    """
    self._events = self._events[:num_events]

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  def __str__(self):
    strs = []
    for event in self:
      if event.event_type == MidiEvent.NOTE_ON:
        strs.append('(%s, ON)' % event.event_value)
      elif event.event_type == MidiEvent.NOTE_OFF:
        strs.append('(%s, OFF)' % event.event_value)
      elif event.event_type == MidiEvent.DRUM:
        strs.append('(%s, DRUM)' % event.event_value)
      elif event.event_type == MidiEvent.TIME_SHIFT:
        strs.append('(%s, SHIFT)' % event.event_value)
      elif event.event_type == MidiEvent.PROGRAM:
        strs.append('(%s, PROGRAM)' % event.event_value)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)
    return '\n'.join(strs)

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    steps = 0
    for event in self:
      if event.event_type == MidiEvent.TIME_SHIFT:
        steps += event.event_value
    return steps

  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step=0,
                               num_velocity_bins=0):
    """Populate self with events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    PROGRAM changes the current program value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if note.quantized_start_step >= start_step]
    sorted_notes = sorted(
        notes, key=lambda note: (note.start_time, note.program, note.pitch))

    # Sort all note start and end events.
    onsets = [(note.quantized_start_step, idx, False)
              for idx, note in enumerate(sorted_notes)]
    offsets = [(note.quantized_end_step, idx, True)
               for idx, note in enumerate(sorted_notes)]
    note_events = sorted(onsets + offsets)

    program_to_bin = (
          lambda p: (p - MIN_MIDI_PROGRAM) // PROGRAMS_PER_BIN + 1)

    current_step = start_step
    current_program_bin = 0
    midi_events = []

    for step, idx, is_offset in note_events:
      if is_offset and sorted_notes[idx].is_drum:
        # Ignore note-off for drums.
        continue

      if step > current_step:
        # Shift time forward from the current step to this event.
        while step > current_step + MAX_SHIFT_STEPS:
          # We need to move further than the maximum shift size.
          midi_events.append(
              MidiEvent(event_type=MidiEvent.TIME_SHIFT,
                        event_value=MAX_SHIFT_STEPS))
          current_step += MAX_SHIFT_STEPS
        midi_events.append(
            MidiEvent(event_type=MidiEvent.TIME_SHIFT,
                      event_value=int(step - current_step)))
        current_step = step

      # If this isn't a drum, and this note's program is different from the
      # current program, change the current program.
      if not sorted_notes[idx].is_drum:
        program_bin = program_to_bin(sorted_notes[idx].program)
        if program_bin != current_program_bin:
          current_program_bin = program_bin
          midi_events.append(
              MidiEvent(event_type=MidiEvent.PROGRAM,
                        event_value=current_program_bin))

      # Add a MIDI event for this note on/off.
      if sorted_notes[idx].is_drum:
        event_type = MidiEvent.DRUM
      elif is_offset:
        event_type = MidiEvent.NOTE_OFF
      else:
        event_type = MidiEvent.NOTE_ON
      midi_events.append(
          MidiEvent(event_type=event_type,
                    event_value=sorted_notes[idx].pitch))

    return midi_events

  def to_sequence(self,
                  velocity=100,
                  qpm=120.0,
                  max_note_duration=None):
    """Converts the MidiSequence to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
      qpm: Tempo in quarter notes per minute.
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.

    Raises:
      ValueError: if an unknown event is encountered.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 60.0 / qpm / self._steps_per_quarter

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = STANDARD_PPQ

    ts = sequence.time_signatures.add()
    ts.numerator = 4
    ts.denominator = 4

    sequence_start_time = self.start_step * seconds_per_step

    step = 0
    program = 0

    # Map pitch to list because one pitch may be active multiple times.
    pitch_start_steps = collections.defaultdict(list)
    for i, event in enumerate(self):
      if event.event_type == MidiEvent.NOTE_ON:
        pitch_start_steps[(program, event.event_value)].append(step)
      elif event.event_type == MidiEvent.NOTE_OFF:
        if not pitch_start_steps[(program, event.event_value)]:
          tf.logging.debug(
              'Ignoring NOTE_OFF at position %d with no previous NOTE_ON' % i)
        else:
          # Create a note for the pitch that is now ending.
          pitch_start_step = pitch_start_steps[(program, event.event_value)][0]
          pitch_start_steps[(program, event.event_value)] = (
              pitch_start_steps[(program, event.event_value)][1:])
          if step == pitch_start_step:
            tf.logging.debug(
                'Ignoring note with zero duration at step %d' % step)
            continue
          note = sequence.notes.add()
          note.start_time = (pitch_start_step * seconds_per_step +
                             sequence_start_time)
          note.end_time = step * seconds_per_step + sequence_start_time
          if (max_note_duration and
              note.end_time - note.start_time > max_note_duration):
            note.end_time = note.start_time + max_note_duration
          note.pitch = event.event_value
          note.velocity = velocity
          note.instrument = program // PROGRAMS_PER_BIN
          if note.instrument >= 9:
            note.instrument += 1
          note.program = program
          if note.end_time > sequence.total_time:
            sequence.total_time = note.end_time
      elif event.event_type == MidiEvent.DRUM:
        note = sequence.notes.add()
        note.start_time = step * seconds_per_step + sequence_start_time
        note.end_time = (step + 1) * seconds_per_step + sequence_start_time
        note.pitch = event.event_value
        note.velocity = velocity
        note.instrument = 9
        note.is_drum = True
        if note.end_time > sequence.total_time:
          sequence.total_time = note.end_time
      elif event.event_type == MidiEvent.TIME_SHIFT:
        step += event.event_value
      elif event.event_type == MidiEvent.PROGRAM:
        program = (
            MIN_MIDI_PROGRAM + (event.event_value - 1) * PROGRAMS_PER_BIN)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)

    # There could be remaining pitches that were never ended. End them now
    # and create notes.
    for program, pitch in pitch_start_steps:
      for pitch_start_step in pitch_start_steps[(program, pitch)]:
        if step == pitch_start_step:
          tf.logging.debug(
              'Ignoring note with zero duration at step %d' % step)
          continue
        note = sequence.notes.add()
        note.start_time = (pitch_start_step * seconds_per_step +
                           sequence_start_time)
        note.end_time = step * seconds_per_step + sequence_start_time
        if (max_note_duration and
            note.end_time - note.start_time > max_note_duration):
          note.end_time = note.start_time + max_note_duration
        note.pitch = pitch
        note.velocity = velocity
        note.instrument = program // PROGRAMS_PER_BIN
        if note.instrument >= 9:
          note.instrument += 1
        note.program = program
        if note.end_time > sequence.total_time:
          sequence.total_time = note.end_time

    return sequence


def extract_midi_sequences(
    quantized_sequence, start_step=0, min_events_discard=None,
    max_events_truncate=None):
  """Extracts a MidiSequence from the given quantized NoteSequence.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step.
    min_events_discard: Minimum length of tracks in events. Shorter tracks are
        discarded.
    max_events_truncate: Maximum length of tracks in events. Longer tracks are
        truncated.

  Returns:
    midi_sequences: A python list of MidiSequence instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)

  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['sequences_discarded_not_4/4',
                 'sequences_discarded_too_short',
                 'sequences_truncated']])

  steps_per_second = quantized_sequence.quantization_info.steps_per_second

  # Create a histogram measuring lengths (in bars not steps).
  stats['sequence_lengths_in_steps'] = statistics.Histogram(
      'sequence_lengths_in_steps', [256, 512, 768])

  midi_sequences = []

  if (len(quantized_sequence.time_signatures) != 1 or
      quantized_sequence.time_signatures[0].numerator != 4 or
      quantized_sequence.time_signatures[0].denominator != 4):
    stats['sequences_discarded_not_4/4'].increment()
    return midi_sequences, stats.values()

  # Translate the quantized sequence into a MidiSequence.
  midi_sequence = MidiSequence(quantized_sequence, start_step=start_step)

  if (max_events_truncate is not None and
      len(midi_sequence) > max_events_truncate):
    midi_sequence.truncate(max_events_truncate)
    stats['sequences_truncated'].increment()

  if min_events_discard is not None and len(midi_sequence) < min_events_discard:
    stats['sequences_discarded_too_short'].increment()
  else:
    midi_sequences.append(midi_sequence)
    stats['sequence_lengths_in_steps'].increment(midi_sequence.num_steps)

  return midi_sequences, stats.values()
