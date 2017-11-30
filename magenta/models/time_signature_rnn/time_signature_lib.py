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
"""Utility functions for working with performances + time signature."""

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

MIN_VELOCITY = 1
MAX_VELOCITY = 255
NUM_VELOCITY_BINS = 8

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_STEPS_PER_SECOND = 100
MAX_SHIFT_STEPS = 100


class RhythmEvent(object):
  """Class for storing rhythm events in a performance."""

  # Start of a new note.
  ONSET = 1
  # Shift time forward.
  TIME_SHIFT = 2

  def __init__(self, event_type, event_value):
    if not RhythmEvent.ONSET <= event_type <= RhythmEvent.TIME_SHIFT:
      raise ValueError('Invalid event type: %s' % event_type)

    if (event_type == RhythmEvent.ONSET):
      if not 1 <= event_value <= NUM_VELOCITY_BINS:
        raise ValueError('Invalid onset velocity value: %s' % event_value)
    elif event_type == RhythmEvent.TIME_SHIFT:
      if not 1 <= event_value <= MAX_SHIFT_STEPS:
        raise ValueError('Invalid time shift value: %s' % event_value)

    self.event_type = event_type
    self.event_value = event_value

  def __repr__(self):
    return 'RhythmEvent(%r, %r)' % (self.event_type, self.event_value)

  def __eq__(self, other):
    if not isinstance(other, RhythmEvent):
      return False
    return (self.event_type == other.event_type and
            self.event_value == other.event_value)


class RhythmSequence(events_lib.EventSequence):
  """Stores a rhythm sequence as a stream of rhythm events.

  Events are RhythmEvent objects that encode event type and value.
  """

  def __init__(self, quantized_sequence=None, steps_per_second=None,
               start_step=0):
    """Construct a RhythmSequence.

    Either quantized_sequence or steps_per_second should be supplied.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      steps_per_second: Number of quantized time steps per second.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.

    Raises:
      ValueError: If `num_velocity_bins` is larger than the number of MIDI
          velocity values.
    """
    if (quantized_sequence, steps_per_second).count(None) != 1:
      raise ValueError(
          'Must specify exactly one of quantized_sequence or steps_per_second')

    if quantized_sequence:
      sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)
      self._events = self._from_quantized_sequence(
          quantized_sequence, start_step)
      self._steps_per_second = (
          quantized_sequence.quantization_info.steps_per_second)
      self._time_signature = (
          quantized_sequence.time_signatures[0].numerator,
          quantized_sequence.time_signatures[0].denominator)
    else:
      self._events = []
      self._steps_per_second = steps_per_second
      self._time_signature = None

    self._start_step = start_step

  @property
  def start_step(self):
    return self._start_step

  @property
  def steps_per_second(self):
    return self._steps_per_second

  @property
  def time_signature(self):
    return self._time_signature

  def _append_steps(self, num_steps):
    """Adds steps to the end of the sequence."""
    if (self._events and
        self._events[-1].event_type == RhythmEvent.TIME_SHIFT and
        self._events[-1].event_value < MAX_SHIFT_STEPS):
      # Last event is already non-maximal time shift. Increase its duration.
      added_steps = min(num_steps,
                        MAX_SHIFT_STEPS - self._events[-1].event_value)
      self._events[-1] = RhythmEvent(
          RhythmEvent.TIME_SHIFT,
          self._events[-1].event_value + added_steps)
      num_steps -= added_steps

    while num_steps >= MAX_SHIFT_STEPS:
      self._events.append(
          RhythmEvent(event_type=RhythmEvent.TIME_SHIFT,
                      event_value=MAX_SHIFT_STEPS))
      num_steps -= MAX_SHIFT_STEPS

    if num_steps > 0:
      self._events.append(
          RhythmEvent(event_type=RhythmEvent.TIME_SHIFT,
                      event_value=num_steps))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    while self._events and steps_trimmed < num_steps:
      if self._events[-1].event_type == RhythmEvent.TIME_SHIFT:
        if steps_trimmed + self._events[-1].event_value > num_steps:
          self._events[-1] = RhythmEvent(
              event_type=RhythmEvent.TIME_SHIFT,
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
      event: The rhythm event to append to the end.

    Raises:
      ValueError: If `event` is not a valid rhythm event.
    """
    if not isinstance(event, RhythmEvent):
      raise ValueError('Invalid rhythm event: %s' % event)
    self._events.append(event)

  def truncate(self, num_events):
    """Truncates this RhythmSequence to the specified number of events.

    Args:
      num_events: The number of events to which this rhythm sequence will be
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
      if event.event_type == RhythmEvent.ONSET:
        strs.append('(%s, ON)' % event.event_value)
      elif event.event_type == RhythmEvent.TIME_SHIFT:
        strs.append('(%s, SHIFT)' % event.event_value)
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
      if event.event_type == RhythmEvent.TIME_SHIFT:
        steps += event.event_value
    return steps

  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step=0):
    """Populate self with events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if note.quantized_start_step >= start_step]

    # Sort all note start events.
    onset_velocities = {}
    for note in notes:
      if note.quantized_start_step not in onset_velocities:
        onset_velocities[note.quantized_start_step] = 0
      onset_velocities[note.quantized_start_step] += note.velocity
    onsets = sorted(onset_velocities.items())

    velocity_bin_size = int(math.ceil(
        (MAX_VELOCITY - MIN_VELOCITY + 1) / NUM_VELOCITY_BINS))
    velocity_to_bin = (
        lambda v: (min(v, MAX_VELOCITY) - MIN_VELOCITY) // velocity_bin_size + 1)

    current_step = start_step
    rhythm_events = []

    for step, velocity in onsets:
      # Shift time forward from the current step to this event.
      while step > current_step + MAX_SHIFT_STEPS:
        # We need to move further than the maximum shift size.
        rhythm_events.append(
            RhythmEvent(event_type=RhythmEvent.TIME_SHIFT,
                        event_value=MAX_SHIFT_STEPS))
        current_step += MAX_SHIFT_STEPS
      if step > current_step:
        rhythm_events.append(
            RhythmEvent(event_type=RhythmEvent.TIME_SHIFT,
                        event_value=int(step - current_step)))
      current_step = step

      # Add a rhythm event for this onset.
      rhythm_events.append(
          RhythmEvent(event_type=RhythmEvent.ONSET,
                      event_value=velocity_to_bin(velocity)))

    return rhythm_events


def extract_rhythm_sequences(
    quantized_sequence, start_step=0, min_events_discard=None,
    max_events_truncate=None):
  """Extracts a rhythm sequence from the given quantized NoteSequence.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a rhythm sequence at this time step.
    min_events_discard: Minimum length of tracks in events. Shorter tracks are
        discarded.
    max_events_truncate: Maximum length of tracks in events. Longer tracks are
        truncated.

  Returns:
    rhythm_sequences: A python list of RhythmSequence instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)

  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['rhythms_discarded_too_short',
                 'rhythms_truncated',
                 'rhythms_discarded_no_time_signature',
                 'rhythms_discarded_multiple_time_signatures']])

  steps_per_second = quantized_sequence.quantization_info.steps_per_second

  # Create a histogram measuring lengths (in seconds not steps).
  stats['rhythm_lengths_in_seconds'] = statistics.Histogram(
      'rhythm_lengths_in_seconds',
      [5, 10, 20, 30, 40, 60, 120])

  rhythm_sequences = []

  if not quantized_sequence.time_signatures:
    stats['rhythms_discarded_no_time_signature'].increment()
    return rhythm_sequences, stats.values()
  if len(quantized_sequence.time_signatures) > 1:
    stats['rhythms_discarded_multiple_time_signatures'].increment()
    return rhythm_sequences, stats.values()

  # Translate the quantized sequence into a RhythmSequence.
  rhythm_sequence = RhythmSequence(quantized_sequence, start_step=start_step)

  if (max_events_truncate is not None and
      len(rhythm_sequence) > max_events_truncate):
    rhythm_sequence.truncate(max_events_truncate)
    stats['rhythms_truncated'].increment()

  if (min_events_discard is not None and
      len(rhythm_sequence) < min_events_discard):
    stats['rhythms_discarded_too_short'].increment()
  else:
    rhythm_sequences.append(rhythm_sequence)
    stats['rhythm_lengths_in_seconds'].increment(
        rhythm_sequence.num_steps // steps_per_second)

  return rhythm_sequences, stats.values()
