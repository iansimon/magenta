# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Classes for computing performance control signals."""

import abc


class PerformanceControlSignal(object):
  """slkdfjlsdkfjlsdkfj"""
  __metaclass__ = abc.ABCMeta

  def __init__(self, name, description):
    """skdlfjsldkfjslkj"""
    self._name = name
    self._description = description

  @property
  def name(self):
    return self._name

  @property
  def description(self):
    return self._description

  @abc.abstractproperty
  def default_value(self):
    """lsdkfjlskjdf"""
    pass

  @abc.abstractmethod
  def extract(self, performance):
    """sldjflsdkjflsdkfj"""
    pass

  @abc.abstractproperty
  def encoder(self):
    """sldkfjsldkfj"""
    pass


class NoteDensityPerformanceControlSignal(PerformanceControlSignal):
  """sldkfjlsdkf"""

  def __init__(self, window_size_seconds, density_bin_ranges):
    """Initialize a NoteDensityPerformanceControlSignal.

    Args:
      window_size_seconds: The size of the window, in seconds, used to compute
          note density (notes per second).
    """
    # TODO: super
    self._window_size_seconds = window_size_seconds
    self._density_bin_ranges = density_bin_ranges

  def extract(performance):
    """Computes note density at every event in a performance.

    Args:
      performance: A Performance object for which to compute a note density
          sequence.

    Returns:
      A list of note densities of the same length as `performance`, with each
      entry equal to the note density in the window starting at the
      corresponding performance event time.
    """
    window_size_steps = int(round(
        window_size_seconds * performance.steps_per_second))

    prev_event_type = None
    prev_density = 0.0

    density_sequence = []

    for i, event in enumerate(performance):
      if (prev_event_type is not None and
          prev_event_type != PerformanceEvent.TIME_SHIFT):
        # The previous event didn't move us forward in time, so the note density
        # here should be the same.
        density_sequence.append(prev_density)
        prev_event_type = event.event_type
        continue

      j = i
      step_offset = 0
      note_count = 0

      # Count the number of note-on events within the window.
      while step_offset < window_size_steps and j < len(performance):
        if performance[j].event_type == PerformanceEvent.NOTE_ON:
          note_count += 1
        elif performance[j].event_type == PerformanceEvent.TIME_SHIFT:
          step_offset += performance[j].event_value
        j += 1

      # If we're near the end of the performance, part of the window will
      # necessarily be empty; we don't include this part of the window when
      # calculating note density.
      actual_window_size_steps = min(step_offset, window_size_steps)
      if actual_window_size_steps > 0:
        density = (
            note_count * performance.steps_per_second / actual_window_size_steps)
      else:
        density = 0.0

      density_sequence.append(density)

      prev_event_type = event.event_type
      prev_density = density

    return density_sequence


class PitchHistogramPerformanceControlSignal(PerformanceControlSignal):
  """sldkfjlsdkjfslkj"""

  def __init__(self, window_size_seconds, prior_count=0.01):
    """Initializes a PitchHistogramPerformanceControlSignal.

    Args:
      window_size_seconds: The size of the window, in seconds, used to compute
          each histogram.
      prior_count: A prior count to smooth the resulting histograms. This value
          will be added to the actual pitch class counts.
    """
    # TODO: super
    self._window_size_seconds = window_size_seconds
    self._prior_count = prior_count

  def extract(performance):
    """Computes local pitch class histogram at every event in a performance.

    Args:
      performance: A Performance object for which to compute a pitch class
          histogram sequence.

    Returns:
      A list of pitch class histograms the same length as `performance`, where
      each pitch class histogram is a length-12 list of float values summing to
      one.
    """
    window_size_steps = int(round(
        window_size_seconds * performance.steps_per_second))

    prev_event_type = None
    prev_histogram = [1.0 / NOTES_PER_OCTAVE] * NOTES_PER_OCTAVE

    base_active_pitches = set()
    histogram_sequence = []

    for i, event in enumerate(performance):
      # Maintain the base set of active pitches.
      if event.event_type == PerformanceEvent.NOTE_ON:
        base_active_pitches.add(event.event_value)
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        base_active_pitches.discard(event.event_value)

      if (prev_event_type is not None and
          prev_event_type != PerformanceEvent.TIME_SHIFT):
        # The previous event didn't move us forward in time, so the histogram
        # here should be the same.
        histogram_sequence.append(prev_histogram)
        prev_event_type = event.event_type
        continue

      j = i
      step_offset = 0

      active_pitches = copy.deepcopy(base_active_pitches)
      histogram = [prior_count] * NOTES_PER_OCTAVE

      # Count the total duration of each pitch class within the window.
      while step_offset < window_size_steps and j < len(performance):
        if performance[j].event_type == PerformanceEvent.NOTE_ON:
          active_pitches.add(performance[j].event_value)
        elif performance[j].event_type == PerformanceEvent.NOTE_OFF:
          active_pitches.discard(performance[j].event_value)
        elif performance[j].event_type == PerformanceEvent.TIME_SHIFT:
          for pitch in active_pitches:
            histogram[pitch % NOTES_PER_OCTAVE] += (
                performance[j].event_value / performance.steps_per_second)
          step_offset += performance[j].event_value
        j += 1

      histogram_sequence.append(histogram)

      prev_event_type = event.event_type
      prev_histogram = histogram

    return histogram_sequence
