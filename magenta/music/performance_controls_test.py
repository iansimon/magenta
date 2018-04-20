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
"""Tests for performance controls."""

# internal imports
import tensorflow as tf

from magenta.music import performance_controls
from magenta.music import performance_lib


class PerformanceControlsTest(tf.test.TestCase):

  def testPerformanceNoteDensitySequence(self):
    performance = performance_lib.Performance(steps_per_second=100)

    pe = performance_lib.PerformanceEvent
    perf_events = [
        pe(pe.NOTE_ON, 60),
        pe(pe.NOTE_ON, 64),
        pe(pe.NOTE_ON, 67),
        pe(pe.TIME_SHIFT, 50),
        pe(pe.NOTE_OFF, 60),
        pe(pe.NOTE_OFF, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 67),
        pe(pe.NOTE_ON, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 64)
    ]
    for event in perf_events:
      performance.append(event)

    expected_density_sequence = [
        4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 0.0]

    density_sequence = performance_lib.performance_note_density_sequence(
        performance, window_size_seconds=1.0)

    self.assertEqual(expected_density_sequence, density_sequence)

  def testPerformancePitchHistogramSequence(self):
    performance = performance_lib.Performance(steps_per_second=100)

    pe = performance_lib.PerformanceEvent
    perf_events = [
        pe(pe.NOTE_ON, 60),
        pe(pe.NOTE_ON, 64),
        pe(pe.NOTE_ON, 67),
        pe(pe.TIME_SHIFT, 50),
        pe(pe.NOTE_OFF, 60),
        pe(pe.NOTE_OFF, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 67),
        pe(pe.NOTE_ON, 64),
        pe(pe.TIME_SHIFT, 25),
        pe(pe.NOTE_OFF, 64)
    ]
    for event in perf_events:
      performance.append(event)

    expected_histogram_sequence = [
        [0.25, 0, 0, 0, 0.375, 0, 0, 0.375, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0.375, 0, 0, 0.375, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0.375, 0, 0, 0.375, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0.375, 0, 0, 0.375, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 1.0, 0, 0, 0.0, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 1.0, 0, 0, 0.0, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 1.0, 0, 0, 0.0, 0, 0, 0, 0],
        [1.0 / 12.0] * 12
    ]

    histogram_sequence = performance_lib.performance_pitch_histogram_sequence(
        performance, window_size_seconds=1.0, prior_count=0)

    self.assertEqual(expected_histogram_sequence, histogram_sequence)

