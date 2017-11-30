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
"""Classes for converting between performance input and model input/output."""

from __future__ import division

# internal imports

from magenta.models.time_signature_rnn import time_signature_lib
from magenta.models.time_signature_rnn.time_signature_lib import RhythmEvent
from magenta.music import constants
from magenta.music import encoder_decoder

# Value ranges for event types, as (event_type, min_value, max_value) tuples.
EVENT_RANGES = [
    (RhythmEvent.ONSET, 1, time_signature_lib.NUM_VELOCITY_BINS),
    (RhythmEvent.TIME_SHIFT, 1, time_signature_lib.MAX_SHIFT_STEPS),
]


class RhythmOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for rhythm events."""

  @property
  def num_classes(self):
    return sum(max_value - min_value + 1
               for event_type, min_value, max_value in EVENT_RANGES)

  @property
  def default_event(self):
    return RhythmEvent(
        event_type=RhythmEvent.TIME_SHIFT,
        event_value=time_signature_lib.MAX_SHIFT_STEPS)

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
        return RhythmEvent(
            event_type=event_type, event_value=min_value + index - offset)
      offset += max_value - min_value + 1

    raise ValueError('Unknown event index: %s' % index)

  def event_to_num_steps(self, event):
    if event.event_type == RhythmEvent.TIME_SHIFT:
      return event.event_value
    else:
      return 0


VALID_TIME_SIGNATURES = [
  (2, 2), (2, 4), (3, 4), (4, 4), (6, 8)
]

class TimeSignatureOneHotEncoding(encoder_decoder.OneHotEncoding):
  @property
  def num_classes(self):
    return len(VALID_TIME_SIGNATURES)

  @property
  def default_event(self):
    return (4, 4)

  def encode_event(self, event):
    for idx, time_signature in enumerate(VALID_TIME_SIGNATURES):
      if event == time_signature:
        return idx
    raise ValueError('unknown time signature: %s' % str(event))

  def decode_event(self, index):
    return VALID_TIME_SIGNATURES[index]

