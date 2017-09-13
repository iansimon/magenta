


def event_sequence_self_similarity(events, window_width=5):
  n = len(events)
  ss = []

  for i in range(n):
    for j in range(i):
      matches = 0
      total = 0
      for k in range(-window_width, window_width + 1):
        if (i + k >= 0) and (i + k < n) and (j + k >= 0) and (j + k < n):
          if events[i + k] == events[j + k]:
            matches += 1
          total += 1
      sim = matches / total
      LSKDJFLSKDJFLKSDJFLSKDJ



class SelfSimilarityEncoderDecoder(EventSequenceEncoderDecoder):
  def __init__(self, base_encoder):
    self._base_encoder_decoder = base_encoder_decoder

  @property
  def input_size(self):
    return (self._base_encoder_decoder.input_size +
            self._base_encoder_decoder.num_classes)

  @property
  def num_classes(self):
    return self._base_encoder_decoder.num_classes

  @property
  def default_event_label(self):
    return self._base_encoder_decoder.num_classes

  def events_to_input(self, events, position):
    # Each element of events is a (self-sim vector, event) tuple.
    similarities, _ = events[position]
    base_events = [event for _, event in events]
    base_input = self._base_encoder_decoder.events_to_input(
        base_events, position)

    self_similarity_input = [0.0] * self._base_encoder_decoder.num_classes
    total_similarity = 0.0
    for i in range(position):
      idx = self._base_encoder_decoder.events_to_label(base_events, i)
      self_similarity_input[idx] += similarities[i]
      total_similarity += similarities[i]

    return base_input + self_similarity_input

  def events_to_label(self, events, position):
    base_events = [event for _, event in events]
    return self._base_encoder_decoder.events_to_label(base_events, position)

  def class_index_to_event(self, class_index, events):
    base_events = [event for _, event in events]
    return self._base_encoder_decoder.class_index_to_event(
        class_index, base_events)
