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
"""Performance RNN generation code as a SequenceGenerator interface."""

import collections
import copy
from functools import partial
import math

# internal imports

import tensorflow as tf

from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn.performance_lib import PerformanceEvent
from magenta.models.performance_rnn import performance_model

import magenta.music as mm

# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 3 seconds.
MAX_NOTE_DURATION_SECONDS = 3.0


class PerformanceRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """Performance RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details,
               steps_per_second=performance_lib.DEFAULT_STEPS_PER_SECOND,
               num_velocity_bins=0, max_note_duration=MAX_NOTE_DURATION_SECONDS,
               fill_generate_section=True, checkpoint=None, bundle=None):
    """Creates a PerformanceRnnSequenceGenerator.

    Args:
      model: Instance of PerformanceRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_second: Number of quantized steps per second.
      num_velocity_bins: Number of quantized velocity bins. If 0, don't use
          velocity.
      max_note_duration: The maximum note duration in seconds to allow during
          generation. This model often forgets to release notes; specifying a
          maximum duration can force it to do so.
      fill_generate_section: If True, the model will generate RNN steps until
          the entire generate section has been filled. If False, the model will
          estimate the number of RNN steps needed and then generate that many
          events, even if the generate section isn't completely filled.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PerformanceRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_second = steps_per_second
    self.num_velocity_bins = num_velocity_bins
    self.max_note_duration = max_note_duration
    self.fill_generate_section = fill_generate_section

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise mm.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise mm.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    else:
      primer_sequence = input_sequence
      input_start_step = 0

    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
      raise mm.SequenceGeneratorException(
          'Got GenerateSection request for section that is before or equal to '
          'the end of the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, last_end_time))

    # Quantize the priming sequence.
    quantized_primer_sequence = mm.quantize_note_sequence_absolute(
        primer_sequence, self.steps_per_second)

    extracted_perfs, _ = performance_lib.extract_performances(
        quantized_primer_sequence, start_step=input_start_step,
        num_velocity_bins=self.num_velocity_bins)
    assert len(extracted_perfs) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, self.steps_per_second, quantize_cutoff=1.0)

    if extracted_perfs and extracted_perfs[0]:
      performance = extracted_perfs[0]
    else:
      # If no track could be extracted, create an empty track that starts at the
      # requested generate_start_step.
      performance = performance_lib.Performance(
          steps_per_second=(
              quantized_primer_sequence.quantization_info.steps_per_second),
          start_step=generate_start_step,
          num_velocity_bins=self.num_velocity_bins)

    # Ensure that the track extends up to the step we want to start generating.
    performance.set_length(generate_start_step - performance.start_step)

    # Extract generation arguments from generator options.
    arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    # Inject the priming performance in the output of the generator, if
    # requested. This option starts with no_ so that if it is unspecified (as
    # will be the case when used with the MIDI interface), the default will be
    # to inject the primer.
    if not (generator_options.args[
        'no_inject_primer_during_generation'].bool_value):
      performance_to_inject = copy.deepcopy(performance)

      args['modify_events_callback'] = partial(
          _inject_performance, performance_to_inject)

      # Initialize the state for the modify events callback.
      args['modify_events_callback_initial_state'] = InjectPerformanceState(
          index=0, num_steps_generated=0, num_steps_injected=0,
          generation_velocity_bin=self.num_velocity_bins,
          injection_velocity_bin=self.num_velocity_bins)

      # If we're injecting the primer performance, we don't want to use it as a
      # primer. Instead we'll use an empty performance.
      performance = performance_lib.Performance(
          steps_per_second=(
              quantized_primer_sequence.quantization_info.steps_per_second),
          start_step=generate_start_step,
          num_velocity_bins=self.num_velocity_bins)

    total_steps = performance.num_steps + (
        generate_end_step - generate_start_step)

    if not performance:
      # Primer is empty; let's just start with silence.
      performance.set_length(min(performance_lib.MAX_SHIFT_STEPS, total_steps))

    while performance.num_steps < total_steps:
      # Assume there's around 10 notes per second and 4 RNN steps per note.
      # Can't know for sure until generation is finished because the number of
      # notes per quantized step is variable.
      steps_to_gen = total_steps - performance.num_steps
      rnn_steps_to_gen = 40 * int(math.ceil(
          float(steps_to_gen) / performance_lib.DEFAULT_STEPS_PER_SECOND))
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      performance = self._model.generate_performance(
          len(performance) + rnn_steps_to_gen, performance, **args)

      if not self.fill_generate_section:
        # In the interest of speed just go through this loop once, which may not
        # entirely fill the generate section.
        break

    performance.set_length(total_steps)

    if not generator_options.args[
        'no_inject_primer_during_generation'].bool_value:
      # Specify a base note sequence because the priming sequence was not
      # included in the performance.
      generated_sequence = performance.to_sequence(
          max_note_duration=self.max_note_duration,
          base_note_sequence=copy.deepcopy(primer_sequence))
    else:
      generated_sequence = performance.to_sequence(
          max_note_duration=self.max_note_duration)

    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


# State to maintain while injecting a performance into a generated performance.
# Maintains the index of the next event to inject, the total number of (time)
# steps generated, the total number of (time) steps injected, and the current
# velocity in the performance to inject and generated performance.
class InjectPerformanceState(object):

  def __init__(self, index, num_steps_generated, num_steps_injected,
               generation_velocity_bin, injection_velocity_bin):
    self.index = index
    self.num_steps_generated = num_steps_generated
    self.num_steps_injected = num_steps_injected
    self.generation_velocity_bin = generation_velocity_bin
    self.injection_velocity_bin = injection_velocity_bin

  def __str__(self):
    return ('InjectPerformanceState(index=%d, num_steps_generated=%d, '
            'num_steps_injected=%d, generation_velocity_bin=%d, '
            'injection_velocity_bin=%d)' % (
                self.index, self.num_steps_generated, self.num_steps_injected,
                self.generation_velocity_bin, self.injection_velocity_bin))


def _inject_performance(performance, encoder_decoder, event_sequences, inputs,
                        states):
  """A modify_events_callback method for generate_performance.

  Should be called with functools.partial first, to fill in the performance and
  start_step arguments.

  Will extend the event sequence using events from the performance argument
  TODO: when?
  whenever the event sequence gets to a new step.

  Args:
    performance: The Performance to inject into the event sequence (also a
        Performance).
    encoder_decoder: Supplied by the callback. The current
        EventSequenceEncoderDecoder.
    event_sequences: Supplied by the callback. The current EventSequence.
    inputs: Supplied by the callback. The current list of encoded events.
    states: Supplied by the callback. A list of InjectPerformanceState tuples
        containing the current index into the performance to inject, the number
        of steps generated so far, and the number of steps injected so far (up
        to the event at the current injection index).

    Returns:
      SKDJFLSKDJFLKJ
  """
  assert len(event_sequences) == len(inputs) == len(states)

  for i in range(len(inputs)):
    event_sequence = event_sequences[i]
    input_ = inputs[i]
    state = states[i]

    # Fast-forward past any time shift and velocity events in the performance to
    # inject, updating the injection state.
    while state.index < len(performance) and (
        performance[state.index].event_type == PerformanceEvent.TIME_SHIFT or
        performance[state.index].event_type == PerformanceEvent.VELOCITY):
      if performance[state.index].event_type == PerformanceEvent.TIME_SHIFT:
        state.num_steps_injected += performance[state.index].event_value
      else:
        state.injection_velocity_bin = performance[state.index].event_value
      state.index += 1

    if state.index == len(performance):
      # We have reached the end of the performance to inject. No need to modify
      # anything, but update the generation state for correctness.
      if event_sequence[-1].event_type == PerformanceEvent.TIME_SHIFT:
        state.num_steps_generated += event_sequence[-1].event_value
      elif event_sequence[-1].event_type == PerformanceEvent.VELOCITY:
        state.generation_velocity_bin = event_sequence[-1].event_value
      continue

    if event_sequence[-1].event_type == PerformanceEvent.TIME_SHIFT:
      if (state.num_steps_generated + event_sequence[-1].event_value >
          state.num_steps_injected):
        # The next generated event takes us past the current injection point.
        input_.pop()
        if state.num_steps_injected == state.num_steps_generated:
          # Remove the time shift entirely and inject the next event.
          event_sequence.pop()
          event_sequence.append(performance[state.index])
        else:
          # Truncate the time shift and inject the next event.
          event_sequence[-1].event_value = (
              state.num_steps_injected - state.num_steps_generated)
          input_.extend(encoder_decoder.get_inputs_batch([event_sequence])[0])
          event_sequence.append(performance[state.index])
        input_.extend(encoder_decoder.get_inputs_batch([event_sequence])[0])
        state.num_steps_generated = state.num_steps_injected
        state.index += 1
      else:
        state.num_steps_generated += event_sequence[-1].event_value

    elif state.num_steps_generated == state.num_steps_injected:
      if performance[state.index].event_type == PerformanceEvent.NOTE_OFF:
        # Note-off events come first, so inject regardless of the generated
        # event.
        event_sequence.pop()
        event_sequence.append(performance[state.index])
        input_.pop()
        input_.extend(encoder_decoder.get_inputs_batch([event_sequence])[0])
        state.index += 1
      elif (performance[state.index].event_type == PerformanceEvent.NOTE_ON and
          event_sequence[-1].event_type != PerformanceEvent.NOTE_OFF):
        # Inject the next note-on event, with a preceding velocity event if
        # necessary.
        event_sequence.pop()
        input_.pop()
        if state.injection_velocity_bin != state.generation_velocity_bin:
          event_sequence.append(PerformanceEvent(
              PerformanceEvent.VELOCITY, state.injection_velocity_bin))
          input_.extend(encoder_decoder.get_inputs_batch([event_sequence])[0])
          state.generation_velocity_bin = state.injection_velocity_bin
        event_sequence.append(performance[state.index])
        input_.extend(encoder_decoder.get_inputs_batch([event_sequence])[0])
        state.index += 1

    else:
      # Generation has not yet caught up with the current injection state.
      if event_sequence[-1].event_type == PerformanceEvent.VELOCITY:
        state.generation_velocity_bin = event_sequence[-1].event_value

  return states


def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return PerformanceRnnSequenceGenerator(
        performance_model.PerformanceRnnModel(config), config.details,
        steps_per_second=config.steps_per_second,
        num_velocity_bins=config.num_velocity_bins, fill_generate_section=False,
        **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in performance_model.default_configs.items()}
