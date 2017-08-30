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
"""MIDI RNN generation code as a SequenceGenerator interface."""

from __future__ import division

from functools import partial
import math
import random

# internal imports

import tensorflow as tf

from magenta.models.midi_rnn import midi_lib
from magenta.models.midi_rnn.midi_lib import MidiEvent
from magenta.models.midi_rnn import midi_rnn_model

import magenta.music as mm

# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 3 seconds.
MAX_NOTE_DURATION_SECONDS = 3.0


class MidiRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """MIDI RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details,
               steps_per_quarter=midi_lib.DEFAULT_STEPS_PER_QUARTER,
               max_note_duration=MAX_NOTE_DURATION_SECONDS,
               fill_generate_section=True, checkpoint=None, bundle=None):
    """Creates a MidiRnnSequenceGenerator.

    Args:
      model: Instance of MidiRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_quarter: Number of quantized steps per quarter note.
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
    super(MidiRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_quarter = steps_per_quarter
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

    qpm = (input_sequence.tempos[0].qpm
           if input_sequence and input_sequence.tempos
           else mm.DEFAULT_QUARTERS_PER_MINUTE)
    steps_per_second = mm.steps_per_quarter_to_steps_per_second(
        self.steps_per_quarter, qpm)

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, steps_per_second, quantize_cutoff=0.0)
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
    quantized_primer_sequence = mm.quantize_note_sequence(
        primer_sequence, self.steps_per_quarter)

    extracted_midis, _ = midi_lib.extract_midi_sequences(
        quantized_primer_sequence, start_step=input_start_step)
    assert len(extracted_midis) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, steps_per_second, quantize_cutoff=0.0)
    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, steps_per_second, quantize_cutoff=1.0)

    if extracted_midis and extracted_midis[0]:
      midi_sequence = extracted_midis[0]
    else:
      # If no track could be extracted, create an empty track that starts at the
      # requested generate_start_step.
      midi_sequence = midi_lib.MidiSequence(
          steps_per_quarter=(
              quantized_primer_sequence.quantization_info.steps_per_quarter),
          start_step=generate_start_step)

    # Ensure that the track extends up to the step we want to start generating.
    midi_sequence.set_length(generate_start_step - midi_sequence.start_step)

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

    total_steps = midi_sequence.num_steps + (
        generate_end_step - generate_start_step)

    if not midi_sequence:
      # Primer is empty; let's just switch to a random program.
      program_bin = random.choice(range(midi_lib.NUM_PROGRAM_BINS)) + 1
      midi_sequence.append(MidiEvent(MidiEvent.PROGRAM, program_bin))

    while midi_sequence.num_steps < total_steps:
      # Assume 8 notes per quarter and 4 RNN steps per note. Can't know for
      # sure until generation is finished because the number of notes per
      # quantized step is variable.
      steps_to_gen = total_steps - midi_sequence.num_steps
      rnn_steps_to_gen = int(math.ceil(
          32.0 * steps_to_gen / self.steps_per_quarter))
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      midi_sequence = self._model.generate_midi(
          len(midi_sequence) + rnn_steps_to_gen, midi_sequence, **args)

      if not self.fill_generate_section:
        # In the interest of speed just go through this loop once, which may not
        # entirely fill the generate section.
        break

    midi_sequence.set_length(total_steps)

    generated_sequence = midi_sequence.to_sequence(
        max_note_duration=self.max_note_duration)

    assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return generated_sequence


def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return MidiRnnSequenceGenerator(
        midi_rnn_model.MidiRnnModel(config), config.details,
        steps_per_quarter=config.steps_per_quarter,
        fill_generate_section=False,
        **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in midi_rnn_model.default_configs.items()}
