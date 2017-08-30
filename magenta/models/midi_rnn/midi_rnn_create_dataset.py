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
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract polyphonic tracks from NoteSequence protos and save
them to TensorFlow's SequenceExample protos for input to the MIDI RNN models.
It will apply data augmentation, transposing each NoteSequence within a limited
range.
"""

import os

# internal imports

import tensorflow as tf

from magenta.models.midi_rnn import midi_lib
from magenta.models.midi_rnn import midi_rnn_model

from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_string('config', 'midi', 'The config to use')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class EncoderPipeline(pipeline.Pipeline):
  """A Pipeline that converts MIDI sequences to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: A MidiRnnConfig that specifies the encoder/decoder.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=midi_lib.MidiSequence,
        output_type=tf.train.SequenceExample,
        name=name)
    self._encoder_decoder = config.encoder_decoder

  def transform(self, midi_sequence):
    encoded = self._encoder_decoder.encode(midi_sequence)
    return [encoded]


class MidiSequenceExtractor(pipeline.Pipeline):
  """Extracts polyphonic tracks from a quantized NoteSequence."""

  def __init__(self, min_events, max_events, name=None):
    super(MidiSequenceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=midi_lib.MidiSequence,
        name=name)
    self._min_events = min_events
    self._max_events = max_events

  def transform(self, quantized_sequence):
    midi_seqs, stats = midi_lib.extract_midi_sequences(
        quantized_sequence,
        min_events_discard=self._min_events,
        max_events_truncate=self._max_events)
    self._set_stats(stats)
    return midi_seqs


def get_pipeline(config, min_events, max_events, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A MidiRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  # Transpose no more than a major third.
  transposition_range = range(-3, 4)

  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_midi_sequences', 'training_midi_sequences'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, name='TranspositionPipeline_' + mode)
    midi_extractor = MidiSequenceExtractor(
        min_events=min_events, max_events=max_events,
        name='MidiSequenceExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + '_midi_sequences']
    dag[quantizer] = time_change_splitter
    dag[transposition_pipeline] = quantizer
    dag[midi_extractor] = transposition_pipeline
    dag[encoder_pipeline] = midi_extractor
    dag[dag_pipeline.DagOutput(mode + '_midi_sequences')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  pipeline_instance = get_pipeline(
      min_events=96,
      max_events=1024,
      eval_ratio=FLAGS.eval_ratio,
      config=midi_rnn_model.default_configs[FLAGS.config])

  input_dir = os.path.expanduser(FLAGS.input)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  pipeline.run_pipeline_serial(
      pipeline_instance,
      pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type),
      output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
