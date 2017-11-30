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
them to TensorFlow's SequenceExample protos for input to the performance RNN
models. It will apply data augmentation, stretching and transposing each
NoteSequence within a limited range.
"""

import os

# internal imports

import tensorflow as tf

from magenta.models.time_signature_rnn import time_signature_lib
from magenta.models.time_signature_rnn import time_signature_model

from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_string('config', 'performance', 'The config to use')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class EncoderPipeline(pipeline.Pipeline):
  """A Pipeline that converts performances to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: A TimeSignatureRnnConfig that specifies the encoder/decoder and
          note density conditioning behavior.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=time_signature_lib.RhythmSequence,
        output_type=tf.train.SequenceExample,
        name=name)
    self._encoder_decoder = config.encoder_decoder

  def transform(self, rhythm_sequence):
    time_signature_sequence = (
        [rhythm_sequence.time_signature] * len(rhythm_sequence))
    try:
      encoded = [self._encoder_decoder.encode(rhythm_sequence,
                                              time_signature_sequence)]
      return encoded
    except ValueError:
      self._set_stats([statistics.Counter('encoding_error')])
      return []


class RhythmSequenceExtractor(pipeline.Pipeline):
  """Extracts polyphonic tracks from a quantized NoteSequence."""

  def __init__(self, min_events, max_events, name=None):
    super(RhythmSequenceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=time_signature_lib.RhythmSequence,
        name=name)
    self._min_events = min_events
    self._max_events = max_events

  def transform(self, quantized_sequence):
    rhythm_sequences, stats = time_signature_lib.extract_rhythm_sequences(
        quantized_sequence,
        min_events_discard=self._min_events,
        max_events_truncate=self._max_events)
    self._set_stats(stats)
    return rhythm_sequences


def get_pipeline(config, min_events, max_events, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A TimeSignatureRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
  stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_rhythm_sequences', 'training_rhythm_sequences'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors if mode == 'training' else [1.0],
        name='StretchPipeline_' + mode)
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)
    rhythm_extractor = RhythmSequenceExtractor(
        min_events=min_events, max_events=max_events,
        name='RhythmSequenceExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[stretch_pipeline] = partitioner[mode + '_rhythm_sequences']
    dag[time_change_splitter] = stretch_pipeline
    dag[splitter] = time_change_splitter
    dag[quantizer] = splitter
    dag[rhythm_extractor] = quantizer
    dag[encoder_pipeline] = rhythm_extractor
    dag[dag_pipeline.DagOutput(mode + '_rhythm_sequences')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  pipeline_instance = get_pipeline(
      min_events=32,
      max_events=512,
      eval_ratio=FLAGS.eval_ratio,
      config=time_signature_model.default_configs[FLAGS.config])

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
