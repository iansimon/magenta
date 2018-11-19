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
"""Performance generation from score in Tensor2Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

import apache_beam as beam

from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities as t2t_modalities
from tensor2tensor.utils import registry

import tensorflow as tf

from magenta.models.score2perf import datagen_beam
from magenta.models.score2perf import modalities
from magenta.models.score2perf import music_encoders

from magenta.music import chord_symbols_lib
from magenta.music import sequences_lib

# TODO(iansimon): figure out the best way not to hard-code these constants

NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108

# pylint: disable=line-too-long
MAESTRO_TFRECORD_PATHS = {
    'train': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_train.tfrecord',
    'dev': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_validation.tfrecord',
    'test': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_test.tfrecord'
}
# pylint: enable=line-too-long


class Score2PerfProblem(problem.Problem):
  """Base class for musical score-to-performance problems.

  Data files contain tf.Example protos with encoded performance in 'targets' and
  optional encoded score in 'inputs'.
  """

  @property
  def splits(self):
    """Dictionary of split names and probabilities. Must sum to one."""
    raise NotImplementedError()

  @property
  def min_hop_size_seconds(self):
    """Minimum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def max_hop_size_seconds(self):
    """Maximum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def num_replications(self):
    """Number of times entire input performances will be split."""
    return 1

  @property
  def add_eos_symbol(self):
    """Whether to append EOS to encoded performances."""
    raise NotImplementedError()

  @property
  def absolute_timing(self):
    """Whether or not score should use absolute (vs. tempo-relative) timing."""
    return False

  @property
  def stretch_factors(self):
    """Temporal stretch factors for data augmentation (in datagen)."""
    return [1.0]

  @property
  def transpose_amounts(self):
    """Pitch transposition amounts for data augmentation (in datagen)."""
    return [0]

  def performances_input_collection(self, tmp_dir):
    """Input performances beam transform (or dictionary thereof) for datagen."""
    raise NotImplementedError()

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del task_id

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
          ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
            in_place=True)
      except chord_symbols_lib.ChordSymbolException:
        raise datagen_beam.DataAugmentationException(
            'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationException(
            'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    augment_fns = [
        functools.partial(
            augment_note_sequence,
            stretch_factor=stretch_factor,
            transpose_amount=transpose_amount)
        for stretch_factor, transpose_amount in itertools.product(
            self.stretch_factors, self.transpose_amounts)
    ]

    datagen_beam.generate_examples(
        input_transform=self.performances_input_transform(tmp_dir),
        output_dir=data_dir,
        problem_name=self.dataset_filename(),
        splits=self.splits,
        min_hop_size_seconds=self.min_hop_size_seconds,
        max_hop_size_seconds=self.max_hop_size_seconds,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        num_replications=self.num_replications,
        encode_performance_fn=self.performance_encoder().encode_note_sequence,
        encode_score_fns=dict((name, encoder.encode_note_sequence)
                              for name, encoder in self.score_encoders()),
        augment_fns=augment_fns,
        absolute_timing=self.absolute_timing)

  def hparams(self, defaults, model_hparams):
    perf_encoder = self.get_feature_encoders()['targets']
    defaults.modality = {'targets': t2t_modalities.SymbolModality}
    defaults.vocab_size = {'targets': perf_encoder.vocab_size}
    if self.has_inputs:
      score_encoder = self.get_feature_encoders()['inputs']
      if isinstance(score_encoder.vocab_size, list):
        modality_cls = modalities.SymbolTupleModality
      else:
        modality_cls = t2t_modalities.SymbolModality
      defaults.modality['inputs'] = modality_cls
      defaults.vocab_size['inputs'] = score_encoder.vocab_size

  def performance_encoder(self):
    """Encoder for target performances."""
    return music_encoders.MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=self.add_eos_symbol)

  def score_encoders(self):
    """List of (name, encoder) tuples for input score components."""
    return []

  def feature_encoders(self, data_dir):
    del data_dir
    encoders = {
        'targets': self.performance_encoder()
    }
    score_encoders = self.score_encoders()
    if score_encoders:
      if len(score_encoders) > 1:
        # Create a composite score encoder, only used for inference.
        encoders['inputs'] = music_encoders.CompositeScoreEncoder(
            [encoder for _, encoder in score_encoders])
      else:
        # If only one score component, just use its encoder.
        _, encoders['inputs'] = score_encoders[0]
    return encoders

  def example_reading_spec(self):
    data_fields = {
        'targets': tf.VarLenFeature(tf.int64)
    }
    for name, _ in self.score_encoders():
      data_fields[name] = tf.VarLenFeature(tf.int64)

    # We don't actually "decode" anything here; the encodings are simply read as
    # tensors.
    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    if self.has_inputs:
      # Stack encoded score components depthwise as inputs.
      inputs = []
      for name, _ in self.score_encoders():
        inputs.append(tf.expand_dims(example[name], axis=1))
        del example[name]
      example['inputs'] = tf.stack(inputs, axis=2)

    return super(Score2PerfProblem, self).preprocess_example(
        example, mode, hparams)


class Chords2PerfProblem(Score2PerfProblem):
  """Base class for musical chords-to-performance problems."""

  def score_encoders(self):
    return [('chords', music_encoders.TextChordsEncoder(steps_per_quarter=1))]


class Melody2PerfProblem(Score2PerfProblem):
  """Base class for musical melody-to-performance problems."""

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class AbsoluteMelody2PerfProblem(Score2PerfProblem):
  """Base class for musical (absolute-timed) melody-to-performance problems."""

  @property
  def absolute_timing(self):
    return True

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoderAbsolute(
            steps_per_second=10, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class LeadSheet2PerfProblem(Score2PerfProblem):
  """Base class for musical lead-sheet-to-performance problems."""

  def score_encoders(self):
    return [
        ('chords', music_encoders.TextChordsEncoder(steps_per_quarter=4)),
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


@registry.register_problem('score2perf_maestro_language_30s_aug')
class Score2PerfMaestroLanguage30sAug(Score2PerfProblem):
  """Piano performance language model on the MAESTRO dataset."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return dict(
        (split_name, datagen_beam.ReadNoteSequencesFromTFRecord(tfrecord_path))
        for split_name, tfrecord_path in MAESTRO_TFRECORD_PATHS.items())

  @property
  def splits(self):
    return None

  @property
  def min_hop_size_seconds(self):
    return 30.0

  @property
  def max_hop_size_seconds(self):
    return 30.0

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]
