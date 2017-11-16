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
"""Mapping from model/config names to model configuration."""

from magenta.models.drums_rnn import drums_rnn_model
from magenta.models.improv_rnn import improv_rnn_model
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.performance_rnn import performance_model
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_graph
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model
from magenta.models.polyphony_rnn import polyphony_model
from magenta.models.shared import events_rnn_graph

_MODELS = [
    drums_rnn_model,
    improv_rnn_model,
    melody_rnn_model,
    performance_model,
    pianoroll_rnn_nade_model,
    polyphony_model,
]

# TODO(iansimon): Need a cleaner way to get graph from model name or config.
GRAPHS = {
    'drums_rnn': events_rnn_graph,
    'improv_rnn': events_rnn_graph,
    'melody_rnn': events_rnn_graph,
    'performance_rnn': events_rnn_graph,
    'pianoroll_rnn_nade': pianoroll_rnn_nade_graph,
    'polyphony_rnn': events_rnn_graph,
}

# Each model declares its own dictionary of default configs. Here we create a
# dictionary mapping each model name to its dictionary of configs.
CONFIGS = dict((model.__name__.split('.')[-2], model.default_configs)
               for model in _MODELS)
