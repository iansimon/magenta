# Copyright 2016 Google Inc. All Rights Reserved.
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

# Description: An RNN model for generating melodies.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "structured_melody_rnn",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":structured_melody_rnn_model",
    ],
)

py_library(
    name = "structured_melody_rnn_model",
    srcs = ["structured_melody_rnn_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal models
        # internal notebook binary
    ],
    deps = [
        ":structured_melody_rnn_graph",
        "//magenta",
        "//magenta/models/shared:events_rnn_model",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "structured_melody_rnn_sequence_generator",
    srcs = ["structured_melody_rnn_sequence_generator.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/interfaces:generator_interfaces",
    ],
    deps = [
        ":structured_melody_rnn_model",
        "//magenta",
    ],
)

py_binary(
    name = "structured_melody_rnn_create_dataset",
    srcs = ["structured_melody_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:structured_melody_rnn
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":structured_melody_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "structured_melody_rnn_create_dataset_test",
    srcs = ["structured_melody_rnn_create_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":structured_melody_rnn_create_dataset",
        ":structured_melody_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "structured_melody_rnn_graph",
    srcs = ["structured_melody_rnn_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "structured_melody_rnn_graph_test",
    srcs = ["structured_melody_rnn_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":structured_melody_rnn_graph",
        "//magenta",
        "//magenta/models/shared:events_rnn_model",
        # tensorflow dep
    ],
)

py_binary(
    name = "structured_melody_rnn_train",
    srcs = ["structured_melody_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":structured_melody_rnn_graph",
        ":structured_melody_rnn_model",
        "//magenta/models/shared:events_rnn_train",
        # tensorflow dep
    ],
)

py_binary(
    name = "structured_melody_rnn_generate",
    srcs = ["structured_melody_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":structured_melody_rnn_model",
        ":structured_melody_rnn_sequence_generator",
        "//magenta",
        # tensorflow dep
    ],
)
