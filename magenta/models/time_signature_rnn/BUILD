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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "time_signature_rnn",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":time_signature_model",
    ],
)

py_library(
    name = "time_signature_lib",
    srcs = ["time_signature_lib.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:discrete_music_gan
    ],
    deps = [
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "time_signature_lib_test",
    srcs = ["time_signature_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":time_signature_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "time_signature_model",
    srcs = ["time_signature_model.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        # internal models
    ],
    deps = [
        ":time_signature_encoder_decoder",
        ":time_signature_lib",
        "//magenta/models/shared:events_rnn_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "time_signature_sequence_generator",
    srcs = ["time_signature_sequence_generator.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal notebook binary
        "//magenta/interfaces:generator_interfaces",
    ],
    deps = [
        ":time_signature_lib",
        ":time_signature_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "time_signature_rnn_generate",
    srcs = ["time_signature_rnn_generate.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":time_signature_lib",
        ":time_signature_model",
        ":time_signature_sequence_generator",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "time_signature_encoder_decoder",
    srcs = ["time_signature_encoder_decoder.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:discrete_music_gan
    ],
    deps = [
        ":time_signature_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "time_signature_encoder_decoder_test",
    srcs = ["time_signature_encoder_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":time_signature_encoder_decoder",
        ":time_signature_lib",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "time_signature_rnn_create_dataset",
    srcs = ["time_signature_rnn_create_dataset.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # internal model:time_signature_rnn
        "//magenta/tools/pip:__subpackages__",
    ],
    deps = [
        ":time_signature_lib",
        ":time_signature_model",
        "//magenta",
        # tensorflow dep
    ],
)

py_test(
    name = "time_signature_rnn_create_dataset_test",
    srcs = ["time_signature_rnn_create_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":time_signature_lib",
        ":time_signature_model",
        ":time_signature_rnn_create_dataset",
        "//magenta",
        # tensorflow dep
    ],
)

py_binary(
    name = "time_signature_rnn_train",
    srcs = ["time_signature_rnn_train.py"],
    srcs_version = "PY2AND3",
    visibility = ["//magenta/tools/pip:__subpackages__"],
    deps = [
        ":time_signature_model",
        "//magenta/models/shared:events_rnn_graph",
        "//magenta/models/shared:events_rnn_train",
        # tensorflow dep
    ],
)
