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

licenses(["notice"])  # Apache 2.0

py_library(
    name = "datagen_beam",
    srcs = ["datagen_beam.py"],
    deps = [
        "//magenta",
        # apache_beam dep
        # numpy dep
        # tensorflow dep
        # tensor2tensor/data_generators:generator_utils dep
    ],
)

py_test(
    name = "datagen_beam_test",
    srcs = ["datagen_beam_test.py"],
    data = ["//magenta/testdata"],
    deps = [
        ":datagen_beam",
        ":music_encoders",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "modalities",
    srcs = ["modalities.py"],
    deps = [
        # tensorflow dep
        # tensor2tensor/layers:common_layers dep
        # tensor2tensor/utils:modality dep
    ],
)

py_test(
    name = "modalities_test",
    srcs = ["modalities_test.py"],
    deps = [
        ":modalities",
        # numpy dep
        # tensorflow dep
        # tensor2tensor/layers:common_hparams dep
        # tensor2tensor/utils:expert_utils dep
    ],
)
py_library(
    name = "music_encoders",
    srcs = ["music_encoders.py"],
    deps = [
        "//magenta",
        # tensor2tensor/data_generators:text_encoder dep
    ],
)

py_test(
    name = "music_encoders_test",
    srcs = ["music_encoders_test.py"],
    deps = [
        ":music_encoders",
        "//magenta",
        # tensorflow dep
    ],
)

py_library(
    name = "score2perf",
    srcs = [
        "__init__.py",
        "score2perf.py",
    ],
    visibility = ["//magenta:__subpackages__"],
    deps = [
        ":datagen_beam",
        ":modalities",
        ":music_encoders",
        # GCS dep
        # tensorflow dep
        # tensor2tensor/data_generators:problem dep
        # tensor2tensor/layers:modalities dep
        # tensor2tensor/utils:registry dep
    ],
)
