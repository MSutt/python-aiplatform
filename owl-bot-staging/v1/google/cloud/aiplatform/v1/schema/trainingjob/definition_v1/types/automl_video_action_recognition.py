# -*- coding: utf-8 -*-
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import proto  # type: ignore


__protobuf__ = proto.module(
    package='google.cloud.aiplatform.v1.schema.trainingjob.definition',
    manifest={
        'AutoMlVideoActionRecognition',
        'AutoMlVideoActionRecognitionInputs',
    },
)


class AutoMlVideoActionRecognition(proto.Message):
    r"""A TrainingJob that trains and uploads an AutoML Video Action
    Recognition Model.

    Attributes:
        inputs (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlVideoActionRecognitionInputs):
            The input parameters of this TrainingJob.
    """

    inputs = proto.Field(
        proto.MESSAGE,
        number=1,
        message='AutoMlVideoActionRecognitionInputs',
    )


class AutoMlVideoActionRecognitionInputs(proto.Message):
    r"""

    Attributes:
        model_type (google.cloud.aiplatform.v1.schema.trainingjob.definition_v1.types.AutoMlVideoActionRecognitionInputs.ModelType):

    """
    class ModelType(proto.Enum):
        r""""""
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD = 1
        MOBILE_VERSATILE_1 = 2
        MOBILE_JETSON_VERSATILE_1 = 3
        MOBILE_CORAL_VERSATILE_1 = 4

    model_type = proto.Field(
        proto.ENUM,
        number=1,
        enum=ModelType,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
