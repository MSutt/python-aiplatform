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

import copy
from importlib import reload
from unittest import mock
from unittest.mock import patch, call

import pytest
from google.api_core import exceptions
from google.api_core import operation
from google.auth import credentials


from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1 import (
    AddContextArtifactsAndExecutionsResponse,
    LineageSubgraph,
    Artifact as GapicArtifact,
    Context as GapicContext,
    Execution as GapicExecution,
    MetadataServiceClient,
    AddExecutionEventsResponse,
    MetadataStore as GapicMetadataStore,
    TensorboardServiceClient,
)
from google.cloud.aiplatform.compat.types import event as gca_event
from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.compat.types import (
    tensorboard_data as gca_tensorboard_data,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_experiment as gca_tensorboard_experiment,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_run as gca_tensorboard_run,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import experiment_run_resource
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform.metadata import metadata_store
from google.cloud.aiplatform.metadata import utils as metadata_utils

from google.cloud.aiplatform import utils

from test_pipeline_jobs import (
    mock_pipeline_service_get,
)  # noqa: F401
from test_pipeline_jobs import (
    _TEST_PIPELINE_JOB_NAME,
)  # noqa: F401

import test_pipeline_jobs
import test_tensorboard

# from sklearn import linear_model
import numpy as np

_TEST_PROJECT = "test-project"
_TEST_OTHER_PROJECT = "test-project-1"
_TEST_LOCATION = "us-central1"
_TEST_PARENT = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)
_TEST_EXPERIMENT = "test-experiment"
_TEST_EXPERIMENT_DESCRIPTION = "test-experiment-description"
_TEST_OTHER_EXPERIMENT_DESCRIPTION = "test-other-experiment-description"
_TEST_PIPELINE = _TEST_EXPERIMENT
_TEST_RUN = "run-1"
_TEST_OTHER_RUN = "run-2"
_TEST_DISPLAY_NAME = "test-display-name"

# resource attributes
_TEST_METADATA = {"test-param1": 1, "test-param2": "test-value", "test-param3": True}

# metadataStore
_TEST_METADATASTORE = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)

# context
_TEST_CONTEXT_ID = _TEST_EXPERIMENT
_TEST_CONTEXT_NAME = f"{_TEST_PARENT}/contexts/{_TEST_CONTEXT_ID}"

# execution
_TEST_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_RUN}"
_TEST_EXECUTION_NAME = f"{_TEST_PARENT}/executions/{_TEST_EXECUTION_ID}"
_TEST_OTHER_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_OTHER_RUN}"
_TEST_OTHER_EXECUTION_NAME = f"{_TEST_PARENT}/executions/{_TEST_OTHER_EXECUTION_ID}"
_TEST_SCHEMA_TITLE = "test.Schema"

_TEST_EXECUTION = GapicExecution(
    name=_TEST_EXECUTION_NAME,
    schema_title=_TEST_SCHEMA_TITLE,
    display_name=_TEST_DISPLAY_NAME,
    metadata=_TEST_METADATA,
    state=GapicExecution.State.RUNNING,
)

# artifact
_TEST_ARTIFACT_ID = f"{_TEST_EXPERIMENT}-{_TEST_RUN}-metrics"
_TEST_ARTIFACT_NAME = f"{_TEST_PARENT}/artifacts/{_TEST_ARTIFACT_ID}"
_TEST_OTHER_ARTIFACT_ID = f"{_TEST_EXPERIMENT}-{_TEST_OTHER_RUN}-metrics"
_TEST_OTHER_ARTIFACT_NAME = f"{_TEST_PARENT}/artifacts/{_TEST_OTHER_ARTIFACT_ID}"

# parameters
_TEST_PARAM_KEY_1 = "learning_rate"
_TEST_PARAM_KEY_2 = "dropout"
_TEST_PARAMS = {_TEST_PARAM_KEY_1: 0.01, _TEST_PARAM_KEY_2: 0.2}
_TEST_OTHER_PARAMS = {_TEST_PARAM_KEY_1: 0.02, _TEST_PARAM_KEY_2: 0.3}

# metrics
_TEST_METRIC_KEY_1 = "rmse"
_TEST_METRIC_KEY_2 = "accuracy"
_TEST_METRICS = {_TEST_METRIC_KEY_1: 222, _TEST_METRIC_KEY_2: 1}
_TEST_OTHER_METRICS = {_TEST_METRIC_KEY_2: 0.9}

# classification_metrics
_TEST_CLASSIFICATION_METRICS = {
    "display_name": "my-classification-metrics",
    "labels": ["cat", "dog"],
    "matrix": [[9, 1], [1, 9]],
    "fpr": [0.1, 0.5, 0.9],
    "tpr": [0.1, 0.7, 0.9],
    "threshold": [0.9, 0.5, 0.1],
}

# schema
_TEST_WRONG_SCHEMA_TITLE = "system.WrongSchema"

# tf model autologging
_TEST_TF_EXPERIMENT_RUN_PARAMS = {
    "batch_size": "None",
    "class_weight": "None",
    "epochs": "5",
    "initial_epoch": "0",
    "max_queue_size": "10",
    "sample_weight": "None",
    "shuffle": "True",
    "steps_per_epoch": "None",
    "use_multiprocessing": "False",
    "validation_batch_size": "None",
    "validation_freq": "1",
    "validation_split": "0.0",
    "validation_steps": "None",
    "workers": "1",
}
_TEST_TF_EXPERIMENT_RUN_METRICS = {
    "accuracy": 0.0,
    "loss": 1.013,
}


@pytest.fixture
def get_metadata_store_mock():
    with patch.object(
        MetadataServiceClient, "get_metadata_store"
    ) as get_metadata_store_mock:
        get_metadata_store_mock.return_value = GapicMetadataStore(
            name=_TEST_METADATASTORE,
        )
        yield get_metadata_store_mock


@pytest.fixture
def get_metadata_store_mock_raise_not_found_exception():
    with patch.object(
        MetadataServiceClient, "get_metadata_store"
    ) as get_metadata_store_mock:
        get_metadata_store_mock.side_effect = [
            exceptions.NotFound("Test store not found."),
            GapicMetadataStore(
                name=_TEST_METADATASTORE,
            ),
        ]

        yield get_metadata_store_mock


@pytest.fixture
def create_metadata_store_mock():
    with patch.object(
        MetadataServiceClient, "create_metadata_store"
    ) as create_metadata_store_mock:
        create_metadata_store_lro_mock = mock.Mock(operation.Operation)
        create_metadata_store_lro_mock.result.return_value = GapicMetadataStore(
            name=_TEST_METADATASTORE,
        )
        create_metadata_store_mock.return_value = create_metadata_store_lro_mock
        yield create_metadata_store_mock


@pytest.fixture
def get_context_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            description=_TEST_EXPERIMENT_DESCRIPTION,
            schema_title=constants.SYSTEM_EXPERIMENT,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_context_mock


@pytest.fixture
def get_context_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_context_wrong_schema_mock:
        get_context_wrong_schema_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_context_wrong_schema_mock


@pytest.fixture
def get_pipeline_context_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_pipeline_context_mock:
        get_pipeline_context_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            schema_title=constants.SYSTEM_PIPELINE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_PIPELINE],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_pipeline_context_mock


@pytest.fixture
def get_context_not_found_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_context_not_found_mock:
        get_context_not_found_mock.side_effect = exceptions.NotFound("test: not found")
        yield get_context_not_found_mock


_TEST_EXPERIMENT_CONTEXT = GapicContext(
    name=_TEST_CONTEXT_NAME,
    display_name=_TEST_EXPERIMENT,
    description=_TEST_EXPERIMENT_DESCRIPTION,
    schema_title=constants.SYSTEM_EXPERIMENT,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
    metadata={
        **constants.EXPERIMENT_METADATA,
        constants._BACKING_TENSORBOARD_RESOURCE_KEY: test_tensorboard._TEST_NAME,
    },
)


@pytest.fixture
def update_context_mock():
    with patch.object(MetadataServiceClient, "update_context") as update_context_mock:
        update_context_mock.return_value = _TEST_EXPERIMENT_CONTEXT
        yield update_context_mock


@pytest.fixture
def add_context_children_mock():
    with patch.object(
        MetadataServiceClient, "add_context_children"
    ) as add_context_children_mock:
        yield add_context_children_mock


@pytest.fixture
def add_context_artifacts_and_executions_mock():
    with patch.object(
        MetadataServiceClient, "add_context_artifacts_and_executions"
    ) as add_context_artifacts_and_executions_mock:
        add_context_artifacts_and_executions_mock.return_value = (
            AddContextArtifactsAndExecutionsResponse()
        )
        yield add_context_artifacts_and_executions_mock


@pytest.fixture
def get_execution_mock():
    with patch.object(MetadataServiceClient, "get_execution") as get_execution_mock:
        get_execution_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=constants.SYSTEM_RUN,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
        )
        yield get_execution_mock


@pytest.fixture
def get_execution_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_execution"
    ) as get_execution_wrong_schema_mock:
        get_execution_wrong_schema_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
        )
        yield get_execution_wrong_schema_mock


@pytest.fixture
def update_execution_mock():
    with patch.object(
        MetadataServiceClient, "update_execution"
    ) as update_execution_mock:
        update_execution_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=constants.SYSTEM_RUN,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
            metadata=_TEST_PARAMS,
        )
        yield update_execution_mock


@pytest.fixture
def add_execution_events_mock():
    with patch.object(
        MetadataServiceClient, "add_execution_events"
    ) as add_execution_events_mock:
        add_execution_events_mock.return_value = AddExecutionEventsResponse()
        yield add_execution_events_mock


@pytest.fixture
def list_executions_mock():
    with patch.object(MetadataServiceClient, "list_executions") as list_executions_mock:
        list_executions_mock.return_value = [
            GapicExecution(
                name=_TEST_EXECUTION_NAME,
                display_name=_TEST_RUN,
                schema_title=constants.SYSTEM_RUN,
                schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
                metadata=_TEST_PARAMS,
            ),
            GapicExecution(
                name=_TEST_OTHER_EXECUTION_NAME,
                display_name=_TEST_OTHER_RUN,
                schema_title=constants.SYSTEM_RUN,
                schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
                metadata=_TEST_OTHER_PARAMS,
            ),
        ]
        yield list_executions_mock


@pytest.fixture
def get_tensorboard_run_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_run"
    ) as get_tensorboard_run_mock:
        get_tensorboard_run_mock.side_effect = [
            exceptions.NotFound(""),
            test_tensorboard._TEST_TENSORBOARD_RUN,
        ]
        yield get_tensorboard_run_mock


@pytest.fixture
def get_tensorboard_experiment_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_experiment"
    ) as get_tensorboard_experiment_mock:
        get_tensorboard_experiment_mock.side_effect = [
            exceptions.NotFound(""),
            test_tensorboard._TEST_TENSORBOARD_EXPERIMENT,
        ]
        yield get_tensorboard_experiment_mock


@pytest.fixture
def get_tensorboard_time_series_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_time_series"
    ) as get_tensorboard_time_series_mock:
        get_tensorboard_time_series_mock.side_effect = [
            exceptions.NotFound(""),
            # test_tensorboard._TEST_TENSORBOARD_TIME_SERIES # change to time series
        ]
        yield get_tensorboard_time_series_mock


@pytest.fixture
def query_execution_inputs_and_outputs_mock():
    with patch.object(
        MetadataServiceClient, "query_execution_inputs_and_outputs"
    ) as query_execution_inputs_and_outputs_mock:
        query_execution_inputs_and_outputs_mock.side_effect = [
            LineageSubgraph(
                artifacts=[
                    GapicArtifact(
                        name=_TEST_ARTIFACT_NAME,
                        display_name=_TEST_ARTIFACT_ID,
                        schema_title=constants.SYSTEM_METRICS,
                        schema_version=constants.SCHEMA_VERSIONS[
                            constants.SYSTEM_METRICS
                        ],
                        metadata=_TEST_METRICS,
                    )
                ],
                events=[
                    gca_event.Event(
                        artifact=_TEST_ARTIFACT_NAME,
                        execution=_TEST_EXECUTION_NAME,
                        type_=gca_event.Event.Type.OUTPUT,
                    )
                ],
            ),
            LineageSubgraph(
                artifacts=[
                    GapicArtifact(
                        name=_TEST_OTHER_ARTIFACT_NAME,
                        display_name=_TEST_OTHER_ARTIFACT_ID,
                        schema_title=constants.SYSTEM_METRICS,
                        schema_version=constants.SCHEMA_VERSIONS[
                            constants.SYSTEM_METRICS
                        ],
                        metadata=_TEST_OTHER_METRICS,
                    ),
                ],
                events=[
                    gca_event.Event(
                        artifact=_TEST_OTHER_ARTIFACT_NAME,
                        execution=_TEST_OTHER_EXECUTION_NAME,
                        type_=gca_event.Event.Type.OUTPUT,
                    )
                ],
            ),
        ]
        yield query_execution_inputs_and_outputs_mock


_TEST_CLASSIFICATION_METRICS_METADATA = {
    "confusionMatrix": {
        "annotationSpecs": [{"displayName": "cat"}, {"displayName": "dog"}],
        "rows": [[9, 1], [1, 9]],
    },
    "confidenceMetrics": [
        {"confidenceThreshold": 0.9, "recall": 0.1, "falsePositiveRate": 0.1},
        {"confidenceThreshold": 0.5, "recall": 0.7, "falsePositiveRate": 0.5},
        {"confidenceThreshold": 0.1, "recall": 0.9, "falsePositiveRate": 0.9},
    ],
}

_TEST_CLASSIFICATION_METRICS_ARTIFACT = GapicArtifact(
    name=_TEST_ARTIFACT_NAME,
    display_name=_TEST_CLASSIFICATION_METRICS["display_name"],
    schema_title=constants.GOOGLE_CLASSIFICATION_METRICS,
    schema_version=constants._DEFAULT_SCHEMA_VERSION,
    metadata=_TEST_CLASSIFICATION_METRICS_METADATA,
    state=GapicArtifact.State.LIVE,
)


@pytest.fixture
def create_classification_metrics_artifact_mock():
    with patch.object(
        MetadataServiceClient, "create_artifact"
    ) as create_classification_metrics_artifact_mock:
        create_classification_metrics_artifact_mock.return_value = (
            _TEST_CLASSIFICATION_METRICS_ARTIFACT
        )
        yield create_classification_metrics_artifact_mock


@pytest.fixture
def get_classification_metrics_artifact_mock():
    with patch.object(
        MetadataServiceClient, "get_artifact"
    ) as get_classification_metrics_artifact_mock:
        get_classification_metrics_artifact_mock.return_value = (
            _TEST_CLASSIFICATION_METRICS_ARTIFACT
        )
        yield get_classification_metrics_artifact_mock


@pytest.fixture
def get_artifact_mock():
    with patch.object(MetadataServiceClient, "get_artifact") as get_artifact_mock:
        get_artifact_mock.return_value = GapicArtifact(
            name=_TEST_ARTIFACT_NAME,
            display_name=_TEST_ARTIFACT_ID,
            schema_title=constants.SYSTEM_METRICS,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_METRICS],
        )
        yield get_artifact_mock


@pytest.fixture
def get_artifact_not_found_mock():
    with patch.object(MetadataServiceClient, "get_artifact") as get_artifact_mock:
        get_artifact_mock.side_effect = exceptions.NotFound("")
        yield get_artifact_mock


@pytest.fixture
def get_artifact_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_artifact"
    ) as get_artifact_wrong_schema_mock:
        get_artifact_wrong_schema_mock.return_value = GapicArtifact(
            name=_TEST_ARTIFACT_NAME,
            display_name=_TEST_ARTIFACT_ID,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_METRICS],
        )
        yield get_artifact_wrong_schema_mock


@pytest.fixture
def update_artifact_mock():
    with patch.object(MetadataServiceClient, "update_artifact") as update_artifact_mock:
        update_artifact_mock.return_value = GapicArtifact(
            name=_TEST_ARTIFACT_NAME,
            display_name=_TEST_ARTIFACT_ID,
            schema_title=constants.SYSTEM_METRICS,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_METRICS],
            metadata=_TEST_METRICS,
        )
        yield update_artifact_mock


_TEST_EXPERIMENT_RUN_CONTEXT_NAME = f"{_TEST_PARENT}/contexts/{_TEST_EXECUTION_ID}"
_TEST_OTHER_EXPERIMENT_RUN_CONTEXT_NAME = (
    f"{_TEST_PARENT}/contexts/{_TEST_OTHER_EXECUTION_ID}"
)

_EXPERIMENT_MOCK = GapicContext(
    name=_TEST_CONTEXT_NAME,
    display_name=_TEST_EXPERIMENT,
    description=_TEST_EXPERIMENT_DESCRIPTION,
    schema_title=constants.SYSTEM_EXPERIMENT,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
    metadata={**constants.EXPERIMENT_METADATA},
)

_EXPERIMENT_RUN_MOCK = GapicContext(
    name=_TEST_EXPERIMENT_RUN_CONTEXT_NAME,
    display_name=_TEST_RUN,
    schema_title=constants.SYSTEM_EXPERIMENT_RUN,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT_RUN],
    metadata={
        constants._PARAM_KEY: {},
        constants._METRIC_KEY: {},
        constants._STATE_KEY: gca_execution.Execution.State.RUNNING.name,
    },
)

_EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT = copy.deepcopy(_EXPERIMENT_RUN_MOCK)
_EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT.parent_contexts = [_TEST_CONTEXT_NAME]


@pytest.fixture
def get_experiment_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.return_value = _EXPERIMENT_MOCK
        yield get_context_mock


@pytest.fixture
def get_experiment_run_run_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.side_effect = [
            _EXPERIMENT_MOCK,
            _EXPERIMENT_RUN_MOCK,
            _EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT,
        ]

        yield get_context_mock


@pytest.fixture
def get_experiment_run_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.side_effect = [
            _EXPERIMENT_MOCK,
            _EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT,
        ]

        yield get_context_mock


@pytest.fixture
def create_experiment_context_mock():
    with patch.object(MetadataServiceClient, "create_context") as create_context_mock:
        create_context_mock.side_effect = [_TEST_EXPERIMENT_CONTEXT]
        yield create_context_mock


@pytest.fixture
def create_experiment_run_context_mock():
    with patch.object(MetadataServiceClient, "create_context") as create_context_mock:
        create_context_mock.side_effect = [_EXPERIMENT_RUN_MOCK]
        yield create_context_mock


@pytest.fixture
def update_experiment_run_context_to_running():
    with patch.object(MetadataServiceClient, "update_context") as update_context_mock:
        update_context_mock.side_effect = [_EXPERIMENT_RUN_MOCK]
        yield update_context_mock


@pytest.fixture
def create_execution_mock():
    with patch.object(
        MetadataServiceClient, "create_execution"
    ) as create_execution_mock:
        create_execution_mock.side_effect = [_TEST_EXECUTION]
        yield create_execution_mock


@pytest.fixture
def update_context_mock_v2():
    with patch.object(MetadataServiceClient, "update_context") as update_context_mock:
        update_context_mock.side_effect = [
            # experiment run
            GapicContext(
                name=_TEST_EXPERIMENT_RUN_CONTEXT_NAME,
                display_name=_TEST_RUN,
                schema_title=constants.SYSTEM_EXPERIMENT_RUN,
                schema_version=constants.SCHEMA_VERSIONS[
                    constants.SYSTEM_EXPERIMENT_RUN
                ],
                metadata={**constants.EXPERIMENT_METADATA},
            ),
            # experiment run
            GapicContext(
                name=_TEST_EXPERIMENT_RUN_CONTEXT_NAME,
                display_name=_TEST_RUN,
                schema_title=constants.SYSTEM_EXPERIMENT_RUN,
                schema_version=constants.SCHEMA_VERSIONS[
                    constants.SYSTEM_EXPERIMENT_RUN
                ],
                metadata=constants.EXPERIMENT_METADATA,
                parent_contexts=[_TEST_CONTEXT_NAME],
            ),
        ]

        yield update_context_mock


@pytest.mark.usefixtures("google_auth_mock")
class TestAutologging:
    def setup_method(self):
        reload(initializer)
        reload(metadata)
        reload(aiplatform)

    def teardown_method(self):
        initializer.global_pool.shutdown(wait=True)

    def test_autologging_init(
        self,
        get_experiment_mock,
        get_metadata_store_mock,
    ):
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLFlow is not installed and is required to test autologging. "
                'Please install the SDK using "pip install google-cloud-aiplatform[autologging]"'
            )
        aiplatform.init(
            project=_TEST_PROJECT, location=_TEST_LOCATION, experiment=_TEST_EXPERIMENT
        )

        aiplatform.autolog()

    def test_autologging_raises_if_experiment_not_set(
        self,
    ):
        aiplatform.init(project=_TEST_PROJECT, location=_TEST_LOCATION)

        with pytest.raises(ValueError):
            aiplatform.autolog()

    def test_autologging_sets_and_resets_mlflow_tracking_uri(
        self, get_experiment_mock, get_metadata_store_mock
    ):
        import mlflow

        aiplatform.init(
            project=_TEST_PROJECT, location=_TEST_LOCATION, experiment=_TEST_EXPERIMENT
        )
        mlflow.set_tracking_uri("file://my-test-tracking-uri")

        aiplatform.autolog()

        assert mlflow.get_tracking_uri() == "vertex-mlflow-plugin://"

        aiplatform.autolog(disable=True)

        assert mlflow.get_tracking_uri() == "file://my-test-tracking-uri"

    def test_autologging_with_auto_run_creation(
        self,
        get_experiment_mock,
        get_metadata_store_mock,
        get_experiment_run_mock,
        create_experiment_run_context_mock,
        add_context_children_mock,
        update_context_mock,
    ):

        import tensorflow as tf

        aiplatform.init(
            project=_TEST_PROJECT, location=_TEST_LOCATION, experiment=_TEST_EXPERIMENT
        )

        aiplatform.autolog()

        X = np.array(
            [
                [1, 1],
                [1, 2],
                [2, 2],
                [2, 3],
                [1, 1],
                [1, 2],
                [2, 2],
                [2, 3],
                [1, 1],
                [1, 2],
                [2, 2],
                [2, 3],
            ]
        )
        y = np.dot(X, np.array([1, 2])) + 3

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(2,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        model.fit(X, y, epochs=5)
        
        # _TRUE_CONTEXT = copy.deepcopy(_EXPERIMENT_RUN_MOCK)
        # _TRUE_CONTEXT.metadata[constants._PARAM_KEY].update(_TEST_TF_EXPERIMENT_RUN_PARAMS)
        # _TRUE_CONTEXT.metadata[constants._METRIC_KEY].update(_TEST_TF_EXPERIMENT_RUN_METRICS)
        
        # update_context_mock.assert_called_once_with(context=_TRUE_CONTEXT)

        # for args, kwargs in create_experiment_run_context_mock.call_args_list:
        #     assert kwargs["context"].display_name.startswith("tensorflow")
        #     assert kwargs["context_id"].startswith(f"{_TEST_EXPERIMENT}-tensorflow")
