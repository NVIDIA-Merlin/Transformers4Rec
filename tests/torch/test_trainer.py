#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os
import tempfile

import pytest
import torch
from merlin.schema import Schema

import transformers4rec.torch as tr
from transformers4rec.config import trainer
from transformers4rec.config import transformer as tconf


@pytest.mark.parametrize("batch_size", [16, 32])
def test_set_train_eval_loaders_attributes(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model, batch_size
):
    train_loader = torch.utils.data.DataLoader([torch_yoochoose_like], batch_size=batch_size)
    train_loader._batch_size = batch_size
    eval_loader = torch.utils.data.DataLoader([torch_yoochoose_like], batch_size=batch_size // 2)
    eval_loader._batch_size = batch_size // 2
    test_loader = torch.utils.data.DataLoader([torch_yoochoose_like], batch_size=batch_size // 2)
    test_loader._batch_size = batch_size // 2

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        max_sequence_length=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
    )
    recsys_trainer = tr.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        test_dataloader=test_loader,
    )

    assert recsys_trainer.get_train_dataloader() == train_loader
    assert recsys_trainer.get_eval_dataloader() == eval_loader
    assert recsys_trainer.get_test_dataloader() == test_loader


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("schema_type", ["msl", "core"])
def test_set_train_eval_loaders_pyarrow(
    torch_yoochoose_next_item_prediction_model, batch_size, schema_type
):
    data = tr.data.tabular_sequence_testing_data
    schema = data.schema if schema_type == "msl" else data.merlin_schema
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        fp16=False,
        no_cuda=True,
    )
    resys_trainer = tr.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
    )

    assert resys_trainer.get_train_dataloader().batch_size == batch_size
    assert resys_trainer.get_eval_dataloader().batch_size == batch_size // 2


def test_set_train_eval_loaders_no_schema(torch_yoochoose_next_item_prediction_model):
    with pytest.raises(AssertionError) as excinfo:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=".",
            max_steps=5,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="merlin_dataloader",
            fp16=False,
        )
        recsys_trainer = tr.Trainer(
            model=torch_yoochoose_next_item_prediction_model,
            args=args,
            train_dataset_or_path=tr.data.tabular_sequence_testing_data.path,
            eval_dataset_or_path=tr.data.tabular_sequence_testing_data.path,
        )
        recsys_trainer.get_train_dataloader()

    assert "schema is required to generate Train Dataloader" in str(excinfo.value)


@pytest.mark.parametrize(
    "scheduler",
    ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
def test_create_scheduler(torch_yoochoose_next_item_prediction_model, scheduler):
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=100,
        fp16=False,
        learning_rate_num_cosine_cycles_by_epoch=1.5,
        lr_scheduler_type=scheduler,
        report_to=[],
        debug=["r"],
    )

    data = tr.data.tabular_sequence_testing_data
    recsys_trainer = tr.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=data.schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
    )

    recsys_trainer.reset_lr_scheduler()
    result = recsys_trainer.train()
    assert result


@pytest.mark.parametrize("schema_type", ["msl", "core"])
def test_trainer_eval_loop(torch_yoochoose_next_item_prediction_model, schema_type):
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        max_sequence_length=100,
        fp16=False,
        no_cuda=True,
        report_to=[],
        debug=["r"],
    )

    data = tr.data.tabular_sequence_testing_data
    schema = data.schema if schema_type == "msl" else data.merlin_schema
    recsys_trainer = tr.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        test_dataset_or_path=data.path,
        compute_metrics=True,
    )

    eval_metrics = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    predictions = recsys_trainer.predict(data.path)

    assert isinstance(eval_metrics, dict)
    default_metric = [
        "eval_/next-item/ndcg_at_10",
        "eval_/next-item/ndcg_at_20",
        "eval_/next-item/avg_precision_at_10",
        "eval_/next-item/avg_precision_at_20",
        "eval_/next-item/recall_at_10",
        "eval_/next-item/recall_at_20",
    ]
    assert set(default_metric).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval_/loss"] is not None

    assert predictions is not None


def test_saves_checkpoints(torch_yoochoose_next_item_prediction_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=tmpdir,
            max_steps=5,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="merlin_dataloader",
            max_sequence_length=100,
            fp16=False,
            report_to=[],
            debug=["r"],
        )

        data = tr.data.tabular_sequence_testing_data
        recsys_trainer = tr.Trainer(
            model=torch_yoochoose_next_item_prediction_model,
            args=args,
            schema=data.schema,
            train_dataset_or_path=data.path,
            eval_dataset_or_path=data.path,
            compute_metrics=True,
        )

        recsys_trainer.train()
        recsys_trainer._save_model_and_checkpoint()

        file_list = [
            "pytorch_model.bin",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ]
        step = recsys_trainer.state.global_step
        checkpoint = os.path.join(tmpdir, f"checkpoint-{step}")

        assert os.path.isdir(checkpoint)
        for filename in file_list:
            assert os.path.isfile(os.path.join(checkpoint, filename))


def test_saves_checkpoints_best_metric(torch_yoochoose_next_item_prediction_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="merlin_dataloader",
            max_sequence_length=100,
            fp16=False,
            report_to=[],
            debug=["r"],
            save_total_limit=2,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="/next-item/recall_at_10",
        )
        data = tr.data.tabular_sequence_testing_data
        recsys_trainer = tr.Trainer(
            model=torch_yoochoose_next_item_prediction_model,
            args=args,
            schema=data.schema,
            train_dataset_or_path=data.path,
            eval_dataset_or_path=data.path,
            compute_metrics=True,
        )
        recsys_trainer.train()
        assert len(os.listdir(tmpdir)) == 1
        assert "checkpoint-100" in os.listdir(tmpdir)


def test_evaluate_results(torch_yoochoose_next_item_prediction_model):
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=100,
        fp16=False,
        report_to=[],
        debug=["r"],
    )

    data = tr.data.tabular_sequence_testing_data
    recsys_trainer = tr.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=data.schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        compute_metrics=True,
    )
    default_metric = [
        "eval_/next-item/ndcg_at_10",
        "eval_/next-item/ndcg_at_20",
        "eval_/next-item/recall_at_10",
        "eval_/next-item/recall_at_20",
        "eval_/loss",
    ]

    result_1 = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    result_1 = {k: result_1[k] for k in default_metric}

    result_2 = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    result_2 = {k: result_2[k] for k in default_metric}

    assert result_1 == result_2


@pytest.mark.parametrize(
    "task_and_metrics",
    [
        (
            tr.NextItemPredictionTask(weight_tying=True),
            [
                "eval_/next-item/ndcg_at_10",
                "eval_/next-item/ndcg_at_20",
                "eval_/next-item/avg_precision_at_10",
                "eval_/next-item/avg_precision_at_20",
                "eval_/next-item/recall_at_10",
                "eval_/next-item/recall_at_20",
            ],
        ),
        (
            tr.BinaryClassificationTask("click", summary_type="mean"),
            [
                "eval_/click/binary_classification_task/binary_accuracy",
                "eval_/click/binary_classification_task/binary_precision",
                "eval_/click/binary_classification_task/binary_recall",
            ],
        ),
        (
            tr.RegressionTask("play_percentage", summary_type="mean"),
            [
                "eval_/play_percentage/regression_task/mean_squared_error",
            ],
        ),
    ],
)
def test_trainer_music_streaming(task_and_metrics):
    data = tr.data.music_streaming_testing_data
    schema = data.schema
    batch_size = 16
    task, default_metric = task_and_metrics

    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        d_output=64,
        masking="mlm",
    )
    transformer_config = tconf.XLNetConfig.build(64, 4, 2, 20)
    model = transformer_config.to_torch_model(inputs, task)

    assert isinstance(model.input_schema, Schema)

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=20,
        fp16=False,
        report_to=[],
        debug=["r"],
    )

    recsys_trainer = tr.Trainer(
        model=model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        test_dataset_or_path=data.path,
        compute_metrics=True,
    )

    recsys_trainer.train()
    eval_metrics = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    predictions = recsys_trainer.predict(data.path)

    assert isinstance(eval_metrics, dict)
    assert set(default_metric).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval_/loss"] is not None

    assert predictions is not None
    # 1000 is the total samples in the testing data
    if isinstance(task, tr.NextItemPredictionTask):
        assert predictions.predictions.shape == (1000, task.target_dim)
    else:
        assert predictions.predictions.shape == (1000,)


# This is broken out as a separate test since combining it leads to strange errors
@pytest.mark.parametrize(
    "task_and_metrics",
    [
        (
            tr.NextItemPredictionTask(weight_tying=True),
            [
                "eval_/next-item/ndcg_at_10",
                "eval_/next-item/ndcg_at_20",
                "eval_/next-item/avg_precision_at_10",
                "eval_/next-item/avg_precision_at_20",
                "eval_/next-item/recall_at_10",
                "eval_/next-item/recall_at_20",
            ],
        ),
        (
            tr.BinaryClassificationTask("click", summary_type="mean"),
            [
                "eval_/click/binary_classification_task/binary_accuracy",
                "eval_/click/binary_classification_task/binary_precision",
                "eval_/click/binary_classification_task/binary_recall",
            ],
        ),
        (
            tr.RegressionTask("play_percentage", summary_type="mean"),
            [
                "eval_/play_percentage/regression_task/mean_squared_error",
            ],
        ),
    ],
)
def test_trainer_music_streaming_core_schema(task_and_metrics):
    data = tr.data.music_streaming_testing_data
    schema = data.merlin_schema
    batch_size = 16
    task, default_metric = task_and_metrics

    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        d_output=64,
        masking="mlm",
    )
    transformer_config = tconf.XLNetConfig.build(64, 4, 2, 20)
    model = transformer_config.to_torch_model(inputs, task)

    assert isinstance(model.input_schema, Schema)

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=20,
        fp16=False,
        report_to=[],
        debug=["r"],
    )

    recsys_trainer = tr.Trainer(
        model=model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        test_dataset_or_path=data.path,
        compute_metrics=True,
    )

    recsys_trainer.train()
    eval_metrics = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    predictions = recsys_trainer.predict(data.path)

    assert isinstance(eval_metrics, dict)
    assert set(default_metric).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval_/loss"] is not None

    assert predictions is not None


@pytest.mark.parametrize("schema_type", ["msl", "core"])
def test_trainer_with_multiple_tasks(schema_type):
    data = tr.data.music_streaming_testing_data
    schema = data.schema if schema_type == "msl" else data.merlin_schema
    batch_size = 16
    predict_top_k = 20
    tasks = [
        tr.NextItemPredictionTask(weight_tying=True),
        tr.BinaryClassificationTask("click", summary_type="mean"),
        tr.RegressionTask("play_percentage", summary_type="mean"),
    ]
    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        d_output=64,
        masking="mlm",
    )
    transformer_config = tconf.XLNetConfig.build(64, 4, 2, 20)

    body = tr.SequentialBlock(
        inputs,
        tr.MLPBlock([64]),
        tr.TransformerBlock(transformer_config),
    )
    model = tr.Model(tr.Head(body, tasks))

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=20,
        predict_top_k=predict_top_k,
        fp16=False,
        report_to=[],
        debug=["r"],
    )

    recsys_trainer = tr.Trainer(
        model=model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        test_dataset_or_path=data.path,
        compute_metrics=True,
    )

    recsys_trainer.train()
    eval_metrics = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    predictions = recsys_trainer.predict(data.path)

    default_metrics = [
        "eval_/next-item/ndcg_at_10",
        "eval_/next-item/ndcg_at_20",
        "eval_/next-item/avg_precision_at_10",
        "eval_/next-item/avg_precision_at_20",
        "eval_/next-item/recall_at_10",
        "eval_/next-item/recall_at_20",
        "eval_/click/binary_classification_task/binary_accuracy",
        "eval_/click/binary_classification_task/binary_precision",
        "eval_/click/binary_classification_task/binary_recall",
        "eval_/play_percentage/regression_task/mean_squared_error",
    ]

    assert isinstance(eval_metrics, dict)
    assert set(default_metrics).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval_/loss"] is not None

    assert predictions is not None
    assert predictions.predictions["next-item"][0].shape == (1000, predict_top_k)
    assert predictions.predictions["play_percentage/regression_task"].shape == (1000,)
    assert predictions.predictions["click/binary_classification_task"].shape == (1000,)


def test_trainer_trop_k_with_wrong_task():
    data = tr.data.music_streaming_testing_data
    schema = data.schema
    batch_size = 16
    predict_top_k = 20

    task = tr.BinaryClassificationTask("click", summary_type="mean")
    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        d_output=64,
    )
    transformer_config = tconf.XLNetConfig.build(64, 4, 2, 20)
    model = transformer_config.to_torch_model(inputs, task)

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="merlin_dataloader",
        max_sequence_length=20,
        predict_top_k=predict_top_k,
        report_to=[],
        debug=["r"],
    )

    recsys_trainer = tr.Trainer(
        model=model,
        args=args,
        schema=schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
        test_dataset_or_path=data.path,
        compute_metrics=True,
    )
    with pytest.raises(AssertionError) as excinfo:
        recsys_trainer.predict(data.path)

    assert "Top-k prediction is specific to NextItemPredictionTask" in str(excinfo.value)
