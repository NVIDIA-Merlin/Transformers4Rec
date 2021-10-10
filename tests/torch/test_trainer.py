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

from transformers4rec.config import trainer

pytorch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers4rec.torch")


@pytest.mark.parametrize("batch_size", [16, 32])
def test_set_train_eval_loaders_attributes(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model, batch_size
):
    train_loader = pytorch.utils.data.DataLoader([torch_yoochoose_like], batch_size=batch_size)
    train_loader._batch_size = batch_size
    eval_loader = pytorch.utils.data.DataLoader([torch_yoochoose_like], batch_size=batch_size // 2)
    eval_loader._batch_size = batch_size // 2

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
    )

    assert recsys_trainer.get_train_dataloader() == train_loader
    assert recsys_trainer.get_eval_dataloader() == eval_loader


@pytest.mark.parametrize("batch_size", [16, 32])
def test_set_train_eval_loaders_pyarrow(torch_yoochoose_next_item_prediction_model, batch_size):

    data = tr.data.tabular_sequence_testing_data
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
        schema=data.schema,
        train_dataset_or_path=data.path,
        eval_dataset_or_path=data.path,
    )

    assert resys_trainer.get_train_dataloader().batch_size == batch_size
    assert resys_trainer.get_eval_dataloader().batch_size == batch_size // 2


def test_set_train_eval_loaders_pyarrow_no_schema(torch_yoochoose_next_item_prediction_model):
    with pytest.raises(AssertionError) as excinfo:
        batch_size = 16
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
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        max_sequence_length=20,
        fp16=False,
        no_cuda=True,
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


def test_trainer_eval_loop(torch_yoochoose_next_item_prediction_model):
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        max_sequence_length=20,
        fp16=False,
        no_cuda=True,
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

    eval_metrics = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")

    assert isinstance(eval_metrics, dict)
    default_metric = [
        "eval/next-item/ndcg_at_10",
        "eval/next-item/ndcg_at_20",
        "eval/next-item/avg_precision_at_10",
        "eval/next-item/avg_precision_at_20",
        "eval/next-item/recall_at_10",
        "eval/next-item/recall_at_20",
    ]
    assert set(default_metric).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval/loss"] is not None


def test_saves_checkpoints(torch_yoochoose_next_item_prediction_model):
    pytest.importorskip("pyarrow")
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=tmpdir,
            max_steps=5,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="pyarrow",
            max_sequence_length=20,
            fp16=False,
            no_cuda=True,
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


def test_evaluate_results(torch_yoochoose_next_item_prediction_model):
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        max_steps=5,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        max_sequence_length=20,
        fp16=False,
        no_cuda=True,
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
        "eval/next-item/ndcg_at_10",
        "eval/next-item/ndcg_at_20",
        "eval/next-item/recall_at_10",
        "eval/next-item/recall_at_20",
        "eval/loss",
    ]

    result_1 = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    result_1 = {k: result_1[k] for k in default_metric}

    result_2 = recsys_trainer.evaluate(eval_dataset=data.path, metric_key_prefix="eval")
    result_2 = {k: result_2[k] for k in default_metric}

    assert result_1 == result_2
