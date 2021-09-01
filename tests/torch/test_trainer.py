import os
import tempfile

import pytest

from transformers4rec.config import trainer

pytorch = pytest.importorskip("torch")
torch4rec = pytest.importorskip("transformers4rec.torch")


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
        avg_session_length=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
    )
    resys_trainer = torch4rec.Trainer(model=torch_yoochoose_next_item_prediction_model, args=args)

    resys_trainer.train_dataloader = train_loader
    resys_trainer.eval_dataloader = eval_loader

    assert resys_trainer.get_train_dataloader().batch_size == batch_size
    assert resys_trainer.get_eval_dataloader().batch_size == batch_size // 2


@pytest.mark.parametrize("batch_size", [16, 32])
def test_set_train_eval_loaders_pyarrow(
    yoochoose_schema, yoochoose_path_file, torch_yoochoose_next_item_prediction_model, batch_size
):

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        avg_session_length=20,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        data_loader_engine="pyarrow",
        fp16=False,
        no_cuda=True,
    )
    resys_trainer = torch4rec.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=yoochoose_schema,
        train_dataset_or_path=yoochoose_path_file,
        eval_dataset_or_path=yoochoose_path_file,
    )

    assert resys_trainer.get_train_dataloader().batch_size == batch_size
    assert resys_trainer.get_eval_dataloader().batch_size == batch_size // 2


def test_set_train_eval_loaders_pyarrow_no_schema(
    yoochoose_path_file,
    torch_yoochoose_next_item_prediction_model,
):
    with pytest.raises(AssertionError) as excinfo:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=".",
            avg_session_length=20,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="pyarrow",
            fp16=False,
            no_cuda=True,
        )
        recsys_trainer = torch4rec.Trainer(
            model=torch_yoochoose_next_item_prediction_model,
            args=args,
            train_dataset_or_path=yoochoose_path_file,
            eval_dataset_or_path=yoochoose_path_file,
        )
        recsys_trainer.get_train_dataloader()

    assert "schema is required to generate Train Dataloader" in str(excinfo.value)


@pytest.mark.parametrize(
    "scheduler",
    ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
def test_create_scheduler(
    yoochoose_schema,
    yoochoose_path_file,
    torch_yoochoose_next_item_prediction_model,
    scheduler,
):
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        avg_session_length=20,
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

    recsys_trainer = torch4rec.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=yoochoose_schema,
        train_dataset_or_path=yoochoose_path_file,
        eval_dataset_or_path=yoochoose_path_file,
    )

    recsys_trainer.reset_lr_scheduler()
    result = recsys_trainer.train()
    assert result


def test_trainer_eval_loop(
    yoochoose_schema,
    yoochoose_path_file,
    torch_yoochoose_next_item_prediction_model,
):
    pytest.importorskip("pyarrow")
    batch_size = 16
    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        avg_session_length=20,
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

    recsys_trainer = torch4rec.Trainer(
        model=torch_yoochoose_next_item_prediction_model,
        args=args,
        schema=yoochoose_schema,
        train_dataset_or_path=yoochoose_path_file,
        eval_dataset_or_path=yoochoose_path_file,
        compute_metrics=True,
    )

    eval_metrics = recsys_trainer.evaluate(
        eval_dataset=yoochoose_path_file, metric_key_prefix="eval"
    )

    assert isinstance(eval_metrics, dict)
    default_metric = ["eval_ndcgat_10", "eval_ndcgat_20", "eval_recallat_10", "eval_recallat_20"]
    assert set(default_metric).issubset(set(eval_metrics.keys()))
    assert eval_metrics["eval_loss"] is not None


def test_saves_checkpoints(
    yoochoose_schema,
    yoochoose_path_file,
    torch_yoochoose_next_item_prediction_model,
):
    pytest.importorskip("pyarrow")
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_size = 16
        args = trainer.T4RecTrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=1,
            avg_session_length=20,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size // 2,
            data_loader_engine="pyarrow",
            max_sequence_length=20,
            fp16=False,
            no_cuda=True,
            report_to=[],
            debug=["r"],
        )
        recsys_trainer = torch4rec.Trainer(
            model=torch_yoochoose_next_item_prediction_model,
            args=args,
            schema=yoochoose_schema,
            train_dataset_or_path=yoochoose_path_file,
            eval_dataset_or_path=yoochoose_path_file,
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
