import pytest

from transformers4rec.config import trainer

pytorch = pytest.importorskip("torch")
DataLoader = pytest.importorskip("torch.utils.data.DataLoader")
torch4rec = pytest.importorskip("transformers4rec.torch")


@pytest.mark.parametrize("batch_size", [16, 32])
def test_set_train_eval_loaders(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model, batch_size
):

    train_loader = DataLoader([torch_yoochoose_like], batch_size=batch_size)
    train_loader._batch_size = batch_size
    eval_loader = DataLoader([torch_yoochoose_like], batch_size=batch_size // 2)
    eval_loader._batch_size = batch_size // 2

    args = trainer.T4RecTrainingArguments(
        output_dir=".",
        avg_session_length=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
    )
    resys_trainer = torch4rec.Trainer(model=torch_yoochoose_next_item_prediction_model, args=args)

    resys_trainer.set_train_dataloader(train_loader)
    resys_trainer.set_eval_dataloader(eval_loader)

    assert resys_trainer.get_train_dataloader().batch_size == batch_size
    assert resys_trainer.eval_dataloader.batch_size == batch_size // 2


def test_set_train_loader_wrong_batch_size(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model
):
    with pytest.raises(AssertionError) as excinfo:
        batch_size = 16
        train_loader = DataLoader([torch_yoochoose_like], batch_size=batch_size)
        train_loader._batch_size = batch_size

        args = trainer.T4RecTrainingArguments(
            output_dir=".",
            avg_session_length=20,
            **{
                "per_device_train_batch_size": batch_size * 2,
            }
        )
        resys_trainer = torch4rec.Trainer(
            model=torch_yoochoose_next_item_prediction_model, args=args
        )

        resys_trainer.set_train_dataloader(train_loader)

    assert "batch size of dataloader 16 should match" in str(excinfo.value)


def test_set_train_loader_wrong_drop_last(
    torch_yoochoose_like, torch_yoochoose_next_item_prediction_model
):
    with pytest.raises(AssertionError) as excinfo:
        batch_size = 16
        train_loader = DataLoader([torch_yoochoose_like], batch_size=batch_size, drop_last=True)
        train_loader._batch_size = batch_size

        args = trainer.T4RecTrainingArguments(
            output_dir=".",
            avg_session_length=20,
            **{
                "per_device_train_batch_size": batch_size,
            }
        )
        resys_trainer = torch4rec.Trainer(
            model=torch_yoochoose_next_item_prediction_model, args=args
        )

        resys_trainer.set_train_dataloader(train_loader)

    assert "Make sure drop_last is set to 'True' " in str(excinfo.value)
