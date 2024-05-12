from model import MLP
from dataset import get_data_loaders, get_datasets
from common import MODEL_CKPT_NAME, FieldType, PreprocessingType
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch
from datetime import datetime
import os
from pathlib import Path
from typing import List, Tuple

LOG_TO_TENSORBOARD = int(os.environ.get('LOG_TO_TENSORBOARD', 0))

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print('TensorboardX not installed. Logging to Tensorboard will be disabled.')
    LOG_TO_TENSORBOARD = 0


HERE = Path(__file__).resolve().parent


def train_one_epoch(
    model: MLP,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    log_freq: int = 10,
) -> float:
    """
    Trains the model for one epoch using the given dataloader and optimizer.

    Args:
        model (MLP): The model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        log_freq (int, optional): The frequency at which to log the training loss. Defaults to 10.

    Returns:
        float: The average loss over the last logged interval.
    """
    running_loss = 0.0
    last_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_freq == 0 and i > 0:
            last_loss = running_loss / log_freq
            running_loss = 0.0

    return last_loss


def train(
    data_path: str,
    fields: List[Tuple[str, FieldType]],
    batch_size: int,
    epochs: int,
    lr: float,
    model_shapes: List[int],
    device: torch.device,
    prepocessing_type: PreprocessingType,
    num_workers: int = 1,
    weight_decay: float = 0.0,
    epoch_log_frqg: int = 1,
    batch_log_freq: int = 100,
) -> None:
    """
    Trains a model using the provided data and hyperparameters.

    Args:
        data_path (str): The path to the data.
        fields (List[Tuple[str, FieldType]]): The fields and their types.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs to train for.
        lr (float): The learning rate.
        model_shapes (List[int]): The shapes of the model layers.
        device (torch.device): The device to train the model on.
        prepocessing_type (PreprocessingType): The type of preprocessing to
            apply to the data.
        num_workers (int, optional): The number of workers for data loading.
            Defaults to 1.
        weight_decay (float, optional): The weight decay for the optimizer.
            Defaults to 0.0.
        epoch_log_frqg (int, optional): The frequency of logging epoch
            information. Defaults to 1.
        batch_log_freq (int, optional): The frequency of logging batch
            information. Defaults to 100.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # create directories for saving checkpoints
    save_dir = HERE / 'ckpts' / timestamp
    save_dir.mkdir(parents=True)
    save_dir_last = save_dir / 'last'
    save_dir_last.mkdir()
    save_dir_best = save_dir / 'best'
    save_dir_best.mkdir()

    if LOG_TO_TENSORBOARD:
        logger = SummaryWriter('runs/' + timestamp)

    # prepare train, val and test datasets and dataloaders
    datasets = get_datasets(data_path, fields, device, prepocessing_type)
    dataloaders = get_data_loaders(datasets, batch_size, num_workers)
    model = MLP(model_shapes)
    model = model.to(device)

    best_loss = float('inf')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = torch.nn.MSELoss()

    desc = 'best loss: {best_loss} - train loss: {train_loss} - val loss: {val_loss} - lr: {lr}'
    pbar = tqdm(
        range(epochs),
        desc=desc.format(
            best_loss=best_loss,
            train_loss=0.0,
            val_loss=0.0,
            lr=lr,
        ),
    )

    for epoch in pbar:
        model.train(True)
        train_loss = train_one_epoch(
            model,
            dataloaders['train'],
            optimizer,
            criterion,
            batch_log_freq,
        )

        if LOG_TO_TENSORBOARD:
            logger.add_scalar('train_loss', train_loss, epoch)
            for tag, value in model.named_parameters():
                if value.grad is not None:
                    logger.add_histogram(
                        tag + '/grad',
                        value.grad.cpu(),
                        epoch,
                    )

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        val_loss = running_loss / len(dataloaders['val'])

        ckpt = {
            'sd': model.state_dict(),
            'shapes': model_shapes,
            'fields': [f[0] for f in fields],
        }

        if val_loss < best_loss:
            best_loss = val_loss

            # save best checkpoint
            ckpt_path = save_dir_best / MODEL_CKPT_NAME
            torch.save(ckpt, ckpt_path)

        # always save current checkpoint for resuming training
        ckpt_path = save_dir_last / MODEL_CKPT_NAME
        torch.save(ckpt, ckpt_path)

        pbar.set_description(
            desc.format(
                best_loss=best_loss,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
            )
        )

        scheduler.step(best_loss)

        if LOG_TO_TENSORBOARD:
            logger.add_scalar('val_loss', val_loss, epoch)
            logger.add_scalar('lr', scheduler.get_last_lr(), epoch)


if __name__ == '__main__':
    data_path = 'ML_zadatak_auti.csv'
    dataset_fields = [
        ('yearManufactured', FieldType.NUMERICAL),
        ('mileage', FieldType.NUMERICAL),
        ('motorSize', FieldType.NUMERICAL),
        ('motorPower', FieldType.NUMERICAL),
        # ('fuelConsumption', FieldType.NUMERICAL),
        # ('co2Emission', FieldType.NUMERICAL),
        ('transmissionTypeId', FieldType.CATEGORICAL),
        # ('manufacturerId', FieldType.CATEGORICAL)
    ]
    batch_size = 128
    epochs = 100
    lr = 0.1
    model_shapes = [76, 128, 128, 64, 32, 16, 1]
    model_shapes = [76, 20, 1]
    model_shapes = [8, 5, 3, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    weight_decay = 1e-3
    preprocessing_type = PreprocessingType.STANDARDIZATION
    num_workers = 0
    train(
        data_path,
        dataset_fields,
        batch_size,
        epochs,
        lr,
        model_shapes,
        device,
        preprocessing_type,
        num_workers,
        weight_decay,
    )
