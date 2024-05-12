from datetime import datetime
import os
from typing import List, Tuple

from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from common import FieldType, PreprocessingType
from dataset import get_data_loaders, get_datasets
from model import MLP


LOG_TO_TENSORBOARD = int(os.environ.get('LOG_TO_TENSORBOARD', 0))


def train_one_epoch(
    model: MLP,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epoch: int,
    log_freq: int = 10,
) -> float:
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if LOG_TO_TENSORBOARD:
        logger = SummaryWriter('runs/' + timestamp)

    datasets = get_datasets(data_path, fields, device, prepocessing_type)
    dataloaders = get_data_loaders(datasets, batch_size, num_workers)
    model = MLP(model_shapes)
    model = model.to(device)

    best_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs * 2, eta_min=1e-5)
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
            epoch,
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

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        val_loss = running_loss / len(dataloaders['val'])
        if val_loss < best_loss:
            best_loss = val_loss
            # torch.save(model.state_dict(), f'best_model_{timestamp}.pth')

        pbar.set_description(
            desc.format(
                best_loss=best_loss,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
            )
        )

        # scheduler.step(best_loss)

        if LOG_TO_TENSORBOARD:
            logger.add_scalar('val_loss', val_loss, epoch)
            # logger.add_scalar('lr', scheduler.get_last_lr(), epoch)


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
