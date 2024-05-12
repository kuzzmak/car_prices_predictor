from typing import List, Tuple

from tensorboardX import SummaryWriter
import torch

from common import FieldType
from dataset import get_data_loaders, get_datasets
from model import MLP


def train_one_epoch(model: MLP, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, epoch: int, log_freq: int = 10) -> float:
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
        if i % log_freq == 0:
            last_loss = running_loss / log_freq
            print(f'Batch: {i + 1} loss: {last_loss}')
            running_loss = 0.0

    return last_loss


def train(data_path: str, fields: List[Tuple[str, FieldType]], batch_size: int, epochs: int, lr: float, model_shapes: List[int], logger: SummaryWriter, epoch_log_frqg: int = 1, batch_log_freq: int = 100) -> None:
    datasets = get_datasets(data_path, fields)
    dataloaders = get_data_loaders(datasets, batch_size)
    model = MLP(model_shapes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = train_one_epoch(
            model, dataloaders['train'], optimizer, criterion, epoch, batch_log_freq)
        logger.add_scalar('loss', epoch_loss, epoch)
        print(f'Epoch {epoch + 1} loss: {epoch_loss}')


if __name__ == '__main__':
    data_path = 'ML_zadatak_auti.csv'
    dataset_fields = [
        ('yearManufactured', FieldType.NUMERICAL),
        ('mileage', FieldType.NUMERICAL),
        ('motorSize', FieldType.NUMERICAL),
        ('motorPower', FieldType.NUMERICAL),
        # ('fuelConsumption', FieldType.NUMERICAL),
        # ('co2Emission', FieldType.NUMERICAL),
        # ('transmissionTypeId', FieldType.CATEGORICAL),
    ]
    batch_size = 16
    epochs = 10
    lr = 0.01
    model_shapes = [4, 10, 10, 1]
    logger = SummaryWriter('runs', flush_secs=10)
    train(data_path, dataset_fields, batch_size, epochs, lr, model_shapes, logger)
