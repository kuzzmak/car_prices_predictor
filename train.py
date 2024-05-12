from datetime import datetime
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


def train(data_path: str, fields: List[Tuple[str, FieldType]], batch_size: int, epochs: int, lr: float, model_shapes: List[int], device: torch.device, epoch_log_frqg: int = 1, batch_log_freq: int = 100) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = SummaryWriter('runs/' + timestamp)

    datasets = get_datasets(data_path, fields, device)
    dataloaders = get_data_loaders(datasets, batch_size)
    model = MLP(model_shapes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train(True)
        train_loss = train_one_epoch(
            model, dataloaders['train'], optimizer, criterion, epoch, batch_log_freq)
        print(f'Epoch {epoch + 1} train loss: {train_loss}')

        logger.add_scalar('train_loss', train_loss, epoch)
        for tag, value in model.named_parameters():
            if value.grad is not None:
                logger.add_histogram(tag + '/grad', value.grad.cpu(), epoch)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloaders['val'])
        print(f'Validation loss: {avg_loss}')

        logger.add_scalar('val_loss', avg_loss, epoch)


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
        ('manufacturerId', FieldType.CATEGORICAL)
    ]
    batch_size = 64
    epochs = 10
    lr = 0.01
    model_shapes = [76, 128, 64, 32, 16, 1]
    model_shapes = [76, 32, 16, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(data_path, dataset_fields, batch_size, epochs, lr, model_shapes, device)
