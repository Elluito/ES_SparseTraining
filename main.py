# Sparse learning imports
from rigl_repo_utils.models.resnet import resnet50, ResNet
from rigl_repo_utils.models.wide_resnet import WideResNet
from rigl_repo_utils.sparselearning.core import Masking

from rigl_repo_utils.data import get_dataloaders
from rigl_repo_utils.loss import LabelSmoothingCrossEntropy
from rigl_repo_utils.models import registry as model_registry
from sparselearning.counting.ops import get_inference_FLOPs
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from sparselearning.utils.smoothen_value import SmoothenValue
from sparselearning.utils.train_helper import (
    get_optimizer,
    load_weights,
    save_weights,
)
from rigl_repo_utils.main import train, single_seed_run
# Imports from sparselearning (is rigl_repo_utils but installed as a package)
from sparselearning.utils import layer_wise_density
# Hessian_eigenthings imports
from hessian_eigenthings import compute_hessian_eigenthings
# Hydra Imports
import hydra
from omegaconf import OmegaConf, DictConfig
# Scientific computation imports
import numpy as np
# CMA-ES library imports
from cmaes import CMA
# Torch imports
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import typing
# WANDB
import wandb
# Other imports
import datetime as date
import tqdm
import copy
import logging

################### class cefinitions #########################################

class MNISTModel(pl.LightningModule):

    def __init__(self, data_dir=".", hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(channels * width * height, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, self.num_classes),
        # )
        self.model = get_mnist_model(hidden_size)

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)

    def on_train_batch_start(self, batch: typing.Any, batch_idx: int, unused: int = 0) -> typing.Optional[int]:

        global found_best_event
        if found_best_event.value:
            return -1
        else:
            return 1


class SimpleMaskingWrapping(Masking):
    def __init__(self, module:nn.Module, sparsity:float):
        self.module = module
        self.sparsity = sparsity


########################### FUNCTION DEFINITIONS #######################################
def get_mnist_model(hidden_size):
    num_classes = 10
    channels, width, height = (1, 28, 28)
    # Define PyTorch model

    model = nn.Sequential(

        nn.Flatten(),
        nn.Linear(channels * width * height, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )

    return model


def minimalCMA_ES():
    def quadratic(x1, x2):
        return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)


def get_cifar10(cfg: DictConfig):
    BATCH_SIZE = cfg.dataset.batch_size
    USABLE_CORES = cfg.dataset.max_threads
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Assign train/val datasets for use in dataloaders
    cifar_full = torchvision.datasets.CIFAR10(cfg.dataset.root, train=True, download=True,
                                              transform=train_transform)
    cifar10_train, cifar10_val = random_split(cifar_full, [45000, 5000])

    # Assign test dataset mfor use in dataloader(s)
    cifar10_test = torchvision.datasets.CIFAR10(cfg.dataset.root, train=False, transform=test_transform)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    val_loader = DataLoader(cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, valLoader: DataLoader, device: torch.device, loss_object: typing.Callable,
             epoch: int, global_step: int, training_flops: float, is_test_set: bool = False, use_wandb: bool = False):
    model.eval()
    top1_list = []
    top5_list = []
    loss = 0
    pbar = tqdm.tqdm(total=len(valLoader), dynamic_ncols=True)
    with torch.no_grad():
        for inputs, targets in valLoader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            loss += loss_object(y_pred, targets).item()

            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                F.log_softmax(y_pred, dim=1), targets, topk=(1, 5)
            )
            top1_list.append(top_1_accuracy)
            top5_list.append(top_5_accuracy)

            pbar.update(1)

    loss /= len(valLoader)
    mean_top_1_accuracy = torch.tensor(top1_list).mean()
    mean_top_5_accuracy = torch.tensor(top5_list).mean()

    val_or_test = "val" if not is_test_set else "test"
    msg = f"{val_or_test.capitalize()} Epoch {epoch} {val_or_test} loss {loss:.6f} top-1 accuracy" \
          f" {mean_top_1_accuracy:.4f} top-5 accuracy {mean_top_5_accuracy:.4f}"
    pbar.set_description(msg)
    logging.info(msg)

    if use_wandb:
        wandb.log({f"{val_or_test}_loss": loss, f"{val_or_test}_accuracy": top_1_accuracy,
                   f"{val_or_test}_top_5_accuracy": top_5_accuracy, "train_FLOPS": training_flops, "EPOCH": epoch})


def get_cifar100(cfg: DictConfig):
    BATCH_SIZE = cfg.dataset.batch_size
    USABLE_CORES = cfg.dataset.max_threads
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Assign train/val datasets for use in dataloaders
    cifar_full = torchvision.datasets.CIFAR100(cfg.dataset.root, train=True, download=True,
                                               transform=train_transform)
    cifar10_train, cifar10_val = random_split(cifar_full, [45000, 5000])

    # Assign test dataset mfor use in dataloader(s)
    cifar10_test = torchvision.datasets.CIFAR10(cfg.dataset.root, train=False, transform=test_transform)

    train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    val_loader = DataLoader(cifar10_val, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=USABLE_CORES)
    return train_loader, val_loader, test_loader


def get_mnist(cfg: DictConfig):
    BATCH_SIZE = cfg.dataset.batch_size
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = train_transform

    # Assign train/val datasets for use in dataloaders
    mnist_full = MNIST(cfg.dataset.root, train=True, download=True, transform=train_transform)
    mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])

    # Assign test dataset mfor use in dataloader(s)
    mnist_test = MNIST(cfg.dataset.root, train=False, transform=test_transform)

    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, num_workers=cfg.dataset.max_threads)
    val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, num_workers=cfg.dataset.max_threads)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, num_workers=cfg.dataset.max_threads)
    return train_loader, val_loader, test_loader


def GA_MNIST(cfg: DictConfig):
    pass


@torch.no_grad()
def calculate_batch_loss(individual: np.ndarray, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                         loss_object: typing.Callable) -> float:
    # Put vector in model
    vector_to_parameters(torch.tensor(individual, dtype=torch.float32), model.parameters())
    model.cuda()
    data.to("cuda")
    target.to("cuda")
    y_hat = model(data)
    loss = loss_object(y_hat, target)
    model.cpu()
    return loss.detach().numpy()


def CMA_ES_MNIST(cfg: DictConfig):
    population = cfg.optimiser.population
    epochs = cfg.optimiser.epochs
    model = get_mnist_model(20)
    loss_object = F.nll_loss
    mu_o = parameters_to_vector(model.parameters()).detach().numpy()
    cov_0 = None
    if cfg.wandb:
        now = date.datetime.now().strftime("%M:%S")
        name = "CMA-ES_WS_eig" if cfg.optimiser.eigen_start else "CMA-ES"
        wandb.init(
            entity="luis_alfredo",
            config=OmegaConf.to_container(cfg, resolve=True),
            project="ES_sparse_training",
            name=name + f"_{now}",
            tags=["MNIST"],
            reinit=True,
            save_code=True,
        )
    if cfg.optimiser.eigen_start:
        mu_o

    optimiser = CMA(mu_o, sigma=0.001, cov=cov_0, population_size=population)
    train_loader, val_loader, test_loader = get_mnist(cfg)
    global_step = 0
    train_flops = 0
    pbar = tqdm.tqdm(total=len(train_loader), dynamic_ncols=True)
    pbar2 = tqdm.tqdm(total=optimiser.population_size, dynamic_ncols=True)
    forwardpass_FLOPS = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = data.to(device), target.to(device)
            solutions = []
            mean = 0

            for _ in range(optimiser.population_size):
                x = optimiser.ask()
                value = calculate_batch_loss(x, model, data, target, loss_object)
                if forwardpass_FLOPS == 0:
                    forwardpass_FLOPS = get_inference_FLOPs(SimpleMaskingWrapping(model,1),data)
                else:
                    train_flops += forwardpass_FLOPS
                solutions.append((x, value))
                mean = mean + (value - mean) / (_ + 1)

                # print(f"#{} {value}")
                pbar2.update(1)
                pbar2.set_description_str(f"Individual:{_}, value:{value}")

            global_step += 1
            if cfg.wandb:
                wandb.log({"step": global_step, "loss": mean})
            optimiser.tell(solutions)
            pbar.update(1)
            pbar.set_description(f"Epoch:{epoch} Loss: {mean}")
        mean_solution = copy.deepcopy(optimiser._mean)
        vector_to_parameters(torch.tensor(mean_solution, dtype=torch.float32), model.parameters())
        evaluate(model, val_loader, torch.device("cuda"), loss_object, epoch, global_step, train_flops,
                 is_test_set=False,
                 use_wandb=True)
    # Evaluate the  mean of the distribution
    mean_solution = copy.deepcopy(optimiser._mean)
    vector_to_parameters(torch.tensor(mean_solution, dtype=torch.float32), model.parameters())
    evaluate(model,test_loader, torch.device("cuda"), loss_object, epochs, global_step, train_flops,
             is_test_set=True,
             use_wandb=True)
    if cfg.wandb:
        wandb.join()


def CMA_ES_CIFAR10(cfg: DictConfig):
    pass


def CMA_ES_CIFAR100(cfg: DictConfig):
    pass


@hydra.main(config_path="configs", config_name="template")
def main(cfg: DictConfig):
    print(cfg)
    if cfg.optimiser.name == "CMAES":
        if cfg.dataset.name == "MNIST":
            CMA_ES_MNIST(cfg)


if __name__ == '__main__':
    main()
