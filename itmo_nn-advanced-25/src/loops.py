import torch
from tqdm import tqdm
from torch import nn


def train_loop(
    net, dataloader,
    loss_func, optimizer,
    device='cpu', show_process=False
):
    """
    Function to train (classification task)
    ...
    
    Parameters
    ----------
        net : torch.nn.Module
            Neural Network to train.
        wavefronts_dataloader : torch.utils.data.DataLoader
            A loader (by batches) for the train dataset.
        loss_func :
            Loss function for a multi-class classification task.
        optimizer: torch.optim
            Optimizer...
        device : str
            Device to compute on...
        show_process : bool
            Flag to show (or not) a progress bar.
        
    Returns
    -------
        batches_losses : list[float]
            Losses for each batch in an epoch.
        batches_accuracies : list[float]
            Accuracies for each batch in an epoch.
        epoch_accuracy : float
            Accuracy for an epoch.
    """
    net.train()  # activate 'train' mode of a model
    batches_losses = []  # to store loss for each batch
    batches_accuracies = []  # to store accuracy for each batch
    
    correct_preds = 0
    size = 0
    ind_batch = 1
    
    for batch_seqs, batch_labels in tqdm(
        dataloader, total=len(dataloader),
        desc='train', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_seqs.size()[0]
        
        batch_seqs = batch_seqs.float().to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()

        # forward of an optical network
        batch_probas = net(batch_seqs)
        # calculate loss for a batch
        loss = loss_func(batch_probas, batch_labels)

        loss.backward()
        optimizer.step()

        # ACCURACY CALCULATION
        _, batch_preds = torch.max(batch_probas, dim=1)  # preds shape: (batch_size)
        batch_correct_preds = (batch_preds == batch_labels).sum().item()
        
        correct_preds += batch_correct_preds    
        size += batch_size
        
        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        batches_accuracies.append(batch_correct_preds / batch_size)

    epoch_accuracy = correct_preds / size
    
    return batches_losses, batches_accuracies, epoch_accuracy


def val_loop(
    net, dataloader,
    loss_func,
    device='cpu', show_process=False
    ):
    """
    Function to validate (classification task)
    ...
    
    Parameters
    ----------
        net : torch.nn.Module
            Neural Network to evaluate.
        dataloader : torch.utils.data.DataLoader
            A loader (by batches) for the evaluation dataset.
        loss_func :
            Loss function for a multi-class classification task.
        device : str
            Device to compute on...
        show_process : bool
            Flag to show (or not) a progress bar.
        
    Returns
    -------
        batches_losses : list[float]
            Losses for each batch in an epoch.
        batches_accuracies : list[float]
            Accuracies for each batch in an epoch.
        epoch_accuracy : float
            Accuracy for an epoch.
    """
    net.eval()  # activate 'eval' mode of a model
    batches_losses = []  # to store loss for each batch
    batches_accuracies = []  # to store accuracy for each batch
    
    correct_preds = 0
    size = 0

    for batch_seqs, batch_labels in tqdm(
        dataloader, total=len(dataloader),
        desc='validation', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        batch_size = batch_seqs.size()[0]
        
        batch_seqs = batch_seqs.float().to(device)
        batch_labels = batch_labels.to(device)
        
        with torch.no_grad():

            batch_probas = net(batch_seqs)
            # calculate loss for a batch
            loss = loss_func(batch_probas, batch_labels)

        # ACCURACY CALCULATION
        # get the predicted classes
        _, batch_preds = torch.max(batch_probas, dim=1)  # preds shape: (batch_size)
        # compare with ground truth
        batch_correct_preds = (batch_preds == batch_labels).sum().item()
        
        correct_preds += batch_correct_preds    
        size += batch_size

        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        batches_accuracies.append(batch_correct_preds / batch_size)

    epoch_accuracy = correct_preds / size
    
    return batches_losses, batches_accuracies, epoch_accuracy