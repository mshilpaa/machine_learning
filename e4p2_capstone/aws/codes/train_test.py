def train(model, device, train_loader, optimizer, criterion):

  import torch
  model.train()

  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    # get samples
    # target = np.asarray(target)
    # target = torch.from_numpy(target.astype('long'))
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    y_pred = model(data)

    # Calculate loss
    # loss = F.nll_loss(y_pred, target)
    loss = criterion(y_pred, target)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
  return 100*correct/processed,loss.item()


def test(model, device, criterion, test_loader):

    import torch


    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # target = np.asarray(target)
            # target = torch.from_numpy(target.astype('long'))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset),test_loss/len(test_loader.dataset)
