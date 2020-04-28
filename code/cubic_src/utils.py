import torch
from config import model, dev, loss_fn, test_loader, sampling_scheme


def get_accuracy(model_, dev_, loss_fn_, loader):
    model_.eval()
    correct, total, loss = (0, 0, 0)

    for batch_idx_, (data_, target_) in enumerate(loader):
        # Get Samples
        data_ = data_.to(dev_)
        target_ = target_.to(dev_)
        outputs = model_(data_)
        loss += loss_fn_(outputs, target_).detach() * len(target_)
        # Get prediction
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += len(target_)
        # Total correct predictions
        correct += (predicted == target_).sum().detach()
        del outputs
        del predicted
    acc = 100 * correct.item() / total
    loss = loss / total
    return loss.item(), acc


def init_train_loader(dataloader, train, sampling_scheme_name='fixed'):
    dataloader_args = dict(shuffle=True, batch_size=sampling_scheme[sampling_scheme_name], num_workers=4)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    return dataloader_args, train_loader


def print_acc(train_loader, epoch):
    train_loss, train_acc = get_accuracy(model, dev, loss_fn, train_loader)
    test_loss, test_acc = get_accuracy(model, dev, loss_fn, test_loader)
    print("Epoch {} Train Loss: {:.4f} Accuracy :{:.4f} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch, train_loss,
                                                                                                   train_acc, test_loss,
                                                                                                   test_acc))
