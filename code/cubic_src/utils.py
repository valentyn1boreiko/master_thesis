import torch
import config
import time
import scipy.linalg as la


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
    dataloader_args = dict(shuffle=True, batch_size=config.sampling_scheme[sampling_scheme_name], num_workers=4)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    return dataloader_args, train_loader


def print_acc(train_loader, epoch):
    train_loss, train_acc = get_accuracy(config.model, config.dev, config.loss_fn, train_loader)
    test_loss, test_acc = get_accuracy(config.model, config.dev, config.loss_fn, config.test_loader)
    print("Epoch {} Train Loss: {:.4f} Accuracy :{:.4f} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch, train_loss,
                                                                                                   train_acc, test_loss,
                                                                                                test_acc))


def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)


def get_grads_and_params(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        params.append(param)
        if param.grad is None:
            continue
        grads.append(param.grad + 0.)
    return flatten_tensor_list(grads), params


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    end3 = time.time()
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v,
                             only_inputs=True, retain_graph=True)
    print('hess vec product time: ', time.time() - end3)

    return flatten_tensor_list(hv)


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return torch.diag(a, k1) + torch.diag(b, k2) + torch.diag(c, k3)


def get_eigen(H_bmm, params=None, matrix=None, maxIter=10, tol=1e-3, method='lanczos'):
    """
    compute the top eigenvalues of model parameters and
    the corresponding eigenvectors.
    """
    # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    # If you call this function during training, remember to change the mode back to training mode.
    if params:
        q = flatten_tensor_list([torch.randn(p.size(), device=p.device) for p in params])
    else:
        q = torch.randn(matrix.size()[0])

    q = q / torch.norm(q)

    eigenvalue = None

    if method == 'power':
        # Power iteration
        for i in range(maxIter):
            Hv = H_bmm(q)
            eigenvalue_tmp = torch.dot(Hv, q)
            Hv_norm = torch.norm(Hv)
            if Hv_norm == 0:
                break
            q = Hv / Hv_norm
            if eigenvalue is None:
                eigenvalue = eigenvalue_tmp
            else:
                if abs(eigenvalue - eigenvalue_tmp) / abs(eigenvalue) < tol:
                    return eigenvalue_tmp, q
                else:
                    eigenvalue = eigenvalue_tmp
        return eigenvalue, q

    elif method == 'lanczos':
        # Lanczos iteration
        b = 0
        if params:
            q_last = flatten_tensor_list([torch.zeros(p.size(), device=p.device) for p in params])
        else:
            q_last = torch.zeros(matrix.size()[0])
        q_s = [q_last]
        a_s = []
        b_s = []
        for i in range(maxIter):
            Hv = H_bmm(q)
            a = torch.dot(Hv, q)
            Hv -= (b*q_last + a*q)
            q_last = q
            q_s.append(q_last)
            b = torch.norm(Hv)
            a_s.append(a)
            b_s.append(b)
            if b == 0:
                break
            q = Hv / b
        eigs, _ = la.eigh_tridiagonal(a_s, b_s[:-1])

        return max(eigs)


def get_hessian_eigen(gradsH, params, **kwargs):
    H_bmm = lambda x: hessian_vector_product(gradsH, params, x)
    return get_eigen(H_bmm, params, **kwargs)


def cauchy_point(grads, grads_norm, params, sigma):
    # Compute Cauchy radius
    # ToDo: replace hessian-vec product with the upper bound (beta)
    product = hessian_vector_product(grads, params, grads).t() @ grads / (sigma * grads_norm**2)
    R_c = -product + torch.sqrt(product**2 + 2*grads_norm/sigma)
    delta = -R_c*grads/grads_norm
    return delta


def sample_spherical(npoints, ndim=3):
    vec = torch.randn(ndim, npoints)
    vec /= vec.norm(p=2, dim=0)
    return vec


