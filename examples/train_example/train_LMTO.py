import sys
from torch.utils.data import DataLoader
from drxnet.drxnet.model import DRXNet
from drxnet.data.data import BatteryData_, collate_batch_

from drxnet.core import Featurizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR



def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_drxNet(train_loader, model, optimizer, epoch, device,
                 loss_Q0 = nn.L1Loss(), loss_Q = nn.L1Loss(), loss_dQdV = nn.MSELoss(),
                 ratio_Q0 = 1, ratio_Q = 1, ratio_dQ = 0.01, ratio_regu = 0.01, print_freq = 10000):


    losses = AverageMeter()

    Q0_mae_errors = AverageMeter()
    Q_mae_errors = AverageMeter()
    dQ_mae_errors = AverageMeter()

    Q0_mae_vals = AverageMeter()
    Q_mae_vals = AverageMeter()
    dQ_mae_vals = AverageMeter()

    # switch to train mode
    model.train()


    for ii, (inputs_, targets, *_) in enumerate(train_loader):

        inputs_var = (tensor.to(device) for tensor in inputs_)
        targets_var = [target.to(device) for target in targets]

        # compute output
        (output_Q0, output_Q, output_dQ, regu) = model(*inputs_var, return_direct = False)

        loss_1 = loss_Q0(output_Q0, targets_var[0]) * ratio_Q0
        loss_2 = loss_Q(output_Q, targets_var[1]) * ratio_Q # / 2 + loss_Q(output_Q_corr, targets_var[0]) * ratio_Q / 2
        loss_3 = loss_dQdV(output_dQ, targets_var[2]) * ratio_dQ


        loss_regu = regu * ratio_regu

        total_loss = loss_1 + loss_2 + loss_3  + loss_regu

        losses.update(total_loss.data.cpu().item(), len(targets))

        targets = torch.stack(targets, dim = 1)

        Q0_mae_error = mae(output_Q0.data.cpu(), targets[:, 0, :].reshape([-1, 1]))
        Q0_mae_errors.update(Q0_mae_error, len(targets))

        Q_mae_error = mae(output_Q.data.cpu(), targets[:, 1, :].reshape([-1, 1]))
        Q_mae_errors.update(Q_mae_error, targets.size(0))

        dQ_mae_error = mae(output_dQ.data.cpu(), targets[:, 2, :].reshape([-1, 1]))
        dQ_mae_errors.update(dQ_mae_error, targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()



        if (ii % print_freq) == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss: {loss.val:.4f}\t'
                          'Q0-MAE {Q0_mae_errors.val:.3f} ({Q0_mae_errors.avg:.3f})\t'
                          'Q-MAE {Q_mae_errors.val:.3f} ({Q_mae_errors.avg:.3f})\t'
                          'dQ-MAE {dQ_mae_errors.val:.3f} ({dQ_mae_errors.avg:.3f})\t'
                          'regu-loss {regu_loss:.3f}'.format(
                        epoch, ii+1, len(train_loader),
                        loss=losses, Q0_mae_errors=Q0_mae_errors,  Q_mae_errors = Q_mae_errors, dQ_mae_errors = dQ_mae_errors,
                        regu_loss = loss_regu.detach().cpu().numpy(),
                        )
                    )



def train():
    ####### Start the test ##########
    dataset =  BatteryData_(data_path= './dataset/',
                          fea_path= '../../drxnet/data/el-embeddings/matscholar-embedding.json',
                          add_noise = False)

    data_params = {
        "batch_size": 32,
        "pin_memory": True,
        "shuffle": True,
        "collate_fn": collate_batch_,
    }

    train_generator = DataLoader(dataset, **data_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device, ", device)


    model = DRXNet(elem_emb_len= 200, elem_fea_len = 32, vol_fea_len= 64,
                   rate_fea_len = 16, cycle_fea_len = 16,
                   n_graph = 3,
                   elem_heads=3,
                   elem_gate=[64],
                   elem_msg=[64],
                   cry_heads=3,
                   cry_gate=[64],
                   cry_msg=[64],
                   activation = nn.SiLU,
                   batchnorm_graph = False,
                   batchnorm_condition = True,
                   batchnorm_mix = True,
                   batchnorm_main = False,
                   )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), 1e-3,
                               weight_decay= 1*1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(40):
        train_drxNet(train_loader = train_generator,
                     model = model , optimizer = optimizer,
                     epoch = epoch, device= device,
                     loss_Q0 = nn.L1Loss(),
                     loss_Q = nn.MSELoss(),
                     loss_dQdV = nn.MSELoss(),
                     ratio_Q0 = 1,
                     ratio_Q = 1,
                     ratio_dQ = 1,
                     ratio_regu = 1e-4,
                     print_freq= 1,)
        scheduler.step()

    torch.save(model.state_dict(), './model_epoch_' + str(epoch+1))

if __name__ == "__main__":
    train()
