""" import torch
import torch.nn.functional as F

def velocity_loss(v_pred, S):

    S_next = S + v_pred

    loss = F.mse_loss(
        S_next[:, :-1],
        S[:, 1:]
    )

    return loss


def kinetic_loss(v_pred, v_kinetic):

    return F.mse_loss(v_pred, v_kinetic) """


import torch
import torch.nn.functional as F


class TransVeloLoss:
    """
    TransVelo Loss Module
    Low coupling, easy to plug into any model.
    """

    def __init__(
        self,
        w_velocity=1.0,
        w_kinetic=0.5,
        w_smooth=0.2,
        w_direction=0.1
    ):

        self.w_velocity = w_velocity
        self.w_kinetic = w_kinetic
        self.w_smooth = w_smooth
        self.w_direction = w_direction

    # --------------------------------------------------
    # 1 Velocity Prediction Loss
    # --------------------------------------------------
    # new version (updated 3-8)
    def velocity_loss(self, v_pred, S):

        #Predict next spliced state

        #S_next = S + v
        # gene-wise difference
        target = torch.zeros_like(S)

        target[:, 1:] = S[:, 1:] - S[:, :-1]

        loss = F.mse_loss(v_pred, target)

        """ S_next = S + v_pred

        loss = F.mse_loss(
            S_next[:, :-1],
            S[:, 1:]
        ) """

        return loss


    # --------------------------------------------------
    # 2 Kinetic Consistency Loss
    # --------------------------------------------------
    def kinetic_loss(self, v_pred, v_kinetic):

        return F.mse_loss(v_pred, v_kinetic)

    # --------------------------------------------------
    # 3 Graph Smoothness Loss
    # --------------------------------------------------
    def smooth_loss(self, v_pred, adj=None):
        """
        Encourage neighbor cells to have similar velocity
        """

        if adj is None:
            return torch.tensor(
                0.0,
                device=v_pred.device
            )

        if adj.is_sparse:

            row, col = adj.indices()

            diff = v_pred[row] - v_pred[col]

            loss = (diff ** 2).sum(dim=1).mean()

        else:

            diff = v_pred.unsqueeze(1) - v_pred.unsqueeze(0)

            loss = (adj * diff.pow(2).sum(-1)).mean()

        return loss

    # --------------------------------------------------
    # 4 Direction Consistency Loss
    # --------------------------------------------------
    def direction_loss(self, v_pred, x, adj=None):
        """
        Velocity should point to neighbor cells
        """

        if adj is None:
            return torch.tensor(
                0.0,
                device=v_pred.device
            )

        if adj.is_sparse:

            row, col = adj.indices()

            diff = x[col] - x[row]

            cos = F.cosine_similarity(
                v_pred[row],
                diff,
                dim=-1
            )

            #loss = (1 - cos).mean()
            loss = ((1 - cos) ** 2).mean()

        else:

            diff = x.unsqueeze(1) - x.unsqueeze(0)

            cos = F.cosine_similarity(
                v_pred.unsqueeze(1),
                diff,
                dim=-1
            )

            loss = (adj * (1 - cos)).mean()

        return loss

    # --------------------------------------------------
    # 5 Total Loss
    # --------------------------------------------------
    def __call__(
        self,
        v_pred,
        v_kinetic=None,
        S=None,
        X=None,
        adj=None
    ):

        total = 0
        loss_dict = {}

        # velocity
        if S is not None:

            l_vel = self.velocity_loss(v_pred, S)

            total += self.w_velocity * l_vel

            loss_dict["velocity"] = l_vel.item()

        # kinetic
        if v_kinetic is not None:

            l_kin = self.kinetic_loss(
                v_pred,
                v_kinetic
            )

            total += self.w_kinetic * l_kin

            loss_dict["kinetic"] = l_kin.item()

        # smooth
        if adj is not None:

            l_smooth = self.smooth_loss(
                v_pred,
                adj
            )

            total += self.w_smooth * l_smooth

            loss_dict["smooth"] = l_smooth.item()

        # direction
        if X is not None and adj is not None:

            l_dir = self.direction_loss(
                v_pred,
                X,
                adj
            )

            total += self.w_direction * l_dir

            loss_dict["direction"] = l_dir.item()

        loss_dict["total"] = total.item()

        return total, loss_dict
