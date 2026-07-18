import torch


def infer_velocity(model, S, U, cond, device):

    model.eval()

    S = S.to(device)
    U = U.to(device)
    cond = cond.to(device)

    with torch.no_grad():

        v_pred, _ = model(S, U, cond)

    return v_pred.cpu().numpy()