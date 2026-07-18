"""Training routine for TransVelo.

Provides :func:`train_model`, imported by ``main.py`` and
``downstream_main.py``. The loop mirrors the reference implementation:
velocity + kinetic-consistency losses with gradient clipping at 1.0.
"""
import torch


def train_model(model, loader, optimizer, epochs=100, device="cpu", loss_fn=None):
    """Train a TransVelo model.

    Parameters
    ----------
    model : torch.nn.Module
        A :class:`~model.TransVelo` instance. When a ``KineticLayer`` is
        attached it returns ``(v_pred, v_kin)``.
    loader : torch.utils.data.DataLoader
        Yields dicts with keys ``"S"``, ``"U"`` and ``"cond"``.
    optimizer : torch.optim.Optimizer
        Optimizer (e.g. ``torch.optim.AdamW``).
    epochs : int
        Number of training epochs.
    device : str
        ``"cuda"`` or ``"cpu"``.
    loss_fn : callable, optional
        A :class:`~loss.TransVeloLoss` instance. If ``None`` a default
        weighted loss is built.

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    if loss_fn is None:
        from loss import TransVeloLoss

        loss_fn = TransVeloLoss(
            w_velocity=1.0,
            w_kinetic=0.5,
            w_smooth=0.2,
            w_direction=0.1,
        )

    model.train()
    for epoch in range(epochs):
        total_loss = total_vel = total_kin = 0.0
        n_batches = 0

        for batch in loader:
            S_batch = batch["S"].to(device)
            U_batch = batch["U"].to(device)
            cond_batch = batch["cond"].to(device)

            v_pred, v_kin = model(S_batch, U_batch, cond_batch)

            loss, loss_dict = loss_fn(
                v_pred=v_pred,
                v_kinetic=v_kin,
                S=S_batch,
                X=S_batch,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_vel += loss_dict.get("velocity", 0.0)
            total_kin += loss_dict.get("kinetic", 0.0)
            n_batches += 1

        print(
            f"Epoch {epoch + 1} | "
            f"Loss {total_loss / n_batches:.3f} | "
            f"vel {total_vel / n_batches:.3f} | "
            f"kin {total_kin / n_batches:.3f}"
        )

    return model
