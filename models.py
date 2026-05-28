"""
Lightning training wrapper with deep supervision for HRM.

Deep supervision (HRM paper Sec. 2): multiple forward passes ("segments"),
each computing loss and backpropagating.  The carry is detached between
segments, creating a 1-step approximation of the recursive deep supervision
gradient.  This provides more frequent feedback to the H-module and acts
as a regularization mechanism.
"""
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

import config
from modules import HRM, HRMCarry


class Leakless(LightningModule):
    """CMB E->B leakage cleaner using Hierarchical Reasoning Model.

    Each training step runs *n_supervision* forward segments.  All segment
    losses are accumulated and backpropagated together (a practical
    approximation of the per-segment updates described in the paper that
    integrates cleanly with Lightning and gradient clipping).

    At inference, predict() runs all supervision segments and returns the
    output from the final segment.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Training config
        self.lr = kwargs.get('lr', config.LEARNING_RATE)
        self.weight_decay = kwargs.get('weight_decay', config.WEIGHT_DECAY)
        self.n_supervision = kwargs.get('n_supervision', config.N_SUPERVISION)
        self.loss_fn = kwargs.get('loss', F.mse_loss)
        self.loss_kwargs = kwargs.get('loss_kwargs', {})
        self.scheduler_kwargs = kwargs.get('scheduler_kwargs', {
            'factor': 0.25, 'patience': 4
        })
        self.optimizer_cls = kwargs.get('optimizer', torch.optim.AdamW)
        self.optimizer_kwargs = kwargs.get('optimizer_kwargs', {})

        # HRM core
        self.net = HRM(
            input_dim=kwargs.get('input_dim', config.INPUT_DIM),
            hidden_size=kwargs.get('hidden_size', config.HIDDEN_SIZE),
            H_layers=kwargs.get('H_layers', config.H_LAYERS),
            L_layers=kwargs.get('L_layers', config.L_LAYERS),
            H_cycles=kwargs.get('H_cycles', config.H_CYCLES),
            L_cycles=kwargs.get('L_cycles', config.L_CYCLES),
            num_heads=kwargs.get('num_heads', config.NUM_HEADS),
            expansion=kwargs.get('expansion', config.EXPANSION),
            max_seq_len=kwargs.get('max_seq_len', config.MAX_SEQ_LEN),
            rope_theta=kwargs.get('rope_theta', config.ROPE_THETA),
        )

        hparams = dict(kwargs)
        hparams.pop('loss', None)
        self.save_hyperparameters(hparams)

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, carry=None):
        param_dtype = next(self.net.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)
        return self.net(x, carry)

    # ── deep supervision steps ────────────────────────────────────────

    def _run_segments(self, x, n_segments):
        """Run *n_segments* forward passes, detaching carry between each."""
        carry = None
        output = None
        for _ in range(n_segments):
            output, carry = self.forward(x, carry)
            carry = HRMCarry(z_H=carry.z_H.detach(), z_L=carry.z_L.detach())
        return output, carry

    def training_step(self, batch, batch_idx):
        x, y = batch
        carry = None
        total_loss = 0.0

        for _ in range(self.n_supervision):
            y_hat, carry = self.forward(x, carry)
            target = y.to(dtype=y_hat.dtype).view_as(y_hat)
            total_loss = total_loss + self.loss_fn(y_hat, target, **self.loss_kwargs)
            carry = HRMCarry(z_H=carry.z_H.detach(), z_L=carry.z_L.detach())

        avg_loss = total_loss / self.n_supervision
        self.log('train_loss', avg_loss, sync_dist=True, prog_bar=True,
                 logger=True, on_epoch=True, on_step=config.ON_STEP)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self._run_segments(x, self.n_supervision)
        loss = self.loss_fn(y_hat, y.to(dtype=y_hat.dtype).view_as(y_hat))
        self.log('val_loss', loss, sync_dist=True, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self._run_segments(x, self.n_supervision)
        loss = self.loss_fn(y_hat, y.to(dtype=y_hat.dtype).view_as(y_hat))
        self.log('test_loss', loss, sync_dist=True, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return loss

    # ── optimizer ─────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt_kwargs = {
            'params': self.parameters(),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        opt_kwargs.update(self.optimizer_kwargs)
        optimizer = self.optimizer_cls(**opt_kwargs)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_kwargs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, X, n_supervision=None):
        """Run all deep-supervision segments and return the final output.

        This replaces the PredictorMixin.predict() so that inference
        always uses the full reasoning chain.

        Args:
            X: Input tensor (B, N, input_dim).
            n_supervision: Override number of segments (for inference-time
                scaling -- see HRM paper Sec. 2, Fig. 5c).
        """
        n = n_supervision or self.n_supervision
        self.eval()
        with torch.no_grad():
            output, _ = self._run_segments(X, n)
        return output
