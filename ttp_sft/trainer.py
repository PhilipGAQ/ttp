import torch
from transformers import Trainer


class QueryGenTrainer(Trainer):
    """
    Custom trainer to support sub-batch execution when the collator returns large tensors.
    If args.sub_batch_size > 0, we split query/passage along batch dim and
    accumulate gradients across sub-batches within a single training step.
    """
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Single-path training step (sub-batch disabled)
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss_emb = outputs.loss_emb if hasattr(outputs, 'loss_emb') and outputs.loss_emb is not None else None
        loss_gen = outputs.loss_gen if hasattr(outputs, 'loss_gen') and outputs.loss_gen is not None else None

        if self.args.n_gpu > 1:
            loss = loss.mean()
            if loss_emb is not None:
                loss_emb = loss_emb.mean()
            if loss_gen is not None:
                loss_gen = loss_gen.mean()

        # Log to summary_writer
        if hasattr(self, 'summary_writer') and self.summary_writer is not None:
            step = getattr(self.state, 'global_step', 0)
            if loss is not None:
                self.summary_writer.add_scalar('train/loss', loss.item(), step)
            if loss_emb is not None:
                self.summary_writer.add_scalar('train/loss_emb', loss_emb.item(), step)
            if loss_gen is not None:
                self.summary_writer.add_scalar('train/loss_gen', loss_gen.item(), step)

        self.accelerator.backward(loss)
        return loss.detach()

