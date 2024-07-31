from torchmetrics import Metric
import torch

class CrossEntropyLossMetric(Metric):
    """
    Wrapper around the torch.nn.CrossEntropyLoss for use
    with torchmetrics.
    """

    def __init__(self):
        super().__init__()
        self.add_state("summed_loss", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_num", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")

        # Note, we use NLLLoss() and log() because we assume that our model
        # directly outputs the probability of contact, instead of probability logits. 
        self.loss_fn = torch.nn.NLLLoss(reduction='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.size(0) != target.size(0):
            raise ValueError("Both tensors must have the same number of batches.")
        self.summed_loss += self.loss_fn(torch.log(preds), target)
        self.total_num += preds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.summed_loss.float() / self.total_num