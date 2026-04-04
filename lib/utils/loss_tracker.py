import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class LossTracker:
    """
    Minimal loss tracker to log per-epoch metrics, print summaries, optionally plot and save to disk.
    Designed to be lightweight and have no hard dependency on matplotlib (plotting is optional).
    """

    def __init__(self, save_dir: str, plot_interval: int = 5) -> None:
        self.save_dir = save_dir
        self.plot_interval = max(1, int(plot_interval))
        os.makedirs(self.save_dir, exist_ok=True)

        # epoch -> {metric: value}
        self.epoch_losses: Dict[int, Dict[str, float]] = {}
        # temporary storage for the current epoch accumulation
        self._current_epoch_values: Dict[str, List[float]] = defaultdict(list)

    # --- write API ---
    def update(self, loss_dict: Dict[str, float]) -> None:
        """Accumulate loss values during an epoch (call multiple times per epoch)."""
        for k, v in loss_dict.items():
            if v is None:
                continue
            try:
                self._current_epoch_values[k].append(float(v))
            except Exception:
                # ignore non-numeric entries
                pass

    def end_epoch(self, epoch: int) -> None:
        """Finalize epoch by averaging accumulated values."""
        if not self._current_epoch_values:
            # nothing was logged; keep previous value if exists
            if epoch not in self.epoch_losses:
                self.epoch_losses[epoch] = {}
            return

        averaged = {k: (sum(vs) / max(1, len(vs))) for k, vs in self._current_epoch_values.items()}
        # also compute a total loss if not provided
        if 'loss' not in averaged and len(averaged) > 0:
            averaged['loss'] = sum(averaged.values())

        self.epoch_losses[epoch] = averaged
        self._current_epoch_values.clear()

    def save_data(self, filepath: Optional[str] = None) -> str:
        """Save epoch losses to JSON; returns the saved path."""
        if filepath is None:
            filepath = os.path.join(self.save_dir, 'loss_data.json')
        with open(filepath, 'w') as f:
            json.dump(self.epoch_losses, f, indent=2)
        return filepath

    # --- read/print API ---
    def should_print(self, epoch: int) -> bool:
        return (epoch % self.plot_interval == 0) or (epoch == 0)

    def print_epoch_summary(self, epoch: int, detailed: bool = False) -> None:
        vals = self.epoch_losses.get(epoch, {})
        if not vals:
            print(f"Epoch {epoch}: no logged losses")
            return
        if detailed:
            items = '  '.join([f"{k}: {v:.6f}" for k, v in sorted(vals.items())])
            print(f"Epoch {epoch} summary -> {items}")
        else:
            loss_val = vals.get('loss', None)
            if loss_val is not None:
                print(f"Epoch {epoch} loss: {loss_val:.6f}")
            else:
                # fallback to print the first metric
                k, v = next(iter(vals.items()))
                print(f"Epoch {epoch} {k}: {v:.6f}")

    def plot_losses(self, epoch: Optional[int] = None, save_path: Optional[str] = None) -> Optional[str]:
        """
        Try to plot curves for metrics across epochs. If matplotlib isn't available, skip gracefully.
        Returns the saved path if a file is written, otherwise None.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # headless backend
            import matplotlib.pyplot as plt
        except Exception:
            # plotting not available
            return None

        if not self.epoch_losses:
            return None

        # prepare data
        epochs = sorted(self.epoch_losses.keys())
        # collect all metric keys
        keys = set()
        for e in epochs:
            keys.update(self.epoch_losses[e].keys())
        keys = sorted(keys)

        # build series
        series = {k: [] for k in keys}
        for e in epochs:
            vals = self.epoch_losses.get(e, {})
            for k in keys:
                series[k].append(vals.get(k, float('nan')))

        # plot
        plt.figure(figsize=(10, 6))
        for k in keys:
            plt.plot(epochs, series[k], label=k)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.save_dir, f'loss_curves_epoch_{epochs[-1]}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def get_best_epoch(self, metric: str = 'loss') -> Tuple[Optional[int], Optional[float]]:
        """Return the epoch with minimal value for given metric."""
        best_epoch: Optional[int] = None
        best_val: Optional[float] = None
        for e, vals in self.epoch_losses.items():
            if metric in vals:
                v = vals[metric]
                if best_val is None or v < best_val:
                    best_val = v
                    best_epoch = e
        return best_epoch, best_val
