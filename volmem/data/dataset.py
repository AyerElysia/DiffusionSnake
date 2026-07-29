from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Volume-grouped slice sequence dataset scaffold.

    This class deliberately cannot load legacy slice-stacking catalogs.
    A dedicated volume-level manifest is required before implementation.
    """

    maturity = "scaffold"

    def __init__(self, manifest_file, split, **kwargs):
        del kwargs
        raise RuntimeError(
            "VolMemDataset is a scaffold. Create a volume-level manifest with "
            "volume_id, slice_index and slice_position_mm before use. "
            "Legacy pseudo-3D slice manifests are not accepted."
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)
