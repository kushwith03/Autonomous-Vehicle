from torch.utils.data import ConcatDataset
from .carla_dataset import CarlaDataset
from .cityscapes_dataset import CityscapesDataset

class CombinedDataset(ConcatDataset):
    def __init__(self, carla_ds, cityscapes_ds):
        super().__init__([carla_ds, cityscapes_ds])
        self.carla_ds = carla_ds
        self.cityscapes_ds = cityscapes_ds

def get_dataset_stats(dataset):
    """
    Returns total count, carla count, cityscapes count
    """
    if isinstance(dataset, CombinedDataset):
        return {
            "total": len(dataset),
            "carla": len(dataset.carla_ds),
            "cityscapes": len(dataset.cityscapes_ds)
        }
    return {"total": len(dataset), "carla": 0, "cityscapes": 0}
