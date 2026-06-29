from actoris_harena.utilities.datasets.cloth_flatten_shelve_dataset import *
from actoris_harena.utilities.datasets.cloth_hd5f_dataset import *
from actoris_harena.utilities.datasets.cloth_hd5f_dataset_v2 import *
from actoris_harena.utilities.datasets.cloth_vision_pick_and_place_hd5f_dataset import *
from actoris_harena.utilities.datasets.transporter.cloth_hd5f_dataset import *
from actoris_harena.data.static_trajectory_dataset import StaticTrajectoryDataset

name_to_dataset = {
    'default': StaticTrajectoryDataset,
    'fabric-pick-and-place': ClothVisionPickAndPlaceHDF5Dataset,
    'mono-square-fabric-pick-and-place-transporter': TransporterCltohHd5fDataset,
}