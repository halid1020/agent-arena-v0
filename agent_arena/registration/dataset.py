from agent_arena.utilities.datasets.cloth_flatten_shelve_dataset import *
from agent_arena.utilities.datasets.cloth_hd5f_dataset import *
from agent_arena.utilities.datasets.cloth_hd5f_dataset_v2 import *
from agent_arena.utilities.datasets.cloth_vision_pick_and_place_hd5f_dataset import *
from agent_arena.utilities.datasets.transporter.cloth_hd5f_dataset import *
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset

name_to_dataset = {
    'default': TrajectoryDataset,
    'fabric-pick-and-place': ClothVisionPickAndPlaceHDF5Dataset,
    'mono-square-fabric-pick-and-place-transporter': TransporterCltohHd5fDataset,
}