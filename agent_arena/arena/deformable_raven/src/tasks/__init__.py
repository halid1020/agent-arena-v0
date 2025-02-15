from arena.deformable_raven.src.tasks.task import Task
from arena.deformable_raven.src.tasks.sorting import Sorting
from arena.deformable_raven.src.tasks.insertion import Insertion
from arena.deformable_raven.src.tasks.insertion import InsertionTranslation
from arena.deformable_raven.src.tasks.hanoi import Hanoi
from arena.deformable_raven.src.tasks.aligning import Aligning
from arena.deformable_raven.src.tasks.stacking import Stacking
from arena.deformable_raven.src.tasks.sweeping import Sweeping
from arena.deformable_raven.src.tasks.pushing import Pushing
from arena.deformable_raven.src.tasks.palletizing import Palletizing
from arena.deformable_raven.src.tasks.kitting import Kitting
from arena.deformable_raven.src.tasks.packing import Packing
from arena.deformable_raven.src.tasks.cable import Cable

# New customized environments. When adding these envs, double check:
#   Environment._is_new_cable_env()
#   Environment._is_cloth_env()
#   Environment._is_bag_env()
# and adjust those methods as needed.

from arena.deformable_raven.src.tasks.insertion_goal import InsertionGoal
from arena.deformable_raven.src.tasks.defs_cables import (
        CableShape, CableShapeNoTarget, CableLineNoTarget,
        CableRing, CableRingNoTarget)
from arena.deformable_raven.src.tasks.defs_cloth import (
        ClothFlat, ClothFlatNoTarget, ClothCover)
from arena.deformable_raven.src.tasks.defs_bags import (
        BagAloneOpen, BagItemsEasy, BagItemsHard, BagColorGoal)

names = {'sorting':             Sorting,
         'insertion':           Insertion,
         'insertion-translation': InsertionTranslation,
         'hanoi':               Hanoi,
         'aligning':            Aligning,
         'stacking':            Stacking,
         'sweeping':            Sweeping,
         'pushing':             Pushing,
         'palletizing':         Palletizing,
         'kitting':             Kitting,
         'packing':             Packing,
         'cable':               Cable,
         'insertion-goal':      InsertionGoal, # start of custom envs
         'cable-shape':         CableShape,
         'cable-shape-notarget': CableShapeNoTarget,
         'cable-line-notarget': CableLineNoTarget,
         'cable-ring':          CableRing,
         'cable-ring-notarget': CableRingNoTarget,
         'cloth-flat':          ClothFlat,
         'cloth-flat-notarget': ClothFlatNoTarget,
         'cloth-cover':         ClothCover,
         'bag-alone-open':      BagAloneOpen,
         'bag-items-easy':      BagItemsEasy,
         'bag-items-hard':      BagItemsHard,
         'bag-color-goal':      BagColorGoal,
}
