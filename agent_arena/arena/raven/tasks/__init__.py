# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Ravens tasks."""

from ...raven.tasks.align_box_corner import AlignBoxCorner
from ...raven.tasks.assembling_kits import AssemblingKits
from ...raven.tasks.assembling_kits import AssemblingKitsEasy
from ...raven.tasks.block_insertion import BlockInsertion
from ...raven.tasks.block_insertion import BlockInsertionEasy
from ...raven.tasks.block_insertion import BlockInsertionNoFixture
from ...raven.tasks.block_insertion import BlockInsertionSixDof
from ...raven.tasks.block_insertion import BlockInsertionTranslation
from ...raven.tasks.manipulating_rope import ManipulatingRope
from ...raven.tasks.packing_boxes import PackingBoxes
from ...raven.tasks.palletizing_boxes import PalletizingBoxes
from ...raven.tasks.place_red_in_green import PlaceRedInGreen
from ...raven.tasks.stack_block_pyramid import StackBlockPyramid
from ...raven.tasks.sweeping_piles import SweepingPiles
from ...raven.tasks.task import Task
from ...raven.tasks.towers_of_hanoi import TowersOfHanoi

names = {
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi
}
