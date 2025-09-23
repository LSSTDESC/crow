"""Module for defining the ClusterRecipe class."""

from abc import ABC, abstractmethod

from clump.binning import SaccBin
from clump.properties import ClusterProperty
from firecrown.updatable import Updatable, UpdatableCollection


class ClusterRecipe(Updatable, ABC):
    """Abstract class defining a cluster recipe.

    A cluster recipe is a combination of different cluster theoretrical predictions
    and models that produces a single prediction for an observable.
    """

    def __init__(self, parameter_prefix: None | str = None) -> None:
        super().__init__(parameter_prefix)
        self.my_updatables: UpdatableCollection = UpdatableCollection()

