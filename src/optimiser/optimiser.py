"""
Base class for optimisers
"""

from abc import ABC, abstractmethod
from typing import Any
import torch
import numpy as np

from ..ri_measurement_operator.pysrc.measOperator import MeasOp


class Optimiser(ABC):
    """
    Base class for optimisers.

    This class provides a common interface for different optimisation algorithms.
    It is initialized with a measurement, a measurement operator, and an optional save path.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOp,
        save_pth: str = "results",
        file_prefix: str = "",
    ) -> None:
        """
        General initialization of the Optimiser.
        Prepare the measurement data, measurement operator.
        Generate and svae the dirty image and PSF.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOp): The measurement operator.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to None.
        """
        self._meas_op = meas_op
        try:
            self._meas = meas.to(self._meas_op.get_data_type_meas())
        except:
            self._meas = meas
        self._save_pth = save_pth
        self._file_prefix = file_prefix

        # common initialisation
        if type(self._meas_op.get_device()) == list:
            device = self._meas_op.get_device()[0]
        else:
            device = self._meas_op.get_device()
        self._meas_bp = torch.zeros(
            1,
            1,
            *self._meas_op.get_img_size(),
            device=device,
            dtype=self._meas_op.get_data_type()
        )
        self._model = torch.zeros_like(self._meas_bp)
        self._model_prev = torch.zeros_like(self._meas_bp)
        self._psf = torch.zeros_like(self._meas_bp)

    @abstractmethod
    def initialisation(self) -> Any:
        """
        Initalise parameters for specific parameters.
        This method should be called before running the optimiser.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def run(self) -> Any:
        """
        Runs the main loop of the optimiser.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_begin(self) -> Any:
        """
        This method should be called at the beginning of each iteration.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _stop_criteria(self) -> Any:
        """
        Checks the stop criteria.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_end(self) -> Any:
        """
        This method should be called at the end of each iteration.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def finalisation(self) -> Any:
        """
        This method should be called after the optimiser's main loop finishes.
        It can be used to save the final results, print final log etc.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    def get_model_image(self, dtype=torch.double) -> np.ndarray:
        """
        Get the final model image.

        args:
            dtype (torch.dtype): The data type of the output model image.

        Returns:
            np.ndarray: The final model.
        """
        return self._model.squeeze().cpu().to(dtype).numpy()

    def get_psf(self, dtype=torch.double) -> np.ndarray:
        """
        Get the PSF.

        args:
            dtype (torch.dtype): The data type of the output PSF.

        Returns:
            np.ndarray: The PSF.
        """
        return self._psf.squeeze().cpu().to(dtype).numpy()

    def get_dirty_image(self, dtype=torch.double) -> np.ndarray:
        """
        Get the dirty image.

        args:
            dtype (torch.dtype): The data type of the output dirty image.

        Returns:
            np.ndarray: The dirty image.
        """
        return self._meas_bp.squeeze().cpu().to(dtype).numpy()

    def get_residual_image(self, dtype=torch.double) -> np.ndarray:
        """
        Get the residual image.

        args:
            dtype (torch.dtype): The data type of the output residual image.

        Returns:
            np.ndarray: The residual image.
        """
        return (
            (
                self._meas_bp
                - self._meas_op.adjoint_op(self._meas_op.forward_op(self._model))
            )
            .squeeze()
            .cpu()
            .to(dtype)
            .numpy()
        )
