"""
Unconstrained AIRI, built on forward-backward algorithm
"""

import os
from typing import Union
import torch
import numpy as np
from astropy.io import fits

from .forward_backward import ForwardBackward
from ..prox_operator import ProxOpAIRI
from ..ri_measurement_operator.pysrc.measOperator import MeasOpNUFFT
from ..ri_measurement_operator.pysrc.measOperator.meas_op_PSF import (
    MeasOpPSF,
)


class FBAIRI(ForwardBackward):
    """
    FBAIRI implements the AIRI algorithm based on Forward-Backward.

    It handles data fidelity in the forward step and uses AIRI denoisers
    in the backward step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOpNUFFT,
        prox_op: ProxOpAIRI,
        use_ROP: bool = False,
        meas_op_approx: Union[MeasOpPSF, None] = None,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: Union[float, None] = None,
        heu_noise_scale: float = 1.0,
        new_heu: bool = False,
        adapt_net_select: bool = True,
        peak_tol_min: float = 1e-3,
        peak_tol_max: float = 0.1,
        peak_tol_step: float = 0.1,
        verbose: bool = True,
        save_pth: str = "results",
        file_prefix: str = "",
        iter_save: int = 1000,
    ) -> None:
        """
        Initializes the AIRI optimiser, which is built on forward-backward algorithm.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOpTkbnRI): The measurement operator.
            prox_op (ProxOpAIRI): The proximal operator.
            meas_op_approx (Union[MeasOpPSF, None], optional): The approximate measurement
                operator build on oversampled PSF. The AIRI algorithm will use this operator
                for reconstruction if it is not None. Defaults to None.
            im_min_itr (int, optional): The minimum number of iterations. Defaults to 100.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            im_var_tol (float, optional): The tolerance for image variation. Defaults to 1e-4.
            im_peak_est (float, optional): The estimated peak of the image. Defaults to None.
            heu_noise_scale (float, optional): The heuristic noise scale. Defaults to 1.0.
            adapt_net_select (bool, optional): Whether to adapt network selection. Defaults to True.
            peak_tol_min (float, optional): The minimum tolerance for peak. Defaults to 1e-3.
            peak_tol_max (float, optional): The maximum tolerance for peak. Defaults to 0.1.
            peak_tol_step (float, optional): The step size for peak tolerance. Defaults to 0.1.
            verbose (bool, optional): Whether to print verbose logs. Defaults to True.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            iter_save (int, optional): The number of iterations after which to save results.
                Defaults to 1000.
        """
        super().__init__(
            meas,
            meas_op if meas_op_approx is None else meas_op_approx,
            meas_op,
            prox_op,
            im_max_itr=im_max_itr,
            new_heu=new_heu,
            save_pth=save_pth,
            file_prefix=file_prefix,
        )

        self._use_ROP = use_ROP
        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._im_peak_est = im_peak_est
        self._heu_noise_scale = heu_noise_scale
        self._adapt_net_select = adapt_net_select
        self._peak_tol_min = peak_tol_min
        self._peak_tol = peak_tol_max
        self._peak_tol_step = peak_tol_step
        self._peak_val_range = []
        self._verbose = verbose
        self._new_heu = new_heu
        self._iter_save = iter_save

        self._prev_peak_val = 1.0
        self._heuristic = 1.0
        self._im_rel_var = 1.0

    def initialisation(self) -> None:
        """
        Initialises specific parameters of AIRI.

        This method initialises the step size, estimates the peak value, sets the
        heuristic noise level, initialises the AIRI proximal operator, and prints
        some information if verbose is True.
        """
        # step size
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # estimate peak value
        if self._im_peak_est is None:
            self._prev_peak_val = self._meas_bp.max().item() / self._psf_peak
            if self._verbose:
                print(
                    "INFO: use normalised dirty peak as estimated",
                    f"image peak value: {self._prev_peak_val}",
                    flush=True,
                )
        else:
            self._prev_peak_val = self._im_peak_est
            if self._verbose:
                print(
                    f"\nINFO: user specified the estimated image peak value: {self._prev_peak_val}",
                    flush=True,
                )

        # heuristic noise level
        # self._heuristic = 1 / np.sqrt(2 * self._meas_op_precise.get_op_norm())
        if self._new_heu:
            noise = (torch.randn_like(self._meas, dtype=self._meas.dtype, device=self._meas.device) + 1j * torch.randn_like(self._meas, dtype=self._meas.dtype, device=self._meas.device)) / np.sqrt(2)
            self._heuristic = (self._meas_op.adjoint_op(noise) / self._meas_op.get_psf().max()).std().item() / self._meas_bp.max().item()
            if self._verbose:
                print(
                    f"INFO: using Sally's new heuristic",
                    flush=True,
                )
        else:
            self._heuristic = 1 / np.sqrt(2 * self._meas_op_precise.get_op_norm())
            if self._verbose:
                print(
                    f"INFO: measurement operator norm {self._meas_op_precise.get_op_norm()}",
                    flush=True,
                )
        if self._verbose:
            print(
                f"INFO: measurement operator norm {self._meas_op_precise.get_op_norm()}",
                flush=True,
            )
            print(f"INFO: heuristic noise level: {self._heuristic}", flush=True)
        if not self._use_ROP:
            heu_corr_factor = np.sqrt(
                self._meas_op_precise.get_op_norm_prime()
                / self._meas_op_precise.get_op_norm()
            )
        else:
            heu_corr_factor = 1.0
        if not np.isclose(heu_corr_factor, 1.0) and not self._use_ROP:
            self._heuristic *= heu_corr_factor
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after correction: {self._heuristic},",
                    f"corection factor {heu_corr_factor}",
                    flush=True,
                )
        if not np.isclose(self._heu_noise_scale, 1.0):
            self._heuristic *= self._heu_noise_scale
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after scaling: {self._heuristic},",
                    f"scaling factor {self._heu_noise_scale}",
                    flush=True,
                )

        # initialise AIRI prox
        self._peak_val_range = self._prox_op.update(
            self._heuristic, self._prev_peak_val
        )

        if self._verbose:
            print(f"INFO: forward-backward step size: {self._gd_step_size}", flush=True)
            if isinstance(self._meas_op, MeasOpPSF):
                print(
                    "INFO: use approximate measurement operator for reconstruction",
                    flush=True,
                )
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:    AIRI   *********", flush=True)
            print("*************************************************", flush=True)

    def _each_iter_begin(self):
        """
        Method to be executed at the beginning of each iteration.

        This method is empty for AIRI.
        """

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """
        Checks the stop criteria for the algorithm.

        This method calculates the image relative variation, logs some information if
        `verbose` is True, and checks if the stop criteria for the algorithm are met.

        Returns:
            bool: True if the stop criteria are met, False otherwise.
        """
        # img relative variation
        self._im_rel_var = torch.linalg.vector_norm(self._model - self._model_prev) / (
            torch.linalg.vector_norm(self._model) + 1e-10
        )

        # log
        if self._verbose:
            print(
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var}",
                f"\ntimings: gradient step {self._t_forward} sec,",
                f"denoising step {self._t_backward} sec,"
                f"iteration {self._t_iter} sec.",
                flush=True,
            )

        # stop criteria
        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """
        Method to be executed at the end of each iteration.

        This method saves intermediate results, and updates the AIRI denoiser selection
        if `adapt_net_select` is True.
        """
        # save intermediate results
        if (self._iter + 1) % self._iter_save == 0:
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix
                    + "tmp_model_itr_"
                    + str(self._iter + 1)
                    + ".fits",
                ),
                self.get_model_image(dtype=torch.float32),
                overwrite=True,
            )
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix
                    + "tmp_residual_itr_"
                    + str(self._iter + 1)
                    + ".fits",
                ),
                self.get_residual_image(dtype=torch.float32) / self._psf_peak,
                overwrite=True,
            )

        # AIRI denoiser selection
        if self._adapt_net_select:
            curr_peak_val = self._model.max().item()
            peak_var = abs(curr_peak_val - self._prev_peak_val) / self._prev_peak_val
            if self._verbose:
                print(
                    f"  Model image peak value {curr_peak_val}, relative variation = {peak_var}",
                    flush=True,
                )

            if peak_var < self._peak_tol and (
                curr_peak_val < self._peak_val_range[0]
                or curr_peak_val > self._peak_val_range[1]
            ):
                if self._verbose and self._new_heu:
                    print(
                        f"  Current heuristic: {self._heuristic}",
                        flush=True,
                    )
                self._peak_val_range = self._prox_op.update(
                    self._heuristic, self._prev_peak_val
                )
            self._prev_peak_val = curr_peak_val

    @torch.no_grad()
    def finalisation(self) -> None:
        """
        Finalises the FBAIRI optimiser.

        This method prints final log information if `verbose` is True, and saves
        the result model and residual images.
        """
        if self._verbose:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total} sec,",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # save final results
        fits.writeto(
            os.path.join(self._save_pth, self._file_prefix + "model_image.fits"),
            self.get_model_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth, self._file_prefix + "residual_dirty_image.fits"
            ),
            self.get_residual_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth,
                self._file_prefix + "normalised_residual_dirty_image.fits",
            ),
            self.get_residual_image() / self._psf_peak,
            overwrite=True,
        )
