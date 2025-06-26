import numpy as np
import torch

# from ..meas_operator import MeasOpTkbnRI
from scipy.optimize import fsolve


def expo_im(x, a) -> torch.tensor:
    r"""
    Exponentiate the image of interest by factor :math:`a` defined as

    .. math::
    x_{\text{exp}} = \frac{a^{x} - 1}{a},

    where :math:`x` is the image of interest and :math:`x_{\text{exp}}` is the exponentiated image.

    :param x: image of interest
    :type x: torch.Tensor
    :param a: exponentiation factor
    :type a: float

    :return: exponentiated image
    :rtype: torch.Tensor
    """
    return (a**x - 1) / a


def solve_expo_factor(sigma_0, sigma) -> float:
    r"""
    Compute exponentation factor :math:`a` to exponentiate the image of interest
    from current dynamic range :math:`1/\sigma_0` to target dynamic range :math:`1/\sigma` by solving

    .. math::
    a = (1 + a \sigma)^{\sigma_0^{-1}}.

    :param sigma_0: inverse of current dynamic range of the image of interest
    :type sigma_0: float
    :param sigma: inverse of target dynamic range.
    :type sigma: float

    :return: exponentiation factor.
    :rtype: float
    """
    fun = lambda a: (1 + a * sigma) ** (1 / sigma_0) - a

    est_c = sigma ** -(1 / (1 / sigma_0 - 1))
    est_a = (est_c - 1) / sigma

    res = fsolve(fun, est_a)
    obj = fun(res)

    if obj > 1e-7 or res < 40:
        print(f"Possible wrong solution. sigma = {sigma}, a = {res[0]}, f(a) = {obj[0]}")
    return res[0]


def compute_tau(
    op,
    data: dict,
    sigma_range_min: float = 1e-5,
    sigma_range_max: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    assert sigma_range_min > 0 and sigma_range_max > 0, "sigma_range_min and sigma_range_max must be positive"
    assert sigma_range_min < sigma_range_max, "sigma_range_min must be smaller than sigma_range_max"
    np.random.seed(data["uv_id"])
    log_sigma = np.random.rand() * (np.log10(sigma_range_max) - np.log10(sigma_range_min)) + np.log10(
        sigma_range_min
    )
    sigma = 10**log_sigma
    print(f'sigma_range_min = {sigma_range_min:.4e}; sigma_range_max = {sigma_range_max:.4e}')
    print(f'uv_id: {data["uv_id"]}: log_sigma = {log_sigma:.4f}; sigma = {sigma:.4e}; dr = {1/sigma:.4f}')
    # sigma = torch.tensor(sigma).to(device)
    if "time" not in data:
        tau = compute_tau_instance(op, data, sigma)
    else:
        print("Time vector detected. tau will be computed for each time instance...")
        time = data["time"].squeeze()
        time_diff = time[1:] - time[:-1]
        new_start = torch.cat(
            [
                torch.tensor([0]).to(device),
                torch.where(abs(time_diff) > data["timeStep"])[0] + 1,
                torch.tensor([max(time.shape)]).to(device),
            ]
        )
        tau = torch.zeros(max(data["u"].shape), device=device)
        print(f"Number of time instances: {len(new_start) - 1}")
        for i in range(len(new_start) - 1):
            print(
                f"Computing tau for time instance {i + 1}/{len(new_start) - 1}: start: {new_start[i]}, end: {new_start[i + 1] - 1}"
            )
            u = data["u"][..., new_start[i] : new_start[i + 1]]
            v = data["v"][..., new_start[i] : new_start[i + 1]]
            data_instance = {"u": u, "v": v}
            if "nWimag" in data and data["nWimag"].numel() > 1:
                nWimag = data["nWimag"][..., new_start[i] : new_start[i + 1]].clone()
                data_instance.update({"nWimag": nWimag})
            if "nW" in data and data["nW"].numel() > 1:
                nW = data["nW"][..., new_start[i] : new_start[i + 1]].clone()
                data_instance.update({"nW": nW})
            tau_tmp = compute_tau_instance(op, data_instance, sigma)
            print(f"{i+1}: tau= {tau_tmp}")
            tau[new_start[i] : new_start[i + 1]] = tau_tmp

    # if len(tau.size()) == 0 or len(tau.shape) == 1:
    #     tau = tau.unsqueeze(0)
    # if len(sigma.size()) == 0:
    #     sigma = sigma.unsqueeze(0)
    return tau, sigma


def compute_tau_instance(op, data: dict, sigma: torch.tensor) -> tuple:
    #  briggs : bool = False) -> tuple:
    r"""
    Compute the standard deviation of the heuristic noise level in the measurement defined as

    .. math::
    \tau = \sigma \sqrt{2 \|\Phi\|_S},

    where :math:`\sigma` is the standard deviation of noise in the image domain
    and :math:`\|\Phi\|_S` is the spectral norm of the measurement operator :math:`\Phi`.

    :param op: measurement operator
    :type op: lib.operator.Operator
    :param data: dictionary containing uv points and imweight weighting from uv data file
    :type data: dict
    :param sigma_range_min: minimum value of :math:`\sigma`, defaults to 2e-6
    :type sigma_range_min: float, optional
    :param sigma_range_max: maximum value of :math:`\sigma`, defaults to 1e-3
    :type sigma_range_max: float, optional
    :param briggs: set briggs weighting, defaults to False
    :type briggs: bool, optional
    :param device: target device for computation of torch tensor, defaults to torch.device('cpu')
    :type device: torch.device, optional

    :return: tuple of :math:`\tau` and :math:`\sigma`, noise level in the data domain and standard deviation of noise in the image domain
    :rtype: tuple

    :raises AssertionError: if sigma_range_min or sigma_range_max is not positive or sigma_range_min is greater than sigma_range_max
    """
    # weight = torch.ones(data["u"].shape, dtype=data["nWimag"].dtype, device=data["u"].device)
    # # if "nWimag" in data:
    # weight *= data["nWimag"]
    # # else:
    # #     print("No nWimag found for computing tau.")
    # meas_op_i = MeasOpTkbnRI(
    #     u=data["u"],
    #     v=data["v"],
    #     natural_weight=torch.tensor([1.0], device=data["u"].device),
    #     image_weight=weight,
    #     img_size=op._img_size,
    #     real_flag=op._real_flag,
    #     grid_size=op._grid_size,
    #     num_points=op._num_points,
    #     mode=op._mode,
    #     device=op._device,
    #     dtype=op._dtype,
    # )
    # op_norm = meas_op_i.get_op_norm()
    op_norm = op.get_op_norm()
    print(f"Raw op_norm: {op_norm}")
    tau = sigma * np.sqrt(2 * op_norm)
    return tau


def gen_noise(op, data, sigma0, sigma_range_min, sigma_range_max, device=torch.device("cpu"), verbose=False):
    tau, sigma = compute_tau(op, data, sigma_range_min, sigma_range_max, device=device)
    if verbose:
        print(f"Target dynamic range: {1/sigma.item():.4f}")
    expo_factor = solve_expo_factor(sigma0, sigma)
    # data["gdth_linear"] = data["gdth"].clone()
    data["gdth"] = expo_im(data["gdth"], expo_factor)
    return data, tau, sigma
