import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import logging


class Simulator:
    def __init__(
        self,
        D: float = 5e-4,
        por: float = 0.29,
        rho_s: float = 2880,
        k_f: float = 3.5e-4,
        n_f: float = 0.874,
        sol: float = 1.0,
        t: float = 500,
        tdim: int = 101,
        x_left: float = 0.0,
        x_right: float = 1.0,
        xdim: int = 50,
        seed: int = 0,
    ):
        self.D = D
        self.por = por
        self.rho_s = rho_s
        self.k_f = k_f
        self.n_f = n_f
        self.sol = sol

        self.T = t
        self.X0 = x_left
        self.X1 = x_right

        self.Nx = xdim
        self.Nt = tdim

        self.dx = (self.X1 - self.X0) / (self.Nx)
        self.x = np.linspace(self.X0 + self.dx / 2, self.X1 - self.dx / 2, self.Nx)

        self.t = np.linspace(0, self.T, self.Nt)

        self.log = logging.getLogger(__name__)

        self.seed = seed
        self.generator = np.random.default_rng(self.seed)


    def generate_sample(self) -> np.ndarray:
        generator = np.random.default_rng(self.seed)
        u0 = np.ones(self.Nx) * generator.uniform(0, 0.2)

        main_diag = -2 * np.ones(self.Nx) / self.dx**2
        left_diag = np.ones(self.Nx - 1) / self.dx**2
        right_diag = np.ones(self.Nx - 1) / self.dx**2

        diagonals = [main_diag, left_diag, right_diag]
        offsets = [0, -1, 1]
        self.lap = diags(diagonals, offsets)

        self.rhs = np.zeros(self.Nx)

        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        sample_c = np.transpose(ode_data)
        sample_c = np.expand_dims(sample_c, axis=-1)

        # spatial_noise = self.generate_spatially_correlated_noise()
        # noisy_sample = sample_c + spatial_noise

        return sample_c


    # def generate_spatially_correlated_noise(self) -> np.ndarray:
    #     noise = self.generator.normal(0, 1, self.Nx)

    #     kernel_size = 5  
    #     smooth_noise = np.convolve(noise, np.ones(kernel_size) / kernel_size, mode="same")

    #     return np.expand_dims(smooth_noise, axis=-1)

    def rc_ode(self, t: float, y):
        left_BC = self.sol
        right_BC = (y[-2] - y[-1]) / self.dx * self.D

        retardation = 1 + (
            (1 - self.por) / self.por
        ) * self.rho_s * self.k_f * self.n_f * np.abs(y + 1e-6) ** (self.n_f - 1)

        self.rhs[0] = self.D / retardation[0] / (self.dx**2) * left_BC
        self.rhs[-1] = self.D / retardation[-1] / (self.dx**2) * right_BC

        spatial_noise = np.sin(2 * np.pi * self.x) * self.generator.uniform(
            -1, 1, self.Nx
        )
        time_dependent_noise = np.sin(2 * np.pi * t / self.T) * self.generator.uniform(
            -1, 1, self.Nx
        )

        return (
            self.D / retardation * (self.lap @ y)
            + self.rhs
            + spatial_noise
            + time_dependent_noise
        )