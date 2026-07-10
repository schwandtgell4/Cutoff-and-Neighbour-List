#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_torch.py

PyTorch version of the Lennard-Jones gas MD code.

The code keeps the public structure of the original NumPy implementation, but
uses PyTorch tensors after initialization so that the expensive pair-distance,
force, integration, and energy operations can run on

- NVIDIA GPUs via CUDA,
- Apple Silicon GPUs via MPS,
- CPU as fallback.
"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import torch
import numpy as np
from scipy.constants import R, Avogadro


#----------------------------------------------------------------
#   H E L P E R S
#----------------------------------------------------------------
def get_torch_device():
    """
    Select the best available compute device.

    Priority:
    1. NVIDIA GPU via CUDA
    2. Apple Silicon GPU via MPS
    3. CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device):
    """
    Synchronize GPU work before timing.

    CUDA and MPS can execute asynchronously. Synchronization makes the measured
    time closer to the actual compute time.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def tensor_to_numpy(x):
    """Convert a torch tensor to a NumPy array, or return NumPy input unchanged."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def tensor_to_float(x):
    """Convert a scalar torch tensor/NumPy scalar/Python number to float."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


#----------------------------------------------------------------
#   C L A S S E S
#----------------------------------------------------------------
class ParticleSystem:
    def __init__(self, n_particles):
        self.n = n_particles

        # Properties for each particle. During initialization these are NumPy arrays.
        # After move_particle_system_to_torch(), they become torch tensors.
        self.mass = np.zeros(n_particles)
        self.sigma = np.zeros(n_particles)
        self.epsilon = np.zeros(n_particles)

        # 3D positions, velocities, forces, and random numbers (shape: n_particles x 3)
        self.position = np.zeros((n_particles, 3))
        self.velocity = np.zeros((n_particles, 3))
        self.force = np.zeros((n_particles, 3))
        self.random_number = np.zeros((n_particles, 3))

    def set_parameters(self, i, mass, sigma, epsilon):
        """Set the parameters of the i-th particle."""
        self.mass[i] = mass
        self.sigma[i] = sigma
        self.epsilon[i] = epsilon

    def set_position(self, i, position):
        """Set the position of the i-th particle."""
        self.position[i] = position

    def set_velocity(self, i, velocity):
        """Set the velocity of the i-th particle."""
        self.velocity[i] = velocity

    def set_force(self, i, force):
        """Set the force of the i-th particle."""
        self.force[i] = force

    def set_random_number(self, i, random_number):
        """Set the random number vector of the i-th particle."""
        self.random_number[i] = random_number

    def __repr__(self):
        return f"<ParticleSystem with {self.n} particles>"


class SimulationParameters:
    def __init__(self, dt, n_steps, temperature, box_length,
                 tau_thermostat=None, rij_min=0.0,
                 r_cut=None, use_cutoff=False):
        """
        Parameters:
            dt (float): Time step in ps.
            n_steps (int): Number of time steps.
            temperature (float): Temperature in K.
            box_length (float): Length of the cubic simulation box in nm.
            tau_thermostat (float or None): Thermostat coupling time in ps.
            rij_min (float): Lower cutoff/minimum distance in nm for numerical stability.
            r_cut (float or None): Upper cutoff radius in nm.
            use_cutoff (bool): If True, ignore LJ interactions with r > r_cut.
        """
        self.dt = dt
        self.n_steps = n_steps
        self.temperature = temperature
        self.box_length = box_length
        self.tau_thermostat = tau_thermostat
        self.rij_min = rij_min
        self.r_cut = r_cut
        self.use_cutoff = use_cutoff

        if self.use_cutoff:
            if self.r_cut is None:
                raise ValueError("use_cutoff=True requires r_cut to be set.")

            # For periodic boundary conditions, r_cut must be <= L/2.
            # Otherwise, one particle could interact with more than one periodic image
            # of the same other particle. This code uses the minimum-image convention,
            # so only the nearest image is considered.
            if self.r_cut > 0.5 * self.box_length:
                raise ValueError("r_cut is set larger than half the box length.")

        # Optional friction coefficient for Langevin dynamics.
        self.xi = None
        if self.tau_thermostat and self.tau_thermostat > 0.0:
            self.xi = 1 / self.tau_thermostat


#----------------------------------------------------------------
#   C O N V E R S I O N
#----------------------------------------------------------------
def move_particle_system_to_torch(ps: ParticleSystem, device=None, dtype=torch.float32):
    """
    Convert the arrays in ParticleSystem from NumPy arrays to PyTorch tensors.

    Do this after initialize_positions() and initialize_velocities(). From this
    point on, positions, velocities, forces, masses, sigma and epsilon stay on
    the selected device.
    """
    if device is None:
        device = get_torch_device()

    ps.mass = torch.as_tensor(ps.mass, dtype=dtype, device=device)
    ps.sigma = torch.as_tensor(ps.sigma, dtype=dtype, device=device)
    ps.epsilon = torch.as_tensor(ps.epsilon, dtype=dtype, device=device)
    ps.position = torch.as_tensor(ps.position, dtype=dtype, device=device)
    ps.velocity = torch.as_tensor(ps.velocity, dtype=dtype, device=device)
    ps.force = torch.as_tensor(ps.force, dtype=dtype, device=device)
    ps.random_number = torch.as_tensor(ps.random_number, dtype=dtype, device=device)

    return device


#----------------------------------------------------------------
#   I N I T I A L I Z A T I O N
#----------------------------------------------------------------
def initialize_positions(ps: ParticleSystem, box_length_in_nm: float):
    """Initialize particle positions uniformly in a cubic box using NumPy."""
    ps.position[:] = np.random.uniform(0, box_length_in_nm, size=(ps.n, 3))


def initialize_velocities(ps: ParticleSystem, temperature: float):
    """
    Initialize velocities according to a Maxwell-Boltzmann distribution.

    This function is intentionally NumPy-based and should be called before
    move_particle_system_to_torch().
    """
    # molar masses in kg/mol, converted from u = 1e-3 kg/mol
    M = ps.mass * 1e-3

    # Standard deviation in m/s
    stddev = np.sqrt(R * temperature / M)

    # Sample velocities and convert from m/s to nm/ps
    velocities_m_s = np.random.normal(0.0, stddev[:, np.newaxis], size=(ps.n, 3))
    velocities_nm_ps = velocities_m_s * 1e-3

    ps.velocity[:] = velocities_nm_ps

    # Remove center-of-mass velocity
    v_cm = np.average(ps.velocity, axis=0, weights=ps.mass)
    ps.velocity -= v_cm


#----------------------------------------------------------------
#   E N E R G I E S / P R O P E R T I E S
#----------------------------------------------------------------
def potential_energy(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """
    Compute total Lennard-Jones potential energy in kJ/mol.

    Uses torch operations when ps.position is a torch tensor; otherwise uses NumPy.
    Returns a Python float so it can be stored in NumPy output arrays.
    """
    if torch.is_tensor(ps.position):
        n_particles = ps.n
        sigma = ps.sigma[0]
        epsilon = ps.epsilon[0]
        L = sim.box_length
        device = ps.position.device

        rij_matrix = ps.position[:, None, :] - ps.position[None, :, :]
        rij_matrix = rij_matrix - L * torch.round(rij_matrix / L)
        r_matrix = torch.linalg.norm(rij_matrix, dim=-1)

        i_pairs, j_pairs = torch.triu_indices(
            n_particles, n_particles, offset=1, device=device
        )
        r = r_matrix[i_pairs, j_pairs]

        if sim.use_cutoff and sim.r_cut is not None:
            r = r[r <= sim.r_cut]

        if r.numel() == 0:
            return 0.0

        r = torch.clamp(r, min=sim.rij_min)
        sr6 = (sigma / r) ** 6
        lj_pairwise = 4 * epsilon * (sr6**2 - sr6)
        return tensor_to_float(torch.sum(lj_pairwise))

    # NumPy fallback before conversion to torch
    n_particles = ps.n
    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    L = sim.box_length

    rij_matrix = ps.position[:, np.newaxis, :] - ps.position[np.newaxis, :, :]
    rij_matrix -= L * np.rint(rij_matrix / L)
    r_matrix = np.linalg.norm(rij_matrix, axis=-1)

    i_upper = np.triu_indices(n_particles, k=1)
    r = r_matrix[i_upper]

    if sim.use_cutoff and sim.r_cut is not None:
        r = r[r <= sim.r_cut]

    if r.size == 0:
        return 0.0

    r = np.clip(r, sim.rij_min, None)
    sr6 = (sigma / r) ** 6
    lj_pairwise = 4 * epsilon * (sr6**2 - sr6)
    return float(np.sum(lj_pairwise))


def kinetic_energy(ps: ParticleSystem) -> float:
    """Compute total kinetic energy in kJ/mol and return a Python float."""
    if torch.is_tensor(ps.velocity):
        v_squared = torch.sum(ps.velocity**2, dim=1)
        return tensor_to_float(0.5 * torch.sum(ps.mass * v_squared))

    v_squared = np.sum(ps.velocity**2, axis=1)
    return float(0.5 * np.sum(ps.mass * v_squared))


def instantaneous_temperature(ps: ParticleSystem) -> float:
    """Compute instantaneous temperature in K and return a Python float."""
    E_kin = kinetic_energy(ps) * 1e3  # kJ/mol -> J/mol
    dof = ps.n * 3
    return float((2 * E_kin) / (dof * R))


def density(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """Compute density in g/cm^3 and return a Python float."""
    L_in_nm = sim.box_length
    V_in_cm3 = L_in_nm**3 * 1e-21

    if torch.is_tensor(ps.mass):
        mass_sum = tensor_to_float(torch.sum(ps.mass))
    else:
        mass_sum = float(np.sum(ps.mass))

    m_in_g = mass_sum / Avogadro
    return float(m_in_g / V_in_cm3)


def ideal_gas_pressure(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """Compute instantaneous ideal gas pressure in Pa and return a Python float."""
    V_in_m3 = sim.box_length**3 * 1e-27
    n_mol = ps.n / Avogadro
    T = instantaneous_temperature(ps)
    return float(n_mol * R * T / V_in_m3)


#----------------------------------------------------------------
#   F O R C E S
#----------------------------------------------------------------
def calculate_force(ps: ParticleSystem, sim: SimulationParameters):
    """
    Compatibility wrapper: use the torch force implementation.
    """
    return calculate_force_torch(ps, sim)


def calculate_force_torch(ps: ParticleSystem, sim: SimulationParameters):
    """
    GPU-compatible Lennard-Jones force calculation using PyTorch.

    The expensive part is the pairwise force calculation. The force accumulation
    is done with index_add_ instead of a Python loop over pairs.
    """
    if not torch.is_tensor(ps.position):
        raise TypeError("calculate_force_torch requires ps.position to be a torch tensor. Call move_particle_system_to_torch(ps) first.")

    n_particles = ps.n
    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    L = sim.box_length
    device = ps.position.device

    # Pairwise distance vectors, shape (N, N, 3)
    rij_matrix = ps.position[:, None, :] - ps.position[None, :, :]

    # Minimum-image convention for periodic boundary conditions
    rij_matrix = rij_matrix - L * torch.round(rij_matrix / L)

    # Pairwise distances, shape (N, N)
    r_matrix = torch.linalg.norm(rij_matrix, dim=-1)

    # Unique pairs i < j
    i_pairs, j_pairs = torch.triu_indices(
        n_particles, n_particles, offset=1, device=device
    )

    rij = rij_matrix[i_pairs, j_pairs]
    r = r_matrix[i_pairs, j_pairs]

    # Apply upper cutoff.
    if sim.use_cutoff and sim.r_cut is not None:
        cutoff_mask = r <= sim.r_cut
        rij = rij[cutoff_mask]
        r = r[cutoff_mask]
        i_pairs = i_pairs[cutoff_mask]
        j_pairs = j_pairs[cutoff_mask]

    # If no pair is inside the cutoff, all LJ forces are zero.
    if r.numel() == 0:
        ps.force = torch.zeros_like(ps.position)
        return None

    # Lower cutoff for numerical stability.
    r = torch.clamp(r, min=sim.rij_min)

    # Normalize rij to unit vectors and rescale to match clipped r.
    rij_norm = torch.linalg.norm(rij, dim=1).clamp_min(sim.rij_min)
    rij = (rij / rij_norm[:, None]) * r[:, None]

    # Lennard-Jones force.
    sr6 = (sigma / r) ** 6
    dV_dr = 24 * epsilon / r * (-2 * sr6**2 + sr6)
    f_ij = (dV_dr[:, None] / r[:, None]) * rij

    # Accumulate forces on particles. This replaces the slow Python loop.
    force = torch.zeros_like(ps.position)
    force.index_add_(0, i_pairs, -f_ij)
    force.index_add_(0, j_pairs,  f_ij)

    ps.force = force
    return None


#----------------------------------------------------------------
#   M D   I N T E G R A T O R S
#----------------------------------------------------------------
def A_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Position update."""
    dt = 0.5 * sim.dt if half_step else sim.dt
    ps.position = ps.position + ps.velocity * dt
    return None


def B_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Velocity update from current forces."""
    dt = 0.5 * sim.dt if half_step else sim.dt
    ps.velocity = ps.velocity + (1 / ps.mass)[:, None] * dt * ps.force
    return None


def O_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """Langevin thermostat step using torch operations."""
    if not torch.is_tensor(ps.velocity):
        raise TypeError("O_step in LJ_gas_torch requires torch tensors.")

    dt = 0.5 * sim.dt if half_step else sim.dt
    device = ps.velocity.device
    dtype = ps.velocity.dtype

    ps.random_number = torch.randn((ps.n, 3), dtype=dtype, device=device)

    d = torch.exp(torch.tensor(-sim.xi * dt, dtype=dtype, device=device))

    scalar = sim.temperature * R * (1.0 - np.exp(-2 * sim.xi * dt))

    # Original code uses mass * 1e3 here. Keep this convention for consistency.
    mass = ps.mass * 1e3
    f = torch.sqrt(torch.as_tensor(scalar, dtype=dtype, device=device) / mass)[:, None]

    ps.velocity = d * ps.velocity + f * ps.random_number
    return None


def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters):
    """One NVE Velocity-Verlet step."""
    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=False)
    calculate_force_torch(ps, sim)
    B_step(ps, sim, half_step=True)
    apply_periodic_boundary(ps, sim)
    return None


def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters):
    """One NVT BAOAB Langevin step."""
    if sim.tau_thermostat is None:
        raise ValueError("Thermostat coupling time (tau_thermostat) is not set. Cannot run NVT simulation.")

    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=True)
    O_step(ps, sim, half_step=False)
    A_step(ps, sim, half_step=True)
    calculate_force_torch(ps, sim)
    B_step(ps, sim, half_step=True)
    apply_periodic_boundary(ps, sim)
    return None


def apply_periodic_boundary(ps: ParticleSystem, sim: SimulationParameters):
    """Wrap positions back into the simulation box."""
    if torch.is_tensor(ps.position):
        ps.position = torch.remainder(ps.position, sim.box_length)
    else:
        ps.position = np.mod(ps.position, sim.box_length)
    return None


#----------------------------------------------------------------
#   O U T P U T
#----------------------------------------------------------------
def write_xyz_trajectory(filename, trajectory, atom_symbol="Ar"):
    """Write a trajectory array to an .xyz file."""
    trajectory = tensor_to_numpy(trajectory)
    trajectory = 10.0 * trajectory  # convert nm to Å
    n_frames, n_atoms, _ = trajectory.shape

    with open(filename, "w") as f:
        for frame in trajectory:
            f.write(f"{n_atoms}\n")
            f.write("Generated by write_xyz_trajectory\n")
            for pos in frame:
                f.write(f"{atom_symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


#----------------------------------------------------------------
#   S T A T I S T I C S
#----------------------------------------------------------------
def cutoff_pair_statistics(ps: ParticleSystem, sim: SimulationParameters):
    """
    Count how many unique particle pairs are inside the cutoff radius.

    Returns:
        n_pairs_total (int): total number of unique pairs i < j
        n_pairs_cutoff (int): number of pairs with r <= r_cut
        percent_pairs_cutoff (float): percentage of pairs inside the cutoff
    """
    if sim.r_cut is None:
        raise ValueError("r_cut must be set to calculate cutoff pair statistics.")

    n_particles = ps.n
    L = sim.box_length

    if torch.is_tensor(ps.position):
        device = ps.position.device
        rij_matrix = ps.position[:, None, :] - ps.position[None, :, :]
        rij_matrix = rij_matrix - L * torch.round(rij_matrix / L)
        r_matrix = torch.linalg.norm(rij_matrix, dim=-1)
        i_pairs, j_pairs = torch.triu_indices(n_particles, n_particles, offset=1, device=device)
        r = r_matrix[i_pairs, j_pairs]
        n_pairs_total = int(r.numel())
        n_pairs_cutoff = int(torch.sum(r <= sim.r_cut).detach().cpu().item())
    else:
        rij_matrix = ps.position[:, np.newaxis, :] - ps.position[np.newaxis, :, :]
        rij_matrix -= L * np.rint(rij_matrix / L)
        r_matrix = np.linalg.norm(rij_matrix, axis=-1)
        i_upper = np.triu_indices(n_particles, k=1)
        r = r_matrix[i_upper]
        n_pairs_total = int(r.size)
        n_pairs_cutoff = int(np.sum(r <= sim.r_cut))

    percent_pairs_cutoff = 100 * n_pairs_cutoff / n_pairs_total if n_pairs_total > 0 else 0.0
    return n_pairs_total, n_pairs_cutoff, percent_pairs_cutoff
