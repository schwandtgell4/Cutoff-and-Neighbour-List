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
def move_particle_system_to_torch(
    ps: ParticleSystem,
    device=None,
    dtype=torch.float32
):
    """
    Convert ParticleSystem arrays to tensors and cache constants
    that do not change during the simulation.
    """
    if device is None:
        device = get_torch_device()

    ps.mass = torch.as_tensor(
        ps.mass,
        dtype=dtype,
        device=device
    )

    ps.sigma = torch.as_tensor(
        ps.sigma,
        dtype=dtype,
        device=device
    )

    ps.epsilon = torch.as_tensor(
        ps.epsilon,
        dtype=dtype,
        device=device
    )

    ps.position = torch.as_tensor(
        ps.position,
        dtype=dtype,
        device=device
    )

    ps.velocity = torch.as_tensor(
        ps.velocity,
        dtype=dtype,
        device=device
    )

    ps.force = torch.as_tensor(
        ps.force,
        dtype=dtype,
        device=device
    )

    ps.random_number = torch.as_tensor(
        ps.random_number,
        dtype=dtype,
        device=device
    )

    # Masses remain constant.
    ps.inverse_mass = torch.reciprocal(
        ps.mass
    )

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


#--------------------------------------
# nachbarliste def #updated, so dass auch aktualisiert wenn ein teilchen mehr als die halbe skin bewegt hat
#--------------------------------------

def update_neighbour_list(
    ps: ParticleSystem,
    sim: SimulationParameters,
    step: int,
    n_update: int = 10
):
    """
    Build and update the neighbour list entirely with Torch.

    The neighbour list is built initially and then every
    n_update MD steps. Between updates, the stored pair list
    is reused.

    All distance calculations remain on the selected Torch
    device: CUDA, MPS or CPU fallback.
    """

    if n_update < 1:
        raise ValueError(
            "n_update must be at least 1"
        )

    if not sim.use_cutoff or sim.r_cut is None:
        raise ValueError(
            "neighbour list requires "
            "use_cutoff=True and r_cut"
        )

    if not torch.is_tensor(ps.position):
        raise TypeError(
            "update_neighbour_list requires "
            "Torch tensors"
        )

    first_build = not hasattr(
        ps,
        "neighbour_pairs"
    )

    # Reuse the current neighbour list unless this
    # is an update step.
    if (
        not first_build
        and step % n_update != 0
    ):
        return ps.neighbour_pairs

    if first_build:
        device = ps.position.device

        # Keep the skin as a scalar tensor on the device.
        ps.neighbour_skin = (
            0.3 * ps.sigma[0]
        )

        ps.neighbour_pairs = torch.empty(
            (0, 2),
            dtype=torch.long,
            device=device
        )

        ps.neighbour_list_step = -1
        ps.neighbour_rebuilds = 0

        # Generate the complete list of unique pair indices
        # only once. These indices remain on the device and
        # are reused at every neighbour-list update.
        all_pairs = torch.triu_indices(
            ps.n,
            ps.n,
            offset=1,
            device=device
        )

        ps.all_pair_i = all_pairs[0]
        ps.all_pair_j = all_pairs[1]

    i_pairs = ps.all_pair_i
    j_pairs = ps.all_pair_j

    # Pair displacement vectors.
    rij = (
        ps.position[i_pairs]
        - ps.position[j_pairs]
    )

    # Minimum-image periodic boundary conditions.
    rij -= (
        sim.box_length
        * torch.round(
            rij / sim.box_length
        )
    )

    # Squared distances; no square root is required.
    r_squared = torch.sum(
        rij * rij,
        dim=1
    )

    # The neighbour list uses cutoff + skin.
    list_cutoff = (
        sim.r_cut
        + ps.neighbour_skin
    )

    inside_neighbour_list = (
        r_squared
        <= list_cutoff * list_cutoff
    )

    # Store only pairs inside cutoff + skin.
    ps.neighbour_pairs = torch.stack(
        (
            i_pairs[inside_neighbour_list],
            j_pairs[inside_neighbour_list]
        ),
        dim=1
    )

    ps.neighbour_list_step = step
    ps.neighbour_rebuilds += 1

    return ps.neighbour_pairs   

#----------------------------------------------------------------
#   E N E R G I E S / P R O P E R T I E S #Neigbbouhr list verwenden, rest bleibt als backup da
#----------------------------------------------------------------
def potential_energy(
    ps: ParticleSystem,
    sim: SimulationParameters
) -> float:
    """
    Return the energy already calculated together
    with the forces.
    """
    if not hasattr(
        ps,
        "current_potential_energy"
    ):
        raise ValueError(
            "forces must be calculated before "
            "potential_energy"
        )

    return tensor_to_float(
        ps.current_potential_energy
    )



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


#kein nutzen der NXN matrix mehr als ganzes, nur noch neighbour list
def calculate_force_torch(
    ps: ParticleSystem,
    sim: SimulationParameters
):
    """
    Calculate cutoff Lennard-Jones forces from
    the current neighbour list.

    Squared distances are used throughout.
    The potential energy is calculated from the
    same pair data.
    """
    if not torch.is_tensor(ps.position):
        raise TypeError(
            "calculate_force_torch requires "
            "torch tensors"
        )

    if not hasattr(
        ps,
        "neighbour_pairs"
    ):
        raise ValueError(
            "neighbour list was not created"
        )

    # Reuse the existing force tensor.
    ps.force.zero_()

    if ps.neighbour_pairs.numel() == 0:
        ps.current_potential_energy = (
            ps.position.new_zeros(())
        )

        return None

    i_pairs = (
        ps.neighbour_pairs[:, 0]
    )

    j_pairs = (
        ps.neighbour_pairs[:, 1]
    )

    rij = (
        ps.position[i_pairs]
        - ps.position[j_pairs]
    )

    rij -= (
        sim.box_length
        * torch.round(
            rij / sim.box_length
        )
    )

    # No square root is required.
    r_squared = torch.sum(
        rij * rij,
        dim=1
    )

    inside_cutoff = (
        r_squared
        <= sim.r_cut * sim.r_cut
    )

    i_pairs = i_pairs[
        inside_cutoff
    ]

    j_pairs = j_pairs[
        inside_cutoff
    ]

    rij = rij[
        inside_cutoff
    ]

    r_squared = r_squared[
        inside_cutoff
    ]

    if r_squared.numel() == 0:
        ps.current_potential_energy = (
            ps.position.new_zeros(())
        )

        return None

    minimum_r_squared = max(
        sim.rij_min * sim.rij_min,
        1e-24
    )

    r_squared = torch.clamp(
        r_squared,
        min=minimum_r_squared
    )

    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]

    inverse_r_squared = (
        torch.reciprocal(
            r_squared
        )
    )

    sigma_over_r_squared = (
        sigma
        * sigma
        * inverse_r_squared
    )

    sr6 = (
        sigma_over_r_squared**3
    )

    sr12 = sr6 * sr6

    force_factor = (
        24.0
        * epsilon
        * inverse_r_squared
        * (
            2.0 * sr12
            - sr6
        )
    )

    pair_force = (
        force_factor[:, None]
        * rij
    )

    ps.force.index_add_(
        0,
        i_pairs,
        pair_force
    )

    ps.force.index_add_(
        0,
        j_pairs,
        -pair_force
    )

    # Reuse the same distances for the energy.
    ps.current_potential_energy = (
        torch.sum(
            4.0
            * epsilon
            * (
                sr12
                - sr6
            )
        )
    )

    return None

#----------------------------------------------------------------
#   M D   I N T E G R A T O R S
#----------------------------------------------------------------
def A_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    half_step=False
):
    """In-place position update."""
    dt = (
        0.5 * sim.dt
        if half_step
        else sim.dt
    )

    ps.position.add_(
        ps.velocity,
        alpha=dt
    )

    return None


def B_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    half_step=False
):
    """
    In-place velocity update using cached
    inverse masses.
    """
    dt = (
        0.5 * sim.dt
        if half_step
        else sim.dt
    )

    if not hasattr(
        ps,
        "inverse_mass"
    ):
        ps.inverse_mass = (
            torch.reciprocal(
                ps.mass
            )
        )

    ps.velocity.add_(
        ps.inverse_mass[:, None]
        * ps.force,
        alpha=dt
    )

    return None


def O_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    half_step=False
):
    """
    In-place Langevin thermostat step with
    cached constants.
    """
    if not torch.is_tensor(ps.velocity):
        raise TypeError(
            "O_step in LJ_gas_torch requires "
            "torch tensors."
        )

    dt = (
        0.5 * sim.dt
        if half_step
        else sim.dt
    )

    cache_key = (
        dt,
        sim.xi,
        sim.temperature
    )

    if (
        getattr(
            ps,
            "_thermostat_cache_key",
            None
        )
        != cache_key
    ):
        dtype = ps.velocity.dtype
        device = ps.velocity.device

        ps.thermostat_decay = (
            torch.exp(
                torch.as_tensor(
                    -sim.xi * dt,
                    dtype=dtype,
                    device=device
                )
            )
        )

        scalar = (
            sim.temperature
            * R
            * (
                1.0
                - np.exp(
                    -2.0
                    * sim.xi
                    * dt
                )
            )
        )

        ps.thermostat_noise_scale = (
            torch.sqrt(
                torch.as_tensor(
                    scalar,
                    dtype=dtype,
                    device=device
                )
                / (
                    ps.mass
                    * 1e3
                )
            )[:, None]
        )

        ps._thermostat_cache_key = (
            cache_key
        )

    # Reuse the existing random-number tensor.
    ps.random_number.normal_()

    ps.velocity.mul_(
        ps.thermostat_decay
    )

    ps.velocity.add_(
        ps.thermostat_noise_scale
        * ps.random_number
    )

    return None



#aktualiesierung der neighbour list in den sims selbst  
def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters, step: int, n_update: int):
    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=False)

    apply_periodic_boundary(ps, sim)

    update_neighbour_list(
        ps,
        sim,
        step,
        n_update
    )

    calculate_force_torch(
        ps,
        sim
    )

    B_step(ps, sim, half_step=True)

    return None


def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters, step: int, n_update: int):
    if sim.tau_thermostat is None:
        raise ValueError("Thermostat coupling time is not set")

    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=True)
    O_step(ps, sim, half_step=False)
    A_step(ps, sim, half_step=True)

    apply_periodic_boundary(ps, sim)

    update_neighbour_list(
        ps,
        sim,
        step,
        n_update
    )

    calculate_force_torch(
        ps,
        sim
    )

    B_step(ps, sim, half_step=True)

    return None

def apply_periodic_boundary(
    ps: ParticleSystem,
    sim: SimulationParameters
):
    """
    Wrap positions into the simulation box
    in place.
    """
    if torch.is_tensor(ps.position):
        ps.position.remainder_(
            sim.box_length
        )
    else:
        np.mod(
            ps.position,
            sim.box_length,
            out=ps.position
        )

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
def cutoff_pair_statistics(
    ps: ParticleSystem,
    sim: SimulationParameters
):
    """
    Count current cutoff pairs using the
    existing neighbour list.
    """
    if sim.r_cut is None:
        raise ValueError(
            "r_cut must be set to calculate "
            "cutoff pair statistics."
        )

    if not hasattr(
        ps,
        "neighbour_pairs"
    ):
        raise ValueError(
            "neighbour list was not created"
        )

    n_pairs_total = (
        ps.n
        * (ps.n - 1)
        // 2
    )

    if ps.neighbour_pairs.numel() == 0:
        return (
            n_pairs_total,
            0,
            0.0
        )

    i_pairs = (
        ps.neighbour_pairs[:, 0]
    )

    j_pairs = (
        ps.neighbour_pairs[:, 1]
    )

    rij = (
        ps.position[i_pairs]
        - ps.position[j_pairs]
    )

    rij -= (
        sim.box_length
        * torch.round(
            rij / sim.box_length
        )
    )

    r_squared = torch.sum(
        rij * rij,
        dim=1
    )

    n_pairs_cutoff = int(
        torch.count_nonzero(
            r_squared
            <= sim.r_cut
            * sim.r_cut
        )
        .detach()
        .cpu()
        .item()
    )

    percent_pairs_cutoff = (
        100.0
        * n_pairs_cutoff
        / n_pairs_total
        if n_pairs_total > 0
        else 0.0
    )

    return (
        n_pairs_total,
        n_pairs_cutoff,
        percent_pairs_cutoff
    )