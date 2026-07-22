import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R

# Argon parameters used in the MD simulation
sigma = 0.34                     # nm
epsilon = 120.0 * R * 1e-3      # kJ/mol

# Selected DOE cutoff
r_cut_factor = 2.4
r_cut = r_cut_factor * sigma

# Do not start at r = 0 because the potential diverges there
r = np.linspace(0.28, 1.25, 2000)

V = 4.0 * epsilon * (
    (sigma / r)**12
    - (sigma / r)**6
)

# Potential with the hard cutoff used in the simulation
V_cut = np.where(r <= r_cut, V, 0.0)

r_min = 2.0**(1.0 / 6.0) * sigma

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(r, V, linewidth=2.2, label="Lennard-Jones potential")
ax.plot(
    r,
    V_cut,
    "--",
    linewidth=2,
    label="Potential with hard cutoff",
)

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(
    sigma,
    color="grey",
    linestyle=":",
    label=r"$\sigma$",
)
ax.axvline(
    r_cut,
    color="red",
    linestyle="--",
    label=rf"$r_{{cut}}={r_cut_factor}\sigma$",
)

ax.scatter(
    r_min,
    -epsilon,
    color="black",
    zorder=5,
    label=r"Minimum $r=2^{1/6}\sigma$",
)

ax.axvspan(
    r_cut,
    r.max(),
    color="grey",
    alpha=0.12,
    label="Neglected region",
)

# Limit the visible repulsive branch so that the minimum remains readable
ax.set_ylim(-1.2 * epsilon, 2.0 * epsilon)
ax.set_xlim(r.min(), r.max())

ax.set_xlabel(r"Particle distance $r$ [nm]")
ax.set_ylabel(r"Lennard-Jones potential $V_{\mathrm{LJ}}(r)$ [kJ/mol]")
ax.set_title("Lennard-Jones potential of argon")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig("LJ_potential_argon.png", dpi=300, bbox_inches="tight")
plt.show()