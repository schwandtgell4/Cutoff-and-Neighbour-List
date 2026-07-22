import numpy as np
import matplotlib.pyplot as plt


# Reduced variables are used only to draw the characteristic LJ shape.
# No numerical scale is shown because this is a conceptual illustration.
r = np.linspace(0.82, 4.0, 2500)
V = 4.0 * ((1.0 / r)**12 - (1.0 / r)**6)

# Representative position of the cutoff. Its numerical value is deliberately
# not shown in the figure.
r_cut = 2.0
V_at_cut = 4.0 * ((1.0 / r_cut)**12 - (1.0 / r_cut)**6)

fig, ax = plt.subplots(figsize=(9, 5.5))

blue = "#0b4db3"
red = "#d62f1f"

# True Lennard-Jones potential
ax.plot(r, V, color=blue, linewidth=3.0)

# Hard cutoff: the potential jumps to zero and remains zero afterwards.
ax.plot(
    [r_cut, r_cut],
    [V_at_cut, 0.0],
    color=red,
    linewidth=3.0,
)
ax.plot(
    [r_cut, r.max()],
    [0.0, 0.0],
    color=red,
    linewidth=3.0,
)

# Conceptual coordinate axes without ticks, values or units.
ax.annotate(
    "",
    xy=(4.08, 0.0),
    xytext=(0.76, 0.0),
    arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.8},
    annotation_clip=False,
)
ax.annotate(
    "",
    xy=(0.78, 1.42),
    xytext=(0.78, -1.15),
    arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.8},
    annotation_clip=False,
)

# Generic cutoff marker
ax.plot(
    [r_cut, r_cut],
    [0.06, 0.58],
    color="black",
    linewidth=1.6,
    linestyle=(0, (2, 2)),
)

# Direct labels keep the graphic readable without a legend.
ax.text(0.66, 1.42, r"$V(r)$", fontsize=22, fontweight="bold")
ax.text(4.09, -0.08, r"$r$", fontsize=22, fontweight="bold")
ax.text(
    r_cut,
    0.62,
    r"$r_{\mathrm{cut}}$",
    fontsize=18,
    ha="center",
)
ax.text(1.34, -0.58, "true potential", color=blue, fontsize=17)
ax.text(2.45, 0.13, "truncated potential", color=red, fontsize=17)

ax.annotate(
    "discontinuity",
    xy=(r_cut, 0.5 * V_at_cut),
    xytext=(2.62, -0.40),
    fontsize=16,
    arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.6},
)

ax.set_xlim(0.70, 4.15)
ax.set_ylim(-1.22, 1.50)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig(
    "LJ_potential_cutoff_concept.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
