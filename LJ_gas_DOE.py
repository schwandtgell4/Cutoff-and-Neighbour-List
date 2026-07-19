import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R
from itertools import product
from pathlib import Path

from LJ_gas import ParticleSystem
from LJ_gas import SimulationParameters
from LJ_gas import initialize_positions
from LJ_gas import initialize_velocities
from LJ_gas import update_neighbour_list
from LJ_gas import calculate_force
from LJ_gas import simulate_NVT_step
from LJ_gas import kinetic_energy
from LJ_gas import density
from LJ_gas import instantaneous_temperature
from LJ_gas import potential_energy

n_particles = 2000
mass_argon = 39.95
sigma_argon = 0.34
epsilon_argon = 120 * R * 1e-3

dt = 0.1
n_steps = 1000
temperature = 300
box_length_start = 100
tau_thermostat = 1
rij_min = 1e-2

n_equil_steps = 20000 # steps used for equilibration
equilibration_sample_interval = 10  # Record equilibration observables every 10 steps
equilibration_n_update = 1 # neighbor list is rebuild after every step during equilibration

n_update_werte = list(range(1, 11))
cutoff_faktoren = [2.5, 5, 7.5, 10]
dichte_faktoren = [1.0]
seeds = [1, 2, 3, 4, 5]
equilibration_cutoff_factor = max(cutoff_faktoren) # use largest cutoff during equilibration

ordner = "doe_ergebnisse"
os.makedirs(ordner, exist_ok=True)

equilibrium_states = {} # save pos + velocity 
equilibration_history = [] # save temp + ekin

ergebnisse = []

print(
    "Creating common equilibrated states...",
    flush=True
)

for dichte_faktor, seed in product(
    dichte_faktoren,
    seeds
):
    # Use a reproducible initialization and thermostat sequence.
    np.random.seed(seed)

    box_length = (box_length_start / dichte_faktor ** (1.0 / 3.0))

    r_cut_equilibration = (equilibration_cutoff_factor * sigma_argon)

    sim_equilibration = SimulationParameters(
        dt=dt,
        n_steps=n_equil_steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
        r_cut=r_cut_equilibration
    )

    ps_equilibration = ParticleSystem(n_particles)

    ps_equilibration.mass[:] = (mass_argon)

    ps_equilibration.sigma[:] = (sigma_argon)

    ps_equilibration.epsilon[:] = (epsilon_argon)

    initialize_positions(ps_equilibration, sim_equilibration.box_length)

    initialize_velocities(ps_equilibration, sim_equilibration.temperature)

    update_neighbour_list(
        ps_equilibration,
        sim_equilibration,
        step=0,
        n_update=equilibration_n_update
    )

    calculate_force(ps_equilibration, sim_equilibration)
    for equilibration_step in range(1, n_equil_steps + 1):
        simulate_NVT_step(
            ps_equilibration,
            sim_equilibration,
            equilibration_step,
            equilibration_n_update
        )
        if (equilibration_step% equilibration_sample_interval== 0):
            equilibration_history.append({
                "dichte_faktor": (dichte_faktor),
                "seed": seed,
                "step": (equilibration_step),
                "temperature_K": (instantaneous_temperature(ps_equilibration)),
                "potential_energy_kJ_mol": (potential_energy(ps_equilibration, sim_equilibration)
                )})
    equilibrium_states[(dichte_faktor, seed)] = {
        "position": (ps_equilibration.position.copy()),
        "velocity": (ps_equilibration.velocity.copy())
    }
    print("Equilibrated state finished:", f"density factor={dichte_faktor}", f"seed={seed}",flush=True)

equilibration_table = pd.DataFrame(equilibration_history)

equilibration_table.to_csv(os.path.join(ordner,"equilibration_history.csv"),index=False)
kombinationen = list(product(n_update_werte, cutoff_faktoren, dichte_faktoren, seeds))

zufall_reihenfolge = np.random.default_rng(12345)
zufall_reihenfolge.shuffle(kombinationen)

anzahl = len(kombinationen)

for nummer, kombination in enumerate(kombinationen):
    n_update = kombination[0]
    cutoff_faktor = kombination[1]
    dichte_faktor = kombination[2]
    seed = kombination[3]

    production_seed = (100000 + seed)
    np.random.seed(production_seed)

    box_length = box_length_start / dichte_faktor ** (1 / 3)
    r_cut = cutoff_faktor * sigma_argon

    sim = SimulationParameters(
        dt=dt,
        n_steps=n_steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
        r_cut=r_cut
    )

    ps = ParticleSystem(n_particles)
    ps.mass[:] = mass_argon
    ps.sigma[:] = sigma_argon
    ps.epsilon[:] = epsilon_argon
    equilibrium_state = (equilibrium_states[(dichte_faktor, seed)])
    ps.position[:] = (equilibrium_state["position"])
    ps.velocity[:] = (equilibrium_state["velocity"])

    update_neighbour_list(ps, sim, step=0, n_update=n_update)
    calculate_force(ps, sim)

    temperaturen = []
    kinetische_energien = []

    e_kin = kinetic_energy(ps)
    temp = 2 * e_kin * 1000 / (3 * ps.n * R)
    kinetische_energien.append(e_kin)
    temperaturen.append(temp)

    start = time.perf_counter()

    for schritt in range(n_steps):
        simulate_NVT_step(ps, sim, schritt + 1, n_update)
        e_kin = kinetic_energy(ps)
        temp = 2 * e_kin * 1000 / (3 * ps.n * R)
        kinetische_energien.append(e_kin)
        temperaturen.append(temp)

    dauer = time.perf_counter() - start

    temperaturen = np.array(temperaturen)
    kinetische_energien = np.array(kinetische_energien)

    temperatur_abweichung = np.mean(np.abs(temperaturen - np.mean(temperaturen)))
    energie_abweichung = np.mean(np.abs(kinetische_energien - np.mean(kinetische_energien)))
    geschwindigkeit = n_steps / dauer
    aktuelle_dichte = density(ps, sim)

    ergebnisse.append({
        "n_update": n_update,
        "cutoff_faktor": cutoff_faktor,
        "cutoff_radius_nm": r_cut,
        "dichte_faktor": dichte_faktor,
        "dichte_g_cm3": aktuelle_dichte,
        "seed": seed,
        "temperatur_abweichung_K": temperatur_abweichung,
        "simulationsgeschwindigkeit_schritte_pro_s": geschwindigkeit,
        "kinetische_energie_abweichung_kJ_mol": energie_abweichung
    })

    tabelle = pd.DataFrame(ergebnisse)
    tabelle.to_csv(os.path.join(ordner, "doe_ergebnisse.csv"), index=False)

    print(
    str(nummer + 1) + "/" + str(anzahl)
    + ", Laufzeit=" + format(dauer, ".2f") + " s"
    + ", n=" + str(n_update)
    + ", rcutoff=" + format(r_cut, ".3f") + " nm"
    + ", Dichte=" + format(aktuelle_dichte, ".6e") + " g/cm3"
    + ", Abw. T=" + format(temperatur_abweichung, ".4f") + " K"
    + ", Abw. Ekin=" + format(energie_abweichung, ".4f") + " kJ/mol",
    flush=True
)


def heatmap(tabelle, x_name, y_name, z_name):
    daten = tabelle.pivot_table(index=y_name, columns=x_name, values=z_name, aggfunc="mean")

    plt.figure(figsize=(10, 7))
    bild = plt.imshow(daten.values, aspect="auto", origin="lower")
    plt.colorbar(bild, label=z_name)
    plt.xticks(range(len(daten.columns)), [str(x) for x in daten.columns], rotation=45)
    plt.yticks(range(len(daten.index)), [str(y) for y in daten.index])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(z_name + " für " + x_name + " und " + y_name)

    for y in range(len(daten.index)):
        for x in range(len(daten.columns)):
            plt.text(x, y, format(daten.iloc[y, x], ".3g"), ha="center", va="center")

    plt.tight_layout()
    dateiname = z_name + "__" + x_name + "__" + y_name + ".png"
    plt.savefig(os.path.join(ordner, dateiname), dpi=300)
    plt.close()


tabelle = pd.DataFrame(ergebnisse)

ziele = [
    "temperatur_abweichung_K",
    "simulationsgeschwindigkeit_schritte_pro_s",
    "kinetische_energie_abweichung_kJ_mol"
]

for ziel in ziele:
    heatmap(tabelle, "n_update", "cutoff_faktor", ziel)
    heatmap(tabelle, "n_update", "dichte_faktor", ziel)
    heatmap(tabelle, "cutoff_faktor", "dichte_faktor", ziel)

mittelwerte = tabelle.groupby(["n_update", "cutoff_faktor", "dichte_faktor"], as_index=False).mean(numeric_only=True)
mittelwerte.to_csv(os.path.join(ordner, "doe_mittelwerte.csv"), index=False)
