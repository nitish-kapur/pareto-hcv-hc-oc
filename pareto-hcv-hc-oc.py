"""
Feedstock Pareto Optimisation for H/C, O/C, and HCV
Copyright (C) 2026 Nitish Kapur
GitHub: github.com/nitish-kapur
Licensed under GNU GPLv3

    This script was made as a part of a biofuel research project.

    1.  Defines the elemental compositions (C, H, O, N, S as % mass) for
        each feedstock in the `biomass` dictionary. Update these values to
        match your own feedstocks and experimental data.

    2.  Implements three property calculators for a given blend vector x
        (percentage of each feedstock):
            - H/C atomic ratio: molar hydrogen to carbon ratio
            - O/C ratio: mass oxygen to carbon ratio
            - HCV: Higher Calorific Value (MJ/kg) estimated using Modified Dulong formula.

    3.  Defines a weighted objective function that simultaneously maximises
        normalised HCV and minimises deviation from target H/C and O/C
        ratios. Update the target values in the objective function to match
        your desired blend properties.

    4.  Applies constraints to the optimisation — feedstock percentages must
        sum to 100%, and H/C, O/C, and HCV must fall within specified bounds.
        Update these bounds in the `constraints` list to match your
        experimental requirements.

    5.  Sets minimum and maximum blend percentages for each feedstock in the
        `bounds` list. Update these to reflect your available feedstock
        quantities.

    6.  Runs SLSQP optimisation via scipy.optimize.minimize across evenly
        spaced weight values between 0 and 1, collecting all successful
        results to form the Pareto front.

    7.  Prints optimal blend percentages, H/C, O/C, and HCV for each
        weight value to the console.

    8.  Plots the Pareto front as H/C ratio vs HCV in a matplotlib window.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"""
        Elemental (CHNS) analysis (% mass) of rice straw (PR 121), LDPE (Verka milk packets), 
        coconut shells (South Indian), and walnut shells (Kashmir, India), was performed using 
        an Elementar Vario EL Cube CHNS Analyser.
"""

biomass = {
    'rice_straw':  {'C': 37.51, 'H': 10.043,  'N': 0.52,  'O': 51.560, 'S': 0.368},
    'LDPE':       {'C': 79.22, 'H': 8.253, 'N': 0.0, 'O': 12.301,  'S': 0.226},
    'coconut':    {'C': 46.53,  'H': 5.829,  'N': 0.30, 'O': 46.991,  'S': 0.350},
    'walnut':     {'C': 45.53,  'H': 5.829,   'N': 0.30,  'O': 47.893, 'S': 0.254},
}

names = ['rice_straw', 'LDPE', 'coconut', 'walnut']
C_atomic_mass = 12
H_atomic_mass = 1
O_atomic_mass = 16

# Calculate H/C ratio
def calc_HC_ratio(x):
    C_mass = sum(biomass[n]['C'] * x[i] / 100 for i, n in enumerate(names))
    H_mass = sum(biomass[n]['H'] * x[i] / 100 for i, n in enumerate(names))
    C_moles = C_mass / C_atomic_mass
    H_moles = H_mass / H_atomic_mass
    if C_moles == 0:
        return 0
    return H_moles / C_moles

# Calculate O/C ratio
def calc_OC_ratio(x):
    C_mass = sum(biomass[n]['C'] * x[i] / 100 for i, n in enumerate(names))
    O_mass = sum(biomass[n]['O'] * x[i] / 100 for i, n in enumerate(names))
    if C_mass == 0:
        return 0
    return O_mass / C_mass

# Calculate Higher Calorific Value (HCV)
def calc_HCV(x):
    total_mass = sum(x)
    C_frac = sum(biomass[n]['C'] * x[i] / 100 for i, n in enumerate(names)) / total_mass
    H_frac = sum(biomass[n]['H'] * x[i] / 100 for i, n in enumerate(names)) / total_mass
    O_frac = sum(biomass[n]['O'] * x[i] / 100 for i, n in enumerate(names)) / total_mass
    LCV = (38.2 * C_frac) + 84.9 * (H_frac - O_frac / 8) - 0.5
    HCV = (LCV + 0.024 * 2 * H_frac)
    return HCV

# Normalize values
def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)

# Objective function: combine HCV maximization, H/C ratio minimization, and O/C ratio minimization
def objective(x, w=0.8, hc_target=1.8, oc_target=1):
    hc = calc_HC_ratio(x)
    hcv = calc_HCV(x)
    oc = calc_OC_ratio(x)
    hcv_norm = normalize(hcv, 16, 26)   # approximate realistic range for HCV
    hc_dev_norm = abs(hc - hc_target) / (1.5 - 0.8)  # normalized absolute deviation from target H/C ratio
    oc_dev_norm = abs(oc - oc_target) / (1 - 0.2)  # normalized O/C deviation (target is 0.4)
    return -(w * hcv_norm - (1 - w) * hc_dev_norm - (1 - w) * oc_dev_norm)

# Constraint: sum of all feedstock percentages should be 100%
def constraint_sum(x):
    return np.sum(x) - 100

# Constraints for sum, H/C ratio, O/C ratio bounds (as inequalities)
constraints = [
    {'type': 'eq', 'fun': constraint_sum},
    {'type': 'ineq', 'fun': lambda x: calc_HC_ratio(x) - 0.8},
    {'type': 'ineq', 'fun': lambda x: 1.8 - calc_HC_ratio(x)},
    {'type': 'ineq', 'fun': lambda x: calc_HCV(x) - 20},
    {'type': 'ineq', 'fun': lambda x: calc_OC_ratio(x) - 0},
    {'type': 'ineq', 'fun': lambda x: 1 - calc_OC_ratio(x)},
]

# Bounds for the four feedstocks
bounds = [
    (10, 30),   # rice_straw
    (10, 25),   # LDPE
    (10, 30),   # coconut
    (10, 20),   # walnut
]

# Initial guess for optimization: [Rice, LDPE, Coconut, Walnut]
x0 = [15, 20, 20, 20]  # Initial values for Rice, LDPE, Coconut Shells, and Walnut SHells

# Create a list to store optimization results for different weightings
weights = np.linspace(0, 1, 20)
results = []

# Perform optimization for each weight factor
for w in weights:
    res = minimize(
        objective,
        x0,
        args=(w,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': False}
    )
    if res.success:
        x_opt = res.x
        hc_opt = calc_HC_ratio(x_opt)
        hcv_opt = calc_HCV(x_opt)
        oc_opt = calc_OC_ratio(x_opt)
        results.append({'w': w, 'x': x_opt, 'hc': hc_opt, 'hcv': hcv_opt, 'oc': oc_opt})

# Print results
print("Pareto optimal mixtures:")
for r in results:
    print(f"Weight w={r['w']:.2f} -> H/C={r['hc']:.3f}, HCV={r['hcv']:.2f} MJ/kg, O/C={r['oc']:.3f}")
    for i, n in enumerate(names):
        print(f"  {n}: {r['x'][i]:.2f} g")
    print()

# Plot the Pareto front
hc_vals = [r['hc'] for r in results]
hcv_vals = [r['hcv'] for r in results]

plt.figure(figsize=(8, 6))
plt.plot(hc_vals, hcv_vals, 'o-', label='Pareto Front')
plt.xlabel('H/C atomic ratio')
plt.ylabel('Higher Calorific Value (MJ/kg)')
plt.title('Pareto Optimization of Biomass Mixture (4 Feedstocks)')
plt.grid(True)
plt.legend()
plt.show()

