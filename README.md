# Feedstock Pareto Optimisation for H/C, O/C, and HCV

A Python script that performs multi-objective Pareto optimisation on a four-feedstock biomass blend, balancing Higher Calorific Value (HCV), H/C ratio, and O/C ratio using weighted SLSQP optimisation.

---

## Author

**Nitish Kapur**<br>
GitHub: [github.com/nitish-kapur](https://github.com/nitish-kapur)

---

## Feedstocks

The elemental compositions (C, H, O, N, S as % mass) for each feedstock are hard-coded in the `biomass` dictionary at the top of the script. Update these values to match your own feedstocks and experimental data.

---

## Optimisation Parameters

The H/C ratio bounds, O/C ratio bounds, HCV minimum, feedstock blend percentages, and weight factor are all hard-coded in the `constraints` and `bounds` sections of the script. Modify these values to suit your desired optimisation targets and experimental constraints.
---

## Requirements
```bash
pip install numpy scipy matplotlib
```

---

## Usage

Run the script directly:
```bash
python pareto-hcv-hc-oc.py
```

Pareto optimal mixtures are printed to the console and a Pareto front plot is displayed interactively.

---

## Output

- Console output listing optimal feedstock blend percentages, H/C, O/C, and HCV for each weight factor.
- An interactive matplotlib plot of the Pareto front (H/C ratio vs HCV).

---

## Notes

- The weight factor `w` controls the trade-off between maximising HCV (`w → 1`) and minimising deviation from target H/C and O/C ratios (`w → 0`).
- Only successful optimisations are included in the results.
- Elemental compositions and blend bounds are hard-coded at the top of the script — edit the `biomass` dictionary and `bounds` list to change feedstocks.
- HCV is estimated using *Modified Dulong* formula.
