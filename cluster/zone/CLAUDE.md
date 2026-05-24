# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Starlink 550 satellite network traffic engineering pipeline. The codebase models satellite constellations as time-varying graphs across 500 time slices, clusters satellites into 18 spatial domains (6 big orbital planes × 3 phase segments), and builds both inter-domain and intra-domain topologies with traffic matrices and K-shortest paths. A Gurobi MILP solver computes optimal link weights for intra-domain traffic engineering.

## Pipeline (run order)

1. **SatSimPro-main** (sibling project) — simulates satellite orbits, outputs `step_*.txt` files with per-timeslice node positions and ISL links
2. `build_starlink550_tms.py` — parses step files, samples population-based traffic demands (3000 flows per slice), computes K=8 shortest paths per pair → `outputs/tms/starlink550.pkl`
3. `cluster_500steps.py` — RAAN-bucket clustering of satellites by orbital plane normals, merges 72 planes → 6 big planes, splits each into 3 phases → 18 domains; outputs per-slice domain mapping files
4. `data_build_500steps.py` — builds super-node (inter-domain) topology by aggregating micro TMs/paths up to 18-domain level → `outputs/tms/starlink550_cluster.pkl`
5. `data_build_intra.py` — extracts intra-domain graphs, TMs, and paths per domain per slice → `outputs/tms/starlink550_intra.pkl`
6. `intra_link_weight_solver.py` — Gurobi MILP: finds integer link weights (1–20) minimizing maximum link utilization (MLU) for a single domain's intra-domain traffic

## Supporting files

- `cluster_single.py` / `data_build.py` — single-timeslice versions of steps 3–4 (for debugging/visualization)
- `extract_domain_plot.py` — exports a single slice+domain as PKL and plots its topology with networkx spring layout

## Key data formats

**Step files** (`inputs/starlink550_data/step_*.txt`): text with `[NODES]` section (ID, Name, Lat, Lon, Alt comma-separated) and `[LINKS]` section (Type, SourceID, TargetID).

**PKL files** (outputs of steps 2/4/5): `list[dict]` where each dict has:
- `graph`: `list[list[int,int]]` — physical edges as [u, v] pairs
- `tm`: `dict[str, int]` — key `"src, dst"` → demand value
- `path`: `dict[str, list[list[int]]]` — key `"src, dst"` → up to 8 parallel paths (each path is a node sequence)
- `data_idx`: int — time slice index

**Intra-domain PKL** (`starlink550_intra.pkl`): `list[dict]` with `domains` key containing per-domain dicts that each have their own `graph`, `tm`, `path`, `active_sat_ids`.

## Configuration constants

All core parameters are set as module-level variables at the top of each script:

| Parameter | Typical value | Where used |
|---|---|---|
| `NUM_SLICES` | 500 | build_starlink550_tms.py |
| `K_PATHS` / `MAX_PARALLEL_LINKS` | 8 | all scripts |
| `N_planes_by_raan` | 72 | cluster*.py |
| `N_BIG_PLANES` | 6 | cluster*.py, data_build*.py |
| `N_PHASES` | 3 | cluster*.py, data_build*.py |
| `N_DOMAINS` / `N_CLUSTERS` | 18 | data_build*.py |
| `RADIUS_KM` | 50 | build_starlink550_tms.py |
| `WEIGHT_MIN` / `WEIGHT_MAX` | 1 / 20 | intra_link_weight_solver.py |
| `TIME_LIMIT` | 300 (seconds) | intra_link_weight_solver.py |

## Dependencies

numpy, networkx, matplotlib, scikit-learn (DBSCAN), rasterio (GeoTIFF population data), gurobipy (MILP solver — requires Gurobi license), pickle (stdlib)

## Environment

Python environment managed via conda (`ms-python.python:conda` in VSCode settings). Input data (`step_*.txt`, `landscan-global-2024.tif`) placed in `inputs/`. All outputs go to `outputs/`.
