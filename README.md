# MDP-Based Heuristic for Planning

This repository contains the implementation of a novel heuristic approach for automated planning as part of my master's thesis. The core contribution is a Markov Decision Process (MDP) based heuristic that leverages pattern projections to create informative estimates for guiding search algorithms.

## Overview

Automated planning is a crucial area in AI concerned with finding sequences of actions to achieve specific goals. Heuristic search is a dominant approach for solving planning problems, where heuristics provide guidance by estimating the distance to the goal.

This project:
1. Implements a new heuristic based on pattern projections and MDPs
2. Provides an efficient GPU-accelerated value iteration algorithm for solving MDPs
3. Compares the performance against established heuristics (PDB, hmax, LM-Cut)
4. Analyzes different pattern selection strategies and their impact on search performance

## Key Components

- **MDP Heuristic**: Creates pattern projections of the state space and solves them as MDPs
- **GPU-Accelerated Value Iteration**: Efficient implementation of value iteration using PyTorch for GPU acceleration
- **Automated Pattern Selection**: Algorithm to automatically identify interesting patterns based on causal graphs
- **Comparative Analysis**: Benchmarking framework for comparing different heuristics

## Repository Structure

- `planner.py`: Main planning algorithm using various heuristics
- `a_star_search.py`: Implementation of A* and Greedy Best-First Search (GBFS) algorithms
- `heuristics/`
  - `abstract_heuristic.py`: Implementation of the MDP-based heuristic
  - `pdb_heuristic.py`: Pattern Database heuristic implementation
  - `hmax_heuristic.py`: h^max heuristic implementation
  - `lmcut_heuristic.py`: Landmark-Cut heuristic implementation
- `sas_parser/`: Parser for SAS+ planning problem format
- `utils/`
  - `abstraction.py`: Functions for creating and manipulating abstractions
  - `gpu_value.py`: GPU-accelerated implementation of Value Iteration
  - `interesting_patterns.py`: Algorithm to identify useful pattern projections
- `scripts/`:
  - `benchmark.py`: Framework for benchmarking different heuristics and configurations
  - `plotter.py`: Script for generating comparative plots from benchmark results

## GPU-Accelerated Value Iteration

One of the highlights of this project is the efficient implementation of Value Iteration for solving MDPs on GPUs. The implementation:

- Uses PyTorch for GPU acceleration
- Efficiently handles sparse transition matrices
- Provides significant speed improvements over CPU-based implementations
- Makes it feasible to use MDPs for real-time heuristic guidance

## Installation

```bash
# Clone the repository
git clone https://github.com/levitvas/abstraction-planner.git
cd abstraction-planner

# Install dependencies
pip install numpy scipy torch matplotlib networkx
```

## Acknowledgments

This project was developed as part of my master's thesis at CTU. Special thanks to [Rostislav Horčík](https://www.aic.fel.cvut.cz/members/rostislav-horcik), as my thesis supervisor, for guidance and support.
