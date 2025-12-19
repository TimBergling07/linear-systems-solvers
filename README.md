# Numerical Methods for Non-Homogeneous Linear Systems

This repository contains the source code and numerical experiments for the course paper:  
**"A Summary of Methods for Solving Non-Homogeneous Linear Systems"**.

All core computations are implemented in **C++ using the Eigen library** (included). Python scripts are used **only for data visualization**. The code is organized by chapter to match the paper’s structure.
、、、
## 📁 Repository Structure
├── chapter4/
│   ├── chapter4.cpp                 # C++ implementation of direct methods (LU)
│   ├── chapter4.exe                 # Compiled executable (Windows)
│   ├── chapter4_performance.png     # Performance plot
│   └── plot_ch4.py                  # Python script to generate plot
│
├── chapter5/
│   ├── chapter5.cpp                 # C++ implementation of QR and SVD
│   ├── chapter5.exe                 # Compiled executable (Windows)
│   ├── chapter5_performance.png     # Performance plot
│   └── plot_ch5.py                  # Python script to generate plot
│
├── chapter6/
│   ├── chapter6.cpp                 # C++ implementation for ill-conditioned systems
│   ├── chapter6.exe                 # Compiled executable (Windows)
│   ├── chapter6.tex                 # LaTeX figure source (optional)
│   ├── normal_eq_error_real.csv     # Raw error data from Normal Equations test
│   ├── normal_eq_error_real.png     # Final error vs. condition number plot
│   ├── normal_eq_error_vs_sigma.py  # Python script to compute errors
│   └── plot_normal_eq_error_real.py # Python script to generate final plot
│
├── eigen/                           # Full Eigen header-only library (v3.x)
├── LICENSE                          # MIT License
└── README.md                        # This file
、、、
## ▶️ How to Run

### 1. Compile and Run C++ Code
You only need a C++ compiler (e.g., `g++`, `clang++`, or MSVC) — **no external dependencies** since Eigen is included.

#### On Windows (using MinGW or WSL):
```bash
g++ -O2 -I ./eigen chapter4/chapter4.cpp -o chapter4/chapter4.exe
./chapter4/chapter4.exe
On Linux/macOS:
g++ -O2 -I ./eigen chapter4/chapter4.cpp -o chapter4/chapter4
./chapter4/chapter4
Repeat the same steps for chapter5/chapter5.cpp and chapter6/chapter6.cpp.

Each program will generate output files (e.g., .csv, .png) in its respective folder.

2. Generate Plots with Python
Install required Python packages:
pip install numpy matplotlib pandas
Run plotting scripts:
python chapter4/plot_ch4.py
python chapter5/plot_ch5.py
python chapter6/plot_normal_eq_error_real.py
These scripts read data from .csv files and produce publication-ready PNG figures.

🧰 Dependencies
C++ Compiler: Any modern C++11-compatible compiler
Eigen: Header-only library (already included in /eigen)
Python: 3.6+ with numpy, matplotlib, and pandas
📝 Notes
All numerical results and figures in the paper were generated using these exact scripts.
The eigen/ directory contains the complete Eigen library — no system installation needed.
Executables (.exe) are provided for convenience but can be safely regenerated.