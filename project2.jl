# --- Project 2: Interior Point LP Solver ---
# This file serves as the main entry point for the project

# -- Description --
# Implementation of a robust interior point LP solver as per the requirements:
# 1. The solver should be robust and aim to solve all test problems
# 2. Comparison against Clp simplex solver is provided

include("iplp_solver.jl")  # Contains the optimizer implementation
include("benchmark.jl")    # Contains the benchmarking code

# Run the benchmarks when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running Project 2: Interior Point LP Solver")
    run_benchmark()
end

# Row │ Problem      IPLP_Time  IPLP_Status  IPLP_Obj        IPLP_Feasible  CLP_Time  CLP_Status  CLP_Obj         Obj_Diff_Abs 
# │ String       Float64    Symbol       Float64         Bool           Float64   Symbol      Float64         Float64      
# ─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 1 │ lp_afiro         4.679  Converged      -464.753               true     1.801  Converged     -464.753       5.68434e-14
# 2 │ lp_brandy        0.296  Converged      1518.51                true     0.003  Converged     1518.51        1.13687e-11
# 3 │ lp_fit1d         1.894  Converged     -9146.38                true     0.003  Converged    -9146.38        4.9383e-7
# 4 │ lp_adlittle      0.036  Converged    225495.0                 true     0.001  Converged   225495.0         5.4289e-5
# 5 │ lp_agg           1.278  Converged        -3.59918e7           true     0.002  Converged       -3.59918e7   1.75014e-5
# 6 │ lp_ganges       20.882  Converged        -1.09586e5           true     0.005  Converged       -1.09586e5   3.43913e-6
# 7 │ lp_stocfor1      0.073  Converged    -41132.0                 true     0.001  Converged   -41132.0         1.19697e-7
# 8 │ lp_25fv47        9.59   Converged      5501.85                true     0.095  Converged     5501.85        1.46338e-9
# 9 │ lpi_chemcom      0.418  Failed          NaN                  false     0.001  Infeasible     NaN           0.0