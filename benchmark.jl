# --- IPLP Benchmark Implementation ---
using LinearAlgebra, SparseArrays
using MatrixDepot
using Printf
using Statistics
using JuMP
using Clp
using DataFrames

include("iplp_solver.jl")

# --- Clp Comparison Function ---
function solve_with_clp(prob::IplpProblem, test_tol::Float64, max_iterations::Int)
    n_orig = length(prob.c)
    m_orig = size(prob.A, 1)
    model = Model(Clp.Optimizer)
    set_optimizer_attribute(model, "LogLevel", 0)
    set_optimizer_attribute(model, "MaximumIterations", 50000)
    set_optimizer_attribute(model, "PrimalTolerance", test_tol)
    set_optimizer_attribute(model, "DualTolerance", test_tol)

    # For lp_agg, use specific Clp settings that work better
    if m_orig > 300 && n_orig > 400
        nnz_count = nnz(prob.A)
        total_elements = m_orig * n_orig
        density = nnz_count / total_elements
    end
    
    try
        @variable(model, x[i=1:n_orig], lower_bound=prob.lo[i], upper_bound=prob.hi[i])
        @objective(model, Min, dot(prob.c, x))
        if m_orig > 0 && n_orig > 0
            if length(prob.b) == m_orig
                @constraint(model, con, prob.A * x .== prob.b)
            else
                @error "Dim mismatch: A $m_orig rows, b $(length(prob.b)) els."
                return (:Error, NaN, Float64[], 0.0)
            end
        elseif m_orig > 0 && n_orig == 0
             if all(abs.(prob.b) .< 1e-9)
                @warn "Problem has constraints but no variables."
                return (:Optimal, 0.0, Float64[], 0.0)
             else
                @warn "Problem constraints but no vars, b!=0."
                return (:Infeasible, NaN, Float64[], 0.0)
             end
        end
        
        timed_result = @timed optimize!(model)
        time_taken = timed_result.time
        status = termination_status(model)
        
        obj_val = NaN
        x_sol = fill(NaN, n_orig)
        
        try 
            obj_val = objective_value(model) 
        catch
            # Do nothing if objective retrieval fails
        end
        
        if primal_status(model) == FEASIBLE_POINT
            x_sol = value.(x)
        end
        
        simple_status = :Unknown
        if status == OPTIMAL
            simple_status = :Optimal
        elseif status == INFEASIBLE
            simple_status = :Infeasible
        elseif status == DUAL_INFEASIBLE
            simple_status = :Unbounded
        elseif status == TIME_LIMIT
            simple_status = :TimeLimit
        elseif status == ITERATION_LIMIT
            simple_status = :MaxIter
        else
            simple_status = status
        end
        
        if simple_status == :Infeasible
            obj_val = NaN
        end
        
        return simple_status, obj_val, x_sol, time_taken
    catch e
        @error "Clp solve error: $e"
        return (:Error, NaN, Float64[], 0.0)
    end
end

# --- Extract Solution Function (for handling standard form solution) ---
function extract_solution_from_standard_form(iplp_soln, iplp_prob_orig)
    try
        n_orig = length(iplp_prob_orig.c)
        
        # Check if xs has enough elements for extraction
        if length(iplp_soln.xs) < 2*n_orig
            @warn "Standard form solution too short for extraction"
            return nothing, :ExtractError
        end
        
        # Extract positive and negative parts from standard form
        x_p = iplp_soln.xs[1:n_orig]
        x_n = iplp_soln.xs[n_orig+1:2*n_orig]
        
        # Apply scaling back to original solution
        x_extracted = x_p - x_n
        
        # For challenging problems, further refine the solution
        if iplp_soln.flag
            # Attempt to improve the solution by ensuring feasibility
            ms = size(iplp_soln.As, 1)
            if ms > 0 && !isempty(iplp_soln.As) && !isempty(iplp_soln.bs)
                try
                    # Try to improve the solution quality
                    @info "Performing final refinement for extracted solution"
                    # Create standard form solution variables
                    xs_refined = copy(iplp_soln.xs)
                    
                    # Apply direct refinement step
                    AAt = iplp_soln.As * iplp_soln.As' + 1e-6 * sparse(I, ms, ms)
                    rhs_refine = iplp_soln.bs - iplp_soln.As * xs_refined
                    
                    # Use more robust solver
                    lam_correction = try
                        ldlt(Symmetric(AAt)) \ rhs_refine
                    catch
                        AAt \ rhs_refine
                    end
                    
                    xs_refined .+= iplp_soln.As' * lam_correction
                    
                    # Extract refined original solution
                    x_p_refined = xs_refined[1:n_orig]
                    x_n_refined = xs_refined[n_orig+1:2*n_orig]
                    x_extracted_refined = x_p_refined - x_n_refined
                    
                    # Check feasibility improvement
                    residual_before = norm(iplp_prob_orig.A * x_extracted - iplp_prob_orig.b, Inf)
                    residual_after = norm(iplp_prob_orig.A * x_extracted_refined - iplp_prob_orig.b, Inf)
                    
                    if residual_after < residual_before
                        @info "Refinement improved feasibility from $residual_before to $residual_after"
                        x_extracted = x_extracted_refined
                    else
                        @info "Refinement did not improve feasibility, using original extraction"
                    end
                catch e
                    @warn "Refinement attempt failed: $e, using original extraction"
                end
            end
        end
        
        # Check if this gives us a valid solution
        if all(isfinite.(x_extracted))
            return x_extracted, :Success
        else
            @warn "Extracted solution has non-finite values"
            return nothing, :NonFinite
        end
    catch e
        @warn "Error extracting solution: $e"
        return nothing, :ExtractError
    end
end

# --- Main Benchmark Script ---
function run_benchmark()
    problem_names = ["lp_afiro", "lp_brandy", "lp_fit1d", "lp_adlittle", "lp_agg", "lp_ganges", "lp_stocfor1", "lp_25fv47", "lpi_chemcom"]
    test_tol = 1e-8
    max_iterations = 100  # Maximum iterations for challenging problems
    results = []
    
    println("\n" * "="^40)
    println(" Starting Benchmark Run (IPLP Solver) ")
    println("="^40 * "\n")
    
    for name in problem_names
        println("--- Processing Problem: $name ---")
        local md_prob
        local iplp_prob_orig
        
        # Problem-specific settings can still be implemented based on testing needs
        problem_specific_maxit = max_iterations
        
        try
            md_prob = mdopen("LPnetlib/$name")
        catch e
            println(" ERROR loading '$name': $e")
            push!(results, (
                Problem=name, 
                IPLP_Time=NaN, 
                IPLP_Status=:LoadError, 
                IPLP_Obj=NaN,
                IPLP_Feasible=false,
                CLP_Time=NaN, 
                CLP_Status=:LoadError, 
                CLP_Obj=NaN, 
                Obj_Diff_Abs=NaN
            ))
            continue
        end
        
        local iplp_prob_for_solver
        try
            iplp_prob_for_solver = convert_matrixdepot(md_prob)
            iplp_prob_orig = deepcopy(iplp_prob_for_solver)
        catch e
            println(" ERROR converting '$name': $e")
            push!(results, (
                Problem=name, 
                IPLP_Time=NaN, 
                IPLP_Status=:ConvertError, 
                IPLP_Obj=NaN,
                IPLP_Feasible=false,
                CLP_Time=NaN, 
                CLP_Status=:ConvertError, 
                CLP_Obj=NaN, 
                Obj_Diff_Abs=NaN
            ))
            continue
        end
        
        if isempty(iplp_prob_orig.c) && size(iplp_prob_orig.A) == (0,0)
            println(" Skipping empty: '$name'.")
            push!(results, (
                Problem=name, 
                IPLP_Time=0.0, 
                IPLP_Status=:Empty, 
                IPLP_Obj=0.0,
                IPLP_Feasible=true,
                CLP_Time=0.0, 
                CLP_Status=:Empty, 
                CLP_Obj=0.0, 
                Obj_Diff_Abs=0.0
            ))
            continue
        end
        
        println(" Solving with IPLP...")
        iplp_timed = @timed iplp(iplp_prob_for_solver, test_tol; maxit=problem_specific_maxit)
        iplp_soln = iplp_timed.value
        iplp_time = iplp_timed.time
        iplp_obj = NaN
        iplp_status = iplp_soln.flag ? :Converged : :Failed
        iplp_feasible = iplp_soln.feas
        
        # Print more detailed info about the solution
        println("  IPLP Solution Status: $(iplp_soln.flag ? "Converged" : "Failed")")
        println("  IPLP Feasibility: $(iplp_soln.feas ? "Feasible" : "Infeasible")")
        
        if iplp_soln.flag && !isempty(iplp_soln.x) && length(iplp_soln.x) == length(iplp_prob_orig.c)
            try
                iplp_obj = dot(iplp_prob_orig.c, iplp_soln.x)
                if !isfinite(iplp_obj)
                    @warn "IPLP objective NaN/Inf: $name."
                    iplp_obj = NaN
                    iplp_status = :NumericalError
                end
            catch e
                @warn "Error calc objective for $name: $e"
                iplp_obj = NaN
                iplp_status = :NumericalError
            end
        elseif iplp_soln.flag
            # Try to extract solution from standard form if original form solution is missing
            if !isempty(iplp_soln.xs) && length(iplp_soln.xs) >= 2*length(iplp_prob_orig.c)
                x_extracted, extract_status = extract_solution_from_standard_form(iplp_soln, iplp_prob_orig)
                
                if extract_status == :Success
                    try
                        # Check if extracted solution is valid
                        iplp_obj = dot(iplp_prob_orig.c, x_extracted)
                        if isfinite(iplp_obj)
                            @info "Successfully extracted solution from standard form."
                            iplp_status = :Converged
                            # Update solution in iplp_soln
                            iplp_soln.x = x_extracted
                        else
                            @warn "Extracted solution has invalid objective"
                            iplp_obj = NaN
                            iplp_status = :NumericalError
                        end
                    catch e
                        @warn "Error computing objective with extracted solution: $e"
                        iplp_obj = NaN
                        iplp_status = :NumericalError
                    end
                else
                    @warn "Solution extraction failed: $extract_status"
                    iplp_obj = NaN
                    iplp_status = :NumericalError
                end
            else
                @warn "IPLP flag=true but x mismatch: $name."
                iplp_status = :NumericalError
            end
        end
        
        println("  IPLP Done. Time: $(round(iplp_time; digits=3))s, Status: $iplp_status")
        
        println(" Solving with Clp (using original problem)...")
        clp_status, clp_obj, clp_x_sol, clp_time = solve_with_clp(iplp_prob_orig, test_tol, max_iterations)
        println("  Clp Done. Time: $(round(clp_time; digits=3))s, Status: $clp_status")
        
        # Calculate absolute difference instead of relative difference
        obj_diff_abs = NaN
        if iplp_status == :Converged && clp_status == :Optimal && isfinite(iplp_obj) && isfinite(clp_obj)
            obj_diff_abs = abs(iplp_obj - clp_obj)
        elseif (iplp_status == :Failed && clp_status == :Infeasible) || 
               (iplp_status == :Failed && clp_status == :Unbounded)
            obj_diff_abs = 0.0
        elseif !(iplp_status in [:Converged, :Failed]) && 
               !(clp_status in [:Optimal, :Infeasible, :Unbounded, :Error, :LoadError, :ConvertError])
            obj_diff_abs = 0.0
        end
        
        push!(results, (
            Problem=name, 
            IPLP_Time=iplp_time, 
            IPLP_Status=iplp_status, 
            IPLP_Obj=iplp_obj,
            IPLP_Feasible=iplp_feasible,
            CLP_Time=clp_time, 
            CLP_Status=clp_status, 
            CLP_Obj=clp_obj, 
            Obj_Diff_Abs=obj_diff_abs
        ))
        
        println("---------------------------------")
    end
    
    println("\n" * "="^40)
    println(" Benchmark Run Complete ")
    println("="^40 * "\n")
    
    results_df = DataFrame(results)
    if !isempty(results_df)
        results_df_display = deepcopy(results_df)
        results_df_display[!, :CLP_Status] = map(results_df_display[!, :CLP_Status]) do status
            status == :Optimal ? :Converged : status
        end
        
        results_df_display[!, :IPLP_Time] = round.(results_df_display[!, :IPLP_Time]; digits=3)
        results_df_display[!, :IPLP_Obj] = round.(coalesce.(results_df_display[!, :IPLP_Obj], NaN); sigdigits=7)
        results_df_display[!, :CLP_Time] = round.(results_df_display[!, :CLP_Time]; digits=3)
        results_df_display[!, :CLP_Obj] = round.(coalesce.(results_df_display[!, :CLP_Obj], NaN); sigdigits=7)
        results_df_display[!, :Obj_Diff_Abs] = round.(coalesce.(results_df_display[!, :Obj_Diff_Abs], NaN); sigdigits=7)
        
        println(results_df_display)
        
        println("\nProblems with significant objective difference (> $(10*test_tol)):")
        diff_check = filter(row -> 
            row.IPLP_Status == :Converged && 
            row.CLP_Status == :Converged && 
            !isnan(row.Obj_Diff_Abs) && 
            row.Obj_Diff_Abs > 10*test_tol, 
            results_df)
            
        if isempty(diff_check)
            println(" None.")
        else
            println(diff_check)
        end
        
        println("\nIPLP Failures or Disagreements:")
        fail_check = filter(results_df) do row
            iplp_s = row.IPLP_Status
            clp_s = row.CLP_Status
            
            !(iplp_s == :Converged && clp_s == :Optimal) && 
            !(iplp_s == :Failed && clp_s == :Infeasible) && 
            !(iplp_s == :Failed && clp_s == :Unbounded) && 
            !(iplp_s in [:LoadError, :ConvertError]) && 
            !(clp_s in [:LoadError, :ConvertError, :Error]) && 
            !(iplp_s == :Empty && clp_s == :Empty)
        end
        
        if isempty(fail_check)
            println(" None.")
        else
            println(fail_check[!, [:Problem, :IPLP_Status, :IPLP_Feasible, :CLP_Status]])
        end
        
        # Additional report on feasibility detection
        println("\nFeasibility vs Clp Status:")
        feas_check = select(results_df, [:Problem, :IPLP_Feasible, :CLP_Status])
        println(feas_check)
    else
        println("No results generated.")
    end
    
    return results_df
end

