using MatrixDepot
using Test
using Printf
using SparseArrays
using LinearAlgebra
using Statistics


# =========================================================================
# Linear System Solver Functions
# =========================================================================

"""
    augmented_system_solve(A, x, s, rb, rc, rxs; refinement_sweeps=2)

Factorize and solve the augmented system for the predictor-corrector method
with iterative refinement to improve accuracy.

# Arguments
- `A`: Constraint matrix
- `x`: Current primal variables
- `s`: Current dual slack variables
- `rb`: Primal residual
- `rc`: Dual residual
- `rxs`: Complementarity residual
- `refinement_sweeps`: Number of iterative refinement steps (default: 2)
"""
function augmented_system_solve(A, x, s, rb, rc, rxs; refinement_sweeps=2)
    m, n = size(A)

    # Build augmented system matrix
    M = [spzeros(m, m) A spzeros(m, n);
         A' spzeros(n, n) Matrix{Float64}(I, n, n);
         spzeros(n, m) spdiagm(0=>s[:,1]) spdiagm(0=>x[:,1])]

    # Compute right-hand side vector
    rhs = Array{Float64}([-rb; -rc; -rxs])
    
    # Try to factorize
    try
        f = lu(M)
        
        # Initial solve with factorization
        solution = f\rhs
        
        # Perform iterative refinement
        for i in 1:refinement_sweeps
            # Convert to higher precision for residual calculation
            M_big = BigFloat.(M)
            solution_big = BigFloat.(solution)
            rhs_big = BigFloat.(rhs)
            
            # Compute residual r = M*solution - rhs in higher precision
            residual_big = M_big * solution_big - rhs_big
            residual = Float64.(residual_big)  # Convert back to Float64
            
            # Solve correction system M*delta = -residual
            delta = f\(-residual)
            
            # Update solution
            solution += delta
            
            # Optional: Check if refinement is still improving the solution
            if norm(delta) < 1e-14 * norm(solution)
                # @printf("Iterative refinement converged after %d sweeps\n", i)
                break
            end
        end
        
        # Extract solution components
        dlam = solution[1:m]
        dx = solution[1+m:m+n]
        ds = solution[1+m+n:m+2*n]
        
        return dlam, dx, ds, true
    catch e
        if isa(e, SingularException)
            @warn "Matrix is singular during factorization. This often indicates an infeasible problem."
            return zeros(m), zeros(n), zeros(n), false
        else
            rethrow(e)
        end
    end
end

# =========================================================================
# Step Size Computation
# =========================================================================

"""
    compute_max_step(x, dx, hi)

Compute the maximum step size that maintains non-negativity.

# Arguments
- `x`: Current point
- `dx`: Search direction
- `hi`: Upper bound on step size
"""
function compute_max_step(x, dx, hi)
    n = length(x)
    alpha = -1.0

    for i=1:n
        if dx[i] < 0
            a = -x[i]/dx[i]

            if alpha < 0
                alpha = a
            else
                alpha = min(alpha, a)
            end
        end
    end

    if alpha < 0
        alpha = Inf
    end

    alpha = min(alpha, hi)

    return alpha
end

# =========================================================================
# Initialization Function
# =========================================================================

"""
    initialize_interior_point(A, b, c)

Generate an initial point for the interior point method.
Based on Wright's "Primal-Dual Interior-Point Methods".
"""
function initialize_interior_point(A, b, c)
    AA = A*A'
    
    try
        f = cholesky(AA)
        
        # tilde
        x = f\b
        x = A'*x

        lambda = A*c
        lambda = f\lambda

        s = A'*lambda
        s = c-s

        # hat
        dx = max(-1.5*minimum(x), 0.0)
        ds = max(-1.5*minimum(s), 0.0)

        x = x.+dx
        s = s.+ds

        # ^0
        xs = dot(x, s)/2.0

        dx = xs/sum(s)
        ds = xs/sum(x)

        x = x.+dx
        s = s.+ds

        return x, lambda, s
    catch e
        if isa(e, PosDefException) || isa(e, SingularException)
            @warn "Matrix AA = A*A' is not positive definite during starting point calculation. May indicate an infeasible problem."
            m, n = size(A)
            return ones(n), zeros(m), ones(n)  # Return safe dummy values
        else
            rethrow(e)
        end
    end
end

# =========================================================================
# Standard Form Conversion
# =========================================================================

"""
    to_standard_form(P)

Convert the LP problem to standard form with only non-negative variables.

# Arguments
- `P`: Original LP problem
"""
function to_standard_form(P)
    inf = 1.0e300

    m, n0 = size(P.A)

    index1 = zeros(Int64, 0)  # Free variables: -∞ < x < ∞
    index2 = zeros(Int64, 0)  # Lower bounded: lo < x < ∞
    index3 = zeros(Int64, 0)  # Upper bounded: -∞ < x < hi
    index4 = zeros(Int64, 0)  # Box constrained: lo < x < hi
    n = zeros(Int64, 4)

    # Classify variables by bound type
    for i=1:n0
        if P.lo[i] < -inf
            if P.hi[i] > inf
                n[1] += 1
                index1 = [index1; i]  # Free
            else
                n[3] += 1
                index3 = [index3; i]  # Upper bounded
            end
        else
            if P.hi[i] > inf
                n[2] += 1
                index2 = [index2; i]  # Lower bounded
            else
                n[4] += 1
                index4 = [index4; i]  # Box constrained
            end
        end
    end

    # Build the standard form objective
    cs = [P.c[index1, 1]; -P.c[index1, 1]; P.c[index2, 1];
          -P.c[index3, 1]; P.c[index4, 1]; zeros(n[4], 1)]

    # Build the standard form constraint matrix
    As = [P.A[:, index1] -P.A[:, index1] P.A[:, index2] -P.A[:, index3] P.A[:, index4] spzeros(m, n[4]);
          spzeros(n[4], 2*n[1]+n[2]+n[3]) Matrix{Float64}(I, n[4], n[4]) Matrix{Float64}(I, n[4], n[4])]

    # Build the standard form right-hand side
    bs = [P.b-P.A[:, index2]*P.lo[index2, 1]-P.A[:, index3]*P.hi[index3, 1]-P.A[:, index4]*P.lo[index4, 1];
          P.hi[index4, 1]-P.lo[index4, 1]]

    return As, bs, cs, index1, index2, index3, index4
end

"""
    from_standard_form(P, ind1, ind2, ind3, ind4, xs)

Convert solution from standard form back to original variables.
"""
function from_standard_form(P, ind1, ind2, ind3, ind4, xs)
    m, n = size(P.A)
    n1 = length(ind1)
    n2 = length(ind2)
    n3 = length(ind3)
    n4 = length(ind4)

    x = zeros(n)

    x[ind1] = xs[1:n1]-xs[1+n1:2*n1]                                # Free variables
    x[ind2] = xs[1+2*n1:n2+2*n1]+P.lo[ind2]                         # Lower bounded
    x[ind3] = P.hi[ind3]-xs[1+2*n1+n2:n3+2*n1+n2]                   # Upper bounded
    x[ind4] = xs[1+2*n1+n2+n3:n4+2*n1+n2+n3]+P.lo[ind4]             # Box constrained

    return x
end

# =========================================================================
# Preprocessing Functions
# =========================================================================

"""
    preprocess_problem(P)

Preprocess the LP problem to identify redundancies and simplify the problem.
"""
function preprocess_problem(P)
    m, n = size(P.A)
    ind0r = zeros(Int64, 0)  # Zero rows
    ind0c = zeros(Int64, 0)  # Zero columns
    ind_dup_r = zeros(Int64, 0)  # Duplicate rows
    ind_dup_c = zeros(Int64, 0)  # Duplicate columns
    dup_main_c = Array[]  # Main columns to keep for duplicate sets
    
    # Find zero rows
    for i = 1:m
        j = 1
        while (j <= n) && (P.A[i, j] == 0.0)
            j += 1
        end

        if j == n+1  # Row is all zeros
            if P.b[i] == 0.0
                ind0r = [ind0r; i]
            else
                @warn "This problem is infeasible."
                return false
            end
        end
    end
    
    # Find zero columns
    for j = 1:n
        i = 1
        while (i <= m) && (P.A[i, j] == 0.0)
            i += 1
        end

        if i == m+1  # Column is all zeros
            ind0c = [ind0c; j]
        end
    end

    # Find duplicate rows
    for i = 1:(m-1)
        if (i in ind_dup_r) || (i in ind0r)
            continue
        end

        for j = (i+1):m
            if (j in ind_dup_r) || (j in ind0r)
                continue
            end

            k = 1
            while (k <= n) && (P.A[i, k] == P.A[j, k])
                k += 1
            end

            if k == n+1  # Rows are identical
                if P.b[i] == P.b[j]
                    ind_dup_r = [ind_dup_r; j]
                else
                    @warn "This problem is infeasible."
                    return false
                end
            end
        end
    end
    
    remove_ind_row = [ind0r; ind_dup_r]
    
    # Find duplicate columns
    for i = 1:(n-1)
        dup_item = Array(i:i)
        for j = (i+1):n
            if !(j in ind_dup_c) && !(j in ind0c) && (P.A[:, i] == P.A[:, j])
                dup_item = [dup_item; j]
            end
        end
        if length(dup_item) > 1
            minv, mini = findmin(P.c[dup_item])
            dup_flags = trues(length(dup_item))
            dup_flags[mini] = false
            dup_main_c = [dup_main_c; dup_item[mini]]
            ind_dup_c = [ind_dup_c; dup_item[dup_flags]]
        end
    end
    
    remove_ind_col = [ind0c; ind_dup_c]
    
    # Create masks for rows/columns to keep
    flags_row = trues(m)
    flags_col = trues(n)
    flags_row[remove_ind_row] .= false
    flags_col[remove_ind_col] .= false
    
    # Create reduced problem
    reduced_prob = IplpProblem(
        P.c[flags_col], 
        P.A[flags_row, flags_col],
        P.b[flags_row], 
        P.lo[flags_col], 
        P.hi[flags_col]
    )
    
    return reduced_prob, ind0c, dup_main_c, ind_dup_c
end

"""
    reconstruct_solution(P, ind0c, dup_main_c, ind_dup_c, x1)

Recover the solution for the original problem from the preprocessed solution.
"""
function reconstruct_solution(P, ind0c, dup_main_c, ind_dup_c, x1)
    m, n = size(P.A)
    x = Array{Float64}(undef, n)
    fill!(x, Inf)
    j = 1
    
    for i = 1:n
        if x[i] == Inf
            if i in ind0c  # Zero column
                if P.c[i] > 0
                    x[i] = P.lo[i]  # Set to lower bound
                elseif P.c[i] < 0
                    x[i] = P.hi[i]  # Set to upper bound
                else
                    x[i] = 0.       # Set to zero
                end
            elseif i in dup_main_c  # Main column of duplicate set
                x[i] = x1[j]
                j += 1
            elseif i in ind_dup_c  # Duplicate column (not main)
                x[i] = 0.
            else  # Regular column
                x[i] = x1[j]
                j += 1
            end
        end
    end
    
    # Only run this check if we have a valid solution
    if length(x1) > 0 && !isnothing(x1) && any(isfinite.(x1))
        @test j == length(x1) + 1
    end
    
    return x
end

# =========================================================================
# Feasibility Check
# =========================================================================

"""
    check_feasibility(A, b; refinement_sweeps=2)

Phase I to check the feasibility of the linear program.
"""
function check_feasibility(A, b; refinement_sweeps=2)
    m, n = size(A)
    
    # Form Phase I problem
    A_phase1 = [A Matrix{Float64}(I, m, m)]
    c_phase1 = [zeros(Float64, n); ones(Float64, m)]
    
    # Solve Phase I problem with iterative refinement
    x1, lambda1, s1, flag, iter = solve_predictor_corrector(A_phase1, b, c_phase1, 100, 1e-8, false, refinement_sweeps)
    
    # Calculate feasibility measure
    feasibility_measure = dot(c_phase1, x1)
    @printf("Phase I feasibility measure: %.2e\n", feasibility_measure)
    
    # Determine if the problem is infeasible
    if !flag || feasibility_measure > 1.0
        # Problem is genuinely infeasible
        return true
    elseif feasibility_measure > 1e-7
        # Problem is marginally infeasible but may be solvable
        @warn "Problem is marginally infeasible (measure = $feasibility_measure) but will attempt to solve."
        return false
    else
        # Problem is feasible
        return false
    end
end

# =========================================================================
# Main Solver Function
# =========================================================================

"""
    solve_predictor_corrector(A, b, c, maxit=100, tol=1e-8, verbose=false, refinement_sweeps=2)

Solve a linear program in standard form using a predictor-corrector interior point method
with iterative refinement.
"""
function solve_predictor_corrector(A, b, c, maxit=100, tol=1e-8, verbose=false, refinement_sweeps=2)
    # Algorithm parameters
    gamma_f = .01  # Step size control parameter
    scaling = 1    # Whether adaptive scaling is used

    m, n = size(A)

    # Initialize point
    x0, lambda0, s0 = initialize_interior_point(A, b, c)

    iter = 0

    if verbose
        @printf("%3d %9.2e %9.2e %9.4g %9.4g\n", 
                iter, mean(x0.*s0), 
                norm([A'*lambda0 + s0 - c; A*x0 - b; x0.*s0])/norm([b;c]), 
                0., 0.)
    end

    # Initialize result values
    x1, lambda1, s1 = x0, lambda0, s0
    flag = false
    
    for iter = 1:maxit
        # --- Predictor step ---
        
        # Compute residuals
        rb  = A*x0-b          # Primal feasibility
        rc  = A'*lambda0+s0-c # Dual feasibility
        rxs = x0.*s0          # Complementarity
        
        # Solve augmented system for affine direction with iterative refinement
        lambda_aff, x_aff, s_aff, factorization_ok = augmented_system_solve(A, x0, s0, rb, rc, rxs; 
                                                                           refinement_sweeps=refinement_sweeps)
        
        # Check if factorization failed
        if !factorization_ok
            @warn "Factorization failed at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # Calculate step sizes
        alpha_aff_pri  = compute_max_step(x0, x_aff, 1.0)
        alpha_aff_dual = compute_max_step(s0, s_aff, 1.0)
        
        # Check for numerical issues
        if !isfinite(alpha_aff_pri) || !isfinite(alpha_aff_dual)
            @warn "Non-finite step size at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # Compute duality measure
        mu = mean(rxs)
        mu_aff = dot(x0+alpha_aff_pri*x_aff, s0+alpha_aff_dual*s_aff)/n
        
        # Check for numerical issues
        if !isfinite(mu_aff)
            @warn "Non-finite mu_aff at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # --- Corrector step ---
        
        # Compute centering parameter
        sigma = (mu_aff/mu)^3
        
        # Check for numerical issues
        if !isfinite(sigma)
            @warn "Non-finite sigma at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end
        
        # Compute corrector right-hand side
        rb_cor = spzeros(m)
        rc_cor = spzeros(n)
        rxs_cor = x_aff.*s_aff.-sigma*mu
        
        # Solve for corrector direction with iterative refinement
        lambda_cc, x_cc, s_cc, _ = augmented_system_solve(A, x0, s0, rb_cor, rc_cor, rxs_cor;
                                                         refinement_sweeps=refinement_sweeps)

        # Combine predictor and corrector directions
        dx = x_aff+x_cc
        dlambda = lambda_aff+lambda_cc
        ds = s_aff+s_cc
        
        # Check for numerical issues
        if any(.!isfinite.(dx)) || any(.!isfinite.(dlambda)) || any(.!isfinite.(ds))
            @warn "Non-finite search direction at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # --- Step size computation ---
        
        # Find maximum feasible step sizes
        alpha_max_pri = compute_max_step(x0, dx, Inf)
        alpha_max_dual = compute_max_step(s0, ds, Inf)
        
        # Check for numerical issues
        if !isfinite(alpha_max_pri) || !isfinite(alpha_max_dual)
            @warn "Non-finite max step size at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end
        
        # Determine step sizes using heuristics
        if scaling == 0
            # Simple fraction of max step
            alpha_pri = min(0.99*alpha_max_pri, 1)
            alpha_dual = min(0.99*alpha_max_dual, 1)
        else
            # Adaptive scaling based on complementarity
            x1_pri = x0+alpha_max_pri*dx
            s1_dual = s0+alpha_max_dual*ds
            
            # Check for numerical issues
            if any(.!isfinite.(x1_pri)) || any(.!isfinite.(s1_dual))
                @warn "Non-finite boundary point at iteration $iter. Terminating solver."
                return x1, lambda1, s1, false, iter
            end
            
            # Estimate duality measure at boundary
            mu_p = dot(x1_pri, s1_dual)/n
            
            # Check for numerical issues
            if !isfinite(mu_p)
                @warn "Non-finite mu_p at iteration $iter. Terminating solver."
                return x1, lambda1, s1, false, iter
            end

            # Compute refined step sizes
            xind = argmin(x1_pri)
            # Check for division by zero
            if abs(dx[xind]) < 1e-12 || !isfinite(s1_dual[xind]) || s1_dual[xind] < 1e-12
                f_pri = 0.0
            else
                f_pri = (gamma_f*mu_p/s1_dual[xind]-x0[xind])/alpha_max_pri/dx[xind]
            end
            
            sind = argmin(s1_dual)
            # Check for division by zero
            if abs(ds[sind]) < 1e-12 || !isfinite(x1_pri[sind]) || x1_pri[sind] < 1e-12
                f_dual = 0.0
            else
                f_dual = (gamma_f*mu_p/x1_pri[sind]-s0[sind])/alpha_max_dual/ds[sind]
            end

            alpha_pri = max(1-gamma_f, f_pri)*alpha_max_pri
            alpha_dual = max(1-gamma_f, f_dual)*alpha_max_dual
        end

        # Check for numerical issues
        if !isfinite(alpha_pri) || !isfinite(alpha_dual)
            @warn "Non-finite step size at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # Check for unboundedness
        if alpha_pri > 1e308 || alpha_dual > 1e308
            @warn "This problem is unbounded"
            return x1, lambda1, s1, false, iter
        end

        # --- Update solution ---
        x1 = x0+alpha_pri*dx
        lambda1 = lambda0+alpha_dual*dlambda
        s1 = s0+alpha_dual*ds
        
        # Check for numerical issues
        if any(.!isfinite.(x1)) || any(.!isfinite.(lambda1)) || any(.!isfinite.(s1))
            @warn "Non-finite solution at iteration $iter. Terminating solver."
            return x1, lambda1, s1, false, iter
        end

        # Print iteration information
        if verbose
            @printf("%3d %9.2e %9.2e %9.4g %9.4g\n",
                    iter, mu,
                    norm([A'*lambda0 + s0 - c; A*x0 - b; x0.*s0])/norm([b;c]), 
                    alpha_pri, alpha_dual);
        end

        # --- Check convergence ---
        r1 = norm(A*x1-b)/(1+norm(b))

        if r1 < tol
            r2 = norm(A'*lambda1+s1-c)/(1+norm(c))

            if r2 < tol
                cx = dot(c, x1)
                r3 = abs(cx-dot(b, lambda1))/(1+abs(cx))

                if r3 < tol
                    flag = true
                    break
                end
            end
        end

        # Check if maximum iterations reached
        if iter == maxit
            flag = false
            break
        end

        # --- Prepare for next iteration ---
        x0 = x1
        lambda0 = lambda1
        s0 = s1
    end

    return x1, lambda1, s1, flag, iter
end

# =========================================================================
# Main Data Structures and Interface
# =========================================================================

"""
    IplpSolution

Solution structure for the interior point LP solver.
"""
struct IplpSolution
    x::Vector{Float64}      # the solution vector 
    flag::Bool              # a true/false flag indicating convergence or not
    cs::Vector{Float64}     # the objective vector in standard form
    As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
    bs::Vector{Float64}     # the right hand side (b) in standard form
    xs::Vector{Float64}     # the solution in standard form
    lam::Vector{Float64}    # the solution lambda in standard form
    s::Vector{Float64}      # the solution s in standard form
    feas::Bool              # whether the problem is feasible
end  

"""
    IplpProblem

Problem structure for the interior point LP solver.
"""
struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64} 
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

"""
    convert_matrixdepot(mmmeta)

Convert a MatrixDepot problem to the IplpProblem format.
"""
function convert_matrixdepot(mmmeta)
    return IplpProblem(
        vec(mmmeta.c),
        mmmeta.A,
        vec(mmmeta.b),
        vec(mmmeta.lo),
        vec(mmmeta.hi))
end

""" 
    iplp(Problem, tol; maxit=100, refinement_sweeps=2)

Solve a linear programming problem using interior point methods with iterative refinement.

# Arguments
- `Problem`: An IplpProblem structure containing the LP problem data
- `tol`: Convergence tolerance
- `maxit`: Maximum number of iterations (default: 100)
- `refinement_sweeps`: Number of iterative refinement steps (default: 2)

# Returns
- An IplpSolution structure containing the solution and related information
"""
function iplp(Problem, tol; maxit=100, refinement_sweeps=1)
    # Check input dimensions
    @show m0, n0 = size(Problem.A)
    
    if length(Problem.b) != m0 || length(Problem.c) != n0 || 
       length(Problem.lo) != n0 || length(Problem.hi) != n0
        DimensionMismatch("Dimension of matrices A, b, c mismatch. Check your input.")
    end

    @printf("Problem size: %d, %d\n", m0, n0)

    # Preprocess the problem
    preprocess_result = preprocess_problem(Problem)
    
    # Check if preprocess detected infeasibility
    if preprocess_result isa Bool
        @warn "This problem is infeasible (detected in preprocessing)."
        return IplpSolution(vec([0.]), false, zeros(Float64, 0), spzeros(0, 0), 
                            zeros(Float64, 0), zeros(Float64, 0), 
                            zeros(Float64, 0), zeros(Float64, 0), false)
    end
    
    # Unpack preprocessing result
    Ps, ind0c, dup_main_c, ind_dup_c = preprocess_result

    # Convert to standard form
    @show size(Ps.A)
    @show rank(Array{Float64}(Ps.A))
    
    A, b, c, ind1, ind2, ind3, ind4 = to_standard_form(Ps)
    @show size(A)
    @show rank(Array{Float64}(A))
    
    # Check feasibility
    is_infeasible = check_feasibility(A, b; refinement_sweeps=refinement_sweeps)
    
    if is_infeasible
        @warn "This problem is infeasible."
        # Return a solution with flag = false and feas = false
        return IplpSolution(vec([0.]), false, vec(c), A, vec(b), vec([0.]), vec([0.]), vec([0.]), false)
    end

    # Solve the standard form problem

    x1, lambda1, s1, flag, iter = solve_predictor_corrector(A, b, c, maxit, tol, false, refinement_sweeps)

    # Post-process the solution if valid
    if flag && length(x1) > 0 && all(isfinite.(x1))
        x_from_std = from_standard_form(Ps, ind1, ind2, ind3, ind4, x1)
        x = reconstruct_solution(Problem, ind0c, dup_main_c, ind_dup_c, x_from_std)
        
        if flag == true
            @printf("This problem is solved with optimal value of %.2f.\n\n", dot(Problem.c, x))
        else
            @printf("\nThis problem does not converge in %d steps.", maxit)
        end
        
        return IplpSolution(vec(x), flag, vec(c), A, vec(b), vec(x1), vec(lambda1), vec(s1), true)
    else
        @printf("\nThis problem could not be solved successfully.\n")
        # Return default solution with appropriate flags
        return IplpSolution(vec([0.]), false, vec(c), A, vec(b), 
                            flag && length(x1) > 0 ? vec(x1) : vec([0.]), 
                            flag && length(lambda1) > 0 ? vec(lambda1) : vec([0.]), 
                            flag && length(s1) > 0 ? vec(s1) : vec([0.]), 
                            !is_infeasible)
    end
end