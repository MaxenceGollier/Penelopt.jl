using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra

"""
    certify_local_infeasibility(name; kwargs...)

Certify whether the CUTEst problem `name` is locally infeasible.

The problem is instantiated from CUTEst, preprocessed (fixed variables are
removed), and solved with L2Penalty to obtain a candidate point `x`.

We then check whether `x` sits at a locally infeasible point by converting

    min_x f(x) s.t. c(x) = 0

into

    min_{x,r} 1/2 ||r||² s.t. c(x) = r,

and solving it with IPOPT, starting from `x`, at high precision. If the
residual ||c(x)|| does not vanish at the solution of that problem, `x` is
declared a locally infeasible point and the function returns `true`.
"""
function certify_local_infeasibility(
  name::AbstractString;
  atol = 1e-6,
  rtol = 0.0,
  max_time = 300.0,
  tol = 1e-12,
  max_cpu_time = 300.0,
  feas_tol = 1e-6,
)
  nlp = CUTEstModel(name)

  try
    preprocess_nlp = nlp
    if length(nlp.meta.ifix) > 0
      preprocess_nlp = remove_fixed_variables(nlp)
    end

    stats = L2Penalty(preprocess_nlp, print_level = 0, atol = atol, rtol = rtol, max_time = max_time)
    x = stats.solution[nlp.meta.ifree]

    # Step 1: convert min_x f(x) s.t. c(x) = 0
    # into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r, where r is a slack variable.

    # Step 1.1: convert min_x f(x) s.t. c(x) = 0 into min_{x,r} 1/2 ||c(x)||^2.
    feas_nls = FeasibilityResidual(preprocess_nlp)

    # Step 1.2: convert min_{x,r} 1/2 ||c(x)||^2 into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r.
    feas_nlp = FeasibilityFormNLS(feas_nls)

    # Step 2: check if the problem is locally infeasible at x.
    # For this, solve the problem min_{x,r} 1/2 ||r||^2 s.t. c(x) = r using IPOPT,
    # starting from x and with high precision.
    x0 = vcat(x, cons(preprocess_nlp, x))
    ipopt_stats =
      ipopt(feas_nlp, x0 = x0, tol = tol, max_cpu_time = max_cpu_time, print_level = 0)

    # Step 3: the problem is certified locally infeasible if the residual
    # ||c(x)|| does not vanish at the solution.
    xsol = ipopt_stats.solution[1:preprocess_nlp.meta.nvar]
    primal_feas = norm(cons(preprocess_nlp, xsol), Inf)
    certified = primal_feas > feas_tol

    @info "Local infeasibility check for $(name): ||c(xsol)|| = $primal_feas -> $(certified ? "certified infeasible" : "not certified")"

    return certified
  finally
    finalize(nlp)
  end
end
