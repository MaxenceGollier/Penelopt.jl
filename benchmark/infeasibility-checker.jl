using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra

function check_local_infeasibility(nlp::AbstractNLPModel, x::AbstractVector)

  # Step 1: convert min_x f(x) s.t. c(x) = 0
  # into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r, where r is a slack variable.

  # Step 1.1: convert min_x f(x) s.t. c(x) = 0 into min_{x,r} 1/2 ||c(x)||^2.
  feas_nls = FeasibilityResidual(nlp)

  # Step 1.2: convert min_{x,r} 1/2 ||c(x)||^2 into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r.
  feas_nlp = FeasibilityFormNLS(feas_nls)

  # Step 2: check if the problem is locally infeasible at x.
  # For this, solve the problem min_{x,r} 1/2 ||r||^2 s.t. c(x) = r using IPOPT,
  # Starting from x and with high precision.
  tol = 1e-12

  x0 = vcat(x, cons(nlp, x))
  stats = ipopt(
    feas_nlp,
    x0 = x0,
    tol = tol,
    max_iter = 10
  )

  # Step 3: check if the solution is such that ||c(x)|| neq 0.
  xsol = stats.solution[1:nlp.meta.nvar]
  primal_feas = norm(cons(nlp, xsol), Inf)

  println("----------------------")
  println("PROBLEM : $(nlp.meta.name)")
  println("Infeasibility check: ||c(x)|| = $primal_feas")
  println("Infeasibility check: ||x-xsol||//||xsol|| = $(norm(x - xsol) / norm(xsol))")
  println("----------------------")
end

nlp = CUTEstModel("ARTIF")
preprocess_nlp = nlp

if length(nlp.meta.ifix) > 0
  preprocess_nlp = remove_fixed_variables(nlp)
end

stats = L2Penalty(preprocess_nlp, print_level = 2, atol = 1e-6, rtol = 0.0, max_time = 300.0)
x = stats.solution[nlp.meta.ifree]

check_local_infeasibility(preprocess_nlp, x)

finalize(nlp)