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

  x0 = vcat(x, zeros(nlp.meta.ncon))
  println(feas_nlp.meta.nvar)
  println(length(x))
  println(nlp.meta.ncon)
  println(length(x0))
  stats = ipopt(
    feas_nlp,
    x0 = x0,
    tol = tol,
  )

  # Step 3: check if the solution is such that ||c(x)|| neq 0.
  xsol = stats.solution[1:nlp.meta.nvar]
  primal_feas = norm(cons(nlp, xsol), Inf)

  println(primal_feas)
end

nlp = CUTEstModel("VANDANIUMS")

stats = L2Penalty(nlp, print_level = 1, atol = 1e-6, rtol = 0.0)
x = stats.solution

check_local_infeasibility(nlp, x)

finalize(nlp)