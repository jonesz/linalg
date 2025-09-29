-- | Generic solver for fix-point (Bellman) equations. This module implements a
-- generic solver for finding solutions to fix-point (Bellman) equations. The
-- module may be used for finding solutions to dynamic programming problems. The
-- implementation is based on John Rust's poly-algorithm, which combines the
-- strategies of successive applications and the Newton-Kantorovich method.
--
-- The module is enriched with a version of the solver that uses Futhark's
-- support for automatic differentiation (forward mode AD) to compute the
-- Jacobian for the passed fix-point function.
--
-- For more details, see Rust's description in Section 4 of
-- https://editorialexpress.com/jrust/sdp/ndp.pdf and a demonstration of its use
-- in https://elsman.com/pdf/dpsolve-2025-09-27.pdf.

import "linalg"
import "lu"

local
-- | Module type specifying generic solvers for Bellman equations. Notice that
-- this module type is declared `local`, which means that it cannot directly be
-- referenced by name from the outside.  By not exposing the module type, the
-- module may be extended with new members in minor versions.
module type dpsolve = {
  -- | The scalar type.
  type t

  -- | Type of square matrices of size n (e.g., square or dense), with t as
  -- scalar type.
  type mat [n]

  -- | The type of solver parameters.
  type param = {
    sa_max          : i64,  -- Maximum and minimum numbers of
    sa_min          : i64,  --   successive approximation steps.
    sa_tol          : t,    -- Stopping tolerance for successive
                            --   approximation.
    max_fxpiter     : i64,  -- Maximum number of times to switch
                            --   between Newton-Kantorovich and
                            --   successive approximation iterations.
    pi_max          : i64,  -- Maximum number of Newton-Kantorovich steps.
    pi_tol          : t,    -- Final exit tolerance in fixed point
                            --   algorithm, measured in units of
                            --   numerical precision
    tol_ratio       : t     -- Relative tolerance before switching
                            --   to N-K algorithm when discount factor is
                            --   supplied as input in `poly`.
    }

  -- | The default parameter value
  val default : param

  -- | Find a fix-point for the function `f` using successive approximation with
  -- the initial guess `v`, parameter `p`, and beta-value `b`. The tolerance,
  -- the minimal, and the maximal number of iterations can be adjusted by
  -- altering the parameter `p`. The argument `b` is the beta discount factor,
  -- which allows for stopping early (when the relative tolerance is close to
  -- `b`). The function returns a quintuple containing an approximate fix-point,
  -- a boolean specifying whether the algorithm converged (according to the
  -- values in `p`), the number of iterations used, and finally, the tolerance
  -- and the relative tolerance of the last two fix-point approximations and the
  -- last two tolerances, respectively (maximum of each dimension).
  val sa [m] : (f:[m]t->[m]t) -> (v:[m]t) -> (p:param) -> (b:t)
               -> ([m]t, bool, i64, t, t)

  -- | Find a fix-point for the function `f` using Newton-Kantorovich iterations
  -- with the initial guess `v`, and parameter `p`. The tolerance, the minimal,
  -- and the maximal number of iterations can be adjusted by altering the
  -- parameter `p`. The function `f` should return a pair of a new next
  -- approximation and the Jacobian matrix for the function `f` relative to the
  -- argument given. The function returns a quintuple containing an approximate
  -- fix-point, a Jacobian matrix for the fix-point, a boolean specifying whether
  -- the algorithm converged (according to the values in `p`), the number of
  -- iterations used, and finally, the tolerance of the last two fix-point
  -- approximations (maximum of each dimension).
  val nk [m] : (f: [m]t -> ([m]t,mat[m])) -> (v:[m]t) -> (p:param)
               ->  ([m]t, mat[m], bool, i64, t)

  -- | Find a fix-point for the function `f` using a combination of successive
  -- approximation iterations and Newton-Kantorovich iterations. The initial
  -- guess is `v` and the parameter `p` is passed to the calls to `sa` and
  -- `nk`. The function `f` should return a pair of a new next approximation and
  -- the Jacobian matrix for the function `f` relative to the argument
  -- given. The function returns a 7-tuple containing an approximate fix-point, a
  -- Jacobian matrix for the fix-point, a boolean specifying whether the
  -- algorithm converged (according to the values in `p`), the number of
  -- iterations used for the total sa iterations, the total nk iterations, and
  -- the number of round-trips. The 7'th element of the result tuple is the
  -- tolerance of the last two fix-point approximations (maximum of each
  -- dimension).
  val poly [m] : (f: [m]t -> ([m]t,mat[m])) -> (v:[m]t) -> (p:param)
                 -> (b:t) -> ([m]t,mat[m], bool, i64, i64, i64, t)

  -- | Find a fix-point for the function `f` using a combination of successive
  -- approximation iterations and Newton-Kantorovich iterations. The initial
  -- guess is `v` and the parameter `p` is passed to the calls to `sa` and
  -- `nk`. The function uses forward-mode automatic differentiation to compute
  -- the Jacobian matrix relative to the argument given to `f`. The function
  -- returns a 6-tuple containing an approximate fix-point, a boolean specifying
  -- whether the algorithm converged (according to the values in `p`), the
  -- number of iterations used for the total sa iterations, the total nk
  -- iterations, and the number of round-trips. The 6'th element of the result
  -- tuple is the tolerance of the last two fix-point approximations (maximum of
  -- each dimension).
  val polyad [m] : (f:[m]t->[m]t) -> (v:[m]t) -> (p:param) -> (b:t)
                   -> ([m]t, bool, i64, i64, i64, t)
}

-- | Module type specifying a linear equations solver and functionality for
-- lifting a function to return also the partial derivative (i.e., the Jacobian
-- matrix) relative to the argument.

module type ols_jac = {
  -- | The scalar type.
  type t

  -- | Type of square matrices of size `n` (e.g., square or dense), with `t` as
  -- scalar type.
  type mat [n]

  -- | The identity matrix of size `n` x `n`.
  val eye : (n:i64) -> mat [n]

  -- | The zero matrix of size `n` x `n`.
  val zero : (n:i64) -> mat [n]

  -- | Element-wise subtraction.
  val sub [n] : mat [n] -> mat [n] -> mat [n]

  -- | Solve systems of linear equations `Ax = b` with respect to `x`.
  val ols [n] : mat [n] -> [n]t -> [n]t

  -- | Augment function result with its partial derivative (Jacobinan) relative
  -- to the function argument. We have `wrap f x = (f x, Df x)`, where `Df x` is
  -- the Jacobian matrix for `f` at `x`.
  val wrapj [n] : ([n]t->[n]t) -> [n]t -> ([n]t,mat [n])
}

-- | Parameterised module for creating generic (e.g., non-linear) solvers. The
-- module is parameterised over an instance of `real`, such as the module `f64`
-- and a compatible module for solving linear equations and computing partial
-- derivatives for a function. One possible instance uses a dense linear
-- equations solver on `f64` (e.g., `lu f64`) and Futhark's AD support.

module mk_dpsolve (T:real)
                  (ols_jac: ols_jac with t = T.t) : dpsolve with t = T.t
                                                            with mat [n] = ols_jac.mat [n] =
{
  type t = T.t
  type mat [n] = ols_jac.mat [n]
  local module lu = mk_lu T

  def maxT (a:t) (b:t) : t =
    if T.(a >= b) then a else b

  def zeroT : t = T.i64 0
  def maximumT [n] (a:[n]t) : t = reduce maxT zeroT a

  type param = {
    sa_max          : i64,  -- Maximum number of contraction steps
    sa_min          : i64,  -- Minimum number of contraction steps
    sa_tol          : t,    -- Absolute tolerance before (in dpsolve.poly: tolerance before switching
                            --   to N-K algorithm)
    max_fxpiter     : i64,  -- Maximum number of times to switch between Newton-Kantorovich iterations
                            --   and contraction iterations.
    pi_max          : i64,  -- Maximum number of Newton-Kantorovich steps
    pi_tol          : t,    -- Final exit tolerance in fixed point algorithm, measured in units of
                            --   numerical precision
    tol_ratio       : t     -- Relative tolerance before switching to N-K algorithm
                            --   when discount factor is supplied as input in dpsolve.poly
  }

  def default : param =
    {sa_max      = 20,
     sa_min      = 2,
     sa_tol      = T.f64 1.0e-3,
     max_fxpiter = 35,
     pi_max      = 40,
     pi_tol      = T.f64 1.0e-13,
     tol_ratio   = T.f64 1.0e-03
    }

  def sa [m] (bellman : [m]t -> [m]t)
             (V0:[m]t)
             (ap:param)
             (bet:t) : ([m]t, bool, i64, t, t) =
      loop (V0,converged,i,tol,_rtol) = (V0, false, 0, T.i64 0, T.i64 0)
      while !converged && i < ap.sa_max do
        let V = bellman V0
        let tol' = maximumT (map2 (\a b -> T.(abs(a-b))) V V0)
        let rtol' = if i == 1 then T.i64 1
                    else T.(tol' / tol)
        let converged =
             -- Rule 1
             (i > ap.sa_min && (T.(abs(bet-rtol') < ap.tol_ratio)))
          || -- Rule 2
             --let adj = f64.(maximum V0 |> abs |> log10 |> ceil)
             --let ltol = ap.sa_tol * f64.(10 ** adj)
             let ltol = ap.sa_tol
             in (i > ap.sa_min && T.(tol' < ltol))
        in (V, converged, i+1, tol',rtol')

  def nk [m] (bellman : [m]t -> ([m]t,mat [m]))
             (V0:[m]t)
             (ap:param) : ([m]t, mat [m], bool, i64, t) =
    loop (V0,_dV0,converged,i,_tol) = (V0, ols_jac.zero m, false, 0, T.i64 1)
      while !converged && i < ap.pi_max do
        let (V1, dV) = bellman V0
        --let V = map2 (-) V0 (la.matvecmul_row (la.inv dV) V1)
        let F = ols_jac.sub (ols_jac.eye m) dV
        --let _hermitian =
        --  map2 (map2 (T.==)) F (transpose F) |> map (reduce (&&) true) |> reduce (&&) true
        let V = map2 (T.-) V0 (ols_jac.ols F (map2 (T.-) V0 V1))  -- NK-iteration
        -- do additional SA iteration for stability and accurate measure of error bound
        let (V0, _) = bellman V
        let tol' = maximumT (map2 (\a b -> T.(abs(a-b))) V V0) -- tolerance

        -- adjusting the N-K tolerance to the magnitude of ev
        let adj = T.(maximumT V0 |> abs |> log10 |> ceil)
        let ltol = T.(ap.pi_tol * (i64 10 ** adj)) -- Adjust final tolerance
        -- ltol=ap.pi_tol  -- tolerance

        let converged = T.(tol' < ltol) -- Convergence achieved
        in (V0, dV, converged, i+1, tol')

  -- dpsolve.poly(f,v0,ap,bet): Solve for fixed point using a combination of
  -- Successive Approximations (SA) and Newton-Kantorovich (NK) iterations.  The
  -- argument `ap` holds algorithm parameters. The argument `bet` is a discount
  -- factor. Enters rule for stopping SA and switching to NK iterations.  SA
  -- should stop prematurely when relative tolerance is close to bet.  The
  -- argument `v0` is the initial value guess. The function returns a fix-point
  -- `V`, the Frechet derivative of the Bellman operator, and various count
  -- information including the obtained tolerance.

  def poly [m] (bellman : [m]t -> ([m]t,mat[m]))
               (V0:[m]t)
               (ap:param)
               (bet:t) : ([m]t, mat [m], bool, i64, i64, i64, t) =
    loop (V0,_dV,converged,i,j,k,_tol) = (V0, ols_jac.zero m, false, 0, 0, 0, T.i64 1)
      while !converged && k < ap.max_fxpiter do
        -- poly-algorithm loop (switching between sa and nk)
        let (V1,_,i',_,_) = sa ((.0) <-< bellman) V0 ap bet
        let (V2, dV, c2, j', tol) = nk bellman V1 ap
        in (V2,dV,c2,i+i',j+j',k+1,tol)

  def polyad f x ap bet =
    let (y,_,b,i,j,k,tol) = poly (ols_jac.wrapj f) x ap bet
    in (y,b,i,j,k,tol)
}

-- | Parameterised module for creating generic (e.g., non-linear) solvers using
-- dense Jacobians. The module is parameterised over an instance of `real`, such
-- as the module `f64`.
module mk_dpsolve_dense (T:real) : dpsolve with t = T.t with mat[n] = [n][n]T.t = {
    local module lu = mk_lu T
    local module ols_jac = {
      type t = T.t
      type mat [n] = [n][n]t
      def blksz : i64 = 16 -- 16 or 1
      def ols [n] (A:mat[n]) (b:[n]t) : [n]t = lu.ols blksz A b
      def idd n i = tabulate n (\j -> T.bool(i==j))
      def eye n = tabulate_2d n n (\i j -> T.bool (i==j))
      def zero n = replicate n (replicate n (T.i64 0))
      def sub a b = map2 (map2 (T.-)) a b
      def wrapj [m] (f: [m]t->[m]t) (x:[m]t) : ([m]t,[m][m]t) =
        (f x, #[sequential_outer] tabulate m (jvp f x <-< idd m) |> transpose)
    }
    open mk_dpsolve T ols_jac
}
