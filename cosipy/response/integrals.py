import numpy as np

from astromodels import (
    Powerlaw,
    Cutoff_powerlaw,
    Band,
    Band_grbm,
    Gaussian,
    DiracDelta,
    StepFunction,
    StepFunctionUpper,
    Constant,
    Line,
    Quadratic,
    Cubic,
    Quartic,
)

from cosipy.threeml import Band_Eflux

def get_integral_values(f, x_in, force_quad = False):
    """
    Compute the integral of a function f between the specified
    endpoints.  If available, use an integral formula specific
    to the function family; otherwise, use adaptive quadrature.

    Inputs
    ------
    f : Astromodels.Function1D
      the function to integrate
    x_in : array-like of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    force_quad: bool, optional
      force integration with adaptive quadrature rather than using
      any available family-specific integrator; default False.
      (This option has no effect if the function family is known to give
      inaccurate results with adaptive quadrature, in particular if
      if has a discontinuity.)

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    x = np.asarray(x_in)

    # Functions with discontinuities either give inaccurate results or
    # fail altogether with adaptive quadrature.
    if force_quad and \
       not isinstance(f, (DiracDelta, StepFunction, StepFunctionUpper)):
        return integral_generic(f, x)

    match f:
        case Powerlaw():
            return integral_powerlaw(x,
                                     f.index.value,
                                     f.piv.value,
                                     f.K.value)

        case Cutoff_powerlaw():
            return integral_co_powerlaw(x,
                                        f.index.value,
                                        f.piv.value,
                                        f.xc.value,
                                        f.K.value)

        case Band():
            return integral_band(x,
                                 f.alpha.value,
                                 f.beta.value,
                                 f.piv.value,
                                 f.xp.value,
                                 f.K.value)

        case Band_grbm():
            # Band_grbm is same as Band, except that everywhere
            # the Band_grbm formula uses xp (the cutoff value), the
            # Band formula uses xc/(2+alpha).  So correct for this
            # before we call the integral.
            return integral_band(x,
                                 f.alpha.value,
                                 f.beta.value,
                                 f.piv.value,
                                 f.xc.value*(2 + f.alpha.value),
                                 f.K.value)

        case Band_Eflux():
            # Band_Eflux is like Band_grbm, except that the
            # returned values are normalized by the integral
            # of the Band function between a and b. There is
            # no pivot value (so treat it as 1.0).

            # normalization constant
            norm = f.get_normalization(f.a.value,
                                       f.b.value,
                                       f.alpha.value,
                                       f.beta.value,
                                       f.E0.value)
            return integral_band(x,
                                 f.alpha.value,
                                 f.beta.value,
                                 1.0,
                                 f.E0.value*(2 + f.alpha.value),
                                 f.K.value / norm)

        case Gaussian():
            return integral_gaussian(x,
                                     f.mu.value,
                                     f.sigma.value,
                                     f.F.value)

        case Constant():
            return integral_polynomial(x,
                                       np.array([f.k.value]))

        case Line():
            return integral_polynomial(x,
                                       np.array([f.b.value,
                                                 f.a.value]))

        case Quadratic():
            return integral_polynomial(x,
                                       np.array([f.c.value,
                                                 f.b.value,
                                                 f.a.value]))

        case Cubic():
            return integral_polynomial(x,
                                       np.array([f.d.value,
                                                 f.c.value,
                                                 f.b.value,
                                                 f.a.value]))

        case Quartic():
            return integral_polynomial(x,
                                       np.array([f.e.value,
                                                 f.d.value,
                                                 f.c.value,
                                                 f.b.value,
                                                 f.a.value]))

        case DiracDelta():
            return integral_diracdelta(x,
                                       f.zero_point.value,
                                       f.value.value)

        case StepFunction() | StepFunctionUpper():
            return integral_stepfunction(x,
                                         f.lower_bound.value,
                                         f.upper_bound.value,
                                         f.value.value)

        case _:
            return integral_generic(f, x)

def integral_generic(f, x):
    """
    Compute the integral of a function f between the specified
    endpoints using adaptive quadrature

    Inputs
    ------
    f : function of type float -> float
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    from scipy import integrate

    return np.array([
        integrate.quad(f, xl, xh)[0] for
        xl, xh in zip(x[:-1], x[1:])
    ])


def integral_powerlaw(x, b, p, c):
    """
    Compute the integral of a powerlaw between the specified
    endpoints.  A Powerlaw has the form

      f(x) = c * (x/p)^b

    where p and c are positive real numbers and b is a non-positive
    real number.  The argument x is a non-negative real number.

    Inputs
    ------
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    a, b, p, c : float
      function parameters

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    if b == -1.:
        # special case: f(x) = cp/x, so
        # integral is cp * ln(x)
        v = np.log(x)
        v *= c*p
    else:
        # integral of powerlaw is
        #   cp/(b+1) (x/p)^(b+1)
        v = np.power(x/p, b + 1.)
        v *= c*p/(b + 1.)

    return np.diff(v)


def integral_co_powerlaw(x, a, p, c, K):
    """
    Compute the integral of a cut-off powerlaw between the specified
    endpoints. A cut-off powerlaw has the form

      f(x) = K (x/p)^a * exp(-x/c)

    where p and c are positive real numbers and a is a positive-negative
    real number.  The argument x is a non-negative real number.

    Inputs
    ------
    x: array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    a, p, b, c, K : float
      parameters of function

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    Note: integral formulas were sourced from Wolfram Alpha.
    """

    from scipy.special import expn, gamma, gammaincc

    z = x/c

    if isinstance(a, (int, np.integer)) or a.is_integer():
        # For integer a, use generalized exponential integral, since
        # the gamma function diverges at integers <= 0.
        v = -np.power(x/p, a) * x
        v *= expn(-a, z) * K
    else:
        # For non-integer a, use upper incomplete gamma function
        # (denoted Gamma(s,z) below).

        # compute Gamma(1 + a, x/c)
        if a > -1.:
            # SciPy gammaincc() is normalized by 1/gamma(a); undo that.
            v = gamma(1. + a) * gammaincc(1. + a, z)
        else:
            # SciPy, unlike some other tools (but similar to the
            # C++ standard library), does not support computing
            # Gamma(s, z) for s <= 0.  So we use this recurrence
            # (see Wikipedia on Gamma(s,z)):
            #
            #   Gamma(s, z) = 1/s [Gamma(s+1, z) - x^s exp(-x)]
            #
            # Apply the recurrence repeatedly until s becomes
            # non-negative, then use the base case above.

            expmz = np.exp(-z)

            v = np.zeros_like(z)
            d = 1.
            s = 1. + a

            while s < 0.:
                d /= s
                v -= np.power(z, s) * expmz * d
                s += 1.

            v += d * gamma(s) * gammaincc(s, z)

        v *= -np.power(c/p, a) * c * K

    return np.diff(v)


def integral_band(x, a, b, p, c, K):
    """
    Compute the integral of a cut-off powerlaw between the specified
    endpoints. A cut-off powerlaw has the form

    A Band spectrum is a combination of a cutoff powerlaw and a
    regular powerlaw.  Specifically, let the critical value
    x_crit = (a - b) c / (a+2). Then

             { (x/p)^a exp(-(a+2)x/c)                 if  x <= x_crit
    f(x) = K {
             { (x/p)^b exp(b-a) [(a-b)c/(a+2)p]^(a-b) if x > x_crit

    where p and c are positive real numbers and a and b are
    non-positive real numbers.  The argument x is a non-negative real
    number.

    Inputs
    ------
    x: array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    a, b, p, c, K : float
      parameters of function

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """
    dc = c/(a + 2.)
    cutoff = dc*(a - b)

    # x[split - 1] <= cutoff < x[split]
    split = np.searchsorted(x, cutoff, side='right')

    def lo_int(x):
        return integral_co_powerlaw(x, a, p, dc, K)

    def hi_int(x):
        return integral_powerlaw(x, b, p,
                                 K * np.exp(b - a) * np.power(cutoff/p, a - b))

    if split == 0:
        # all of x is above cutoff
        v = hi_int(x)
    elif split == len(x):
        # all of x is below cutoff
        v = lo_int(x)
    else:
        if x[split - 1] == cutoff:
            # subarrays of x below and above cutoff share no bins
            x_lo = x[:split]
            x_hi = x[split-1:]
            v = np.concatenate((lo_int(x_lo), hi_int(x_hi)))
        else:
            # split bin containing cutoff and add low and high
            # contributions to that bin
            x_lo = np.concatenate((x[:split], (cutoff,)))
            x_hi = np.concatenate(((cutoff,), x[split:]))

            v = np.zeros(len(x) - 1)
            v[:split]   += lo_int(x_lo)
            v[split-1:] += hi_int(x_hi)

    return v


def integral_gaussian(x, mu, sigma, F):
    """
    Compute the integral of a Gaussian between the specified
    endpoints. A Gaussian has the form

      f(x) = F/[sqrt(2*pi) * sigma] exp(-(x-mu)^2/(2 sigma^2))

    where mu is a real number and sigma a non-negative real number.
    The argument x is a real number.

    Inputs
    ------
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    mu, sigma, F : float
      parameters of function

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x
    """

    from scipy.special import erf

    # Gaussian integral from -inf to x is given by
    #   1/2 [ 1 + erf((x - mu)/(sqrt(2) * sigma)) ]

    z = (x - mu)/sigma

    isqrt2 = 0.7071067811865475  # 1/sqrt(2)
    v = erf(isqrt2 * z)
    v *= 0.5 * F

    # skip this, since we only compute diffs below
    # v += 0.5

    return np.diff(v)

def integral_polynomial(x, coeffs):
    """
    Compute the integral of a polynomial between the specified
    endpoints. The polynomial is given by its coefficients
    in order from highest to lowest power; e.g., [3, 4, 1]
    describes the polynomial 3x^2 + 4x + 1.

    Inputs
    ------
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    coeffs : array-like of float
      coefficients of polynomial

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    # denominators associated with each term of integral
    corrs = np.arange(len(coeffs), 0, -1)

    # evaluate the integral at x
    v = np.polyval(np.append(coeffs/corrs, 0.), x)

    return np.diff(v)

def integral_stepfunction(x, x_lo, x_hi, value):
    """
    Compute the integral of a step function between the specified
    endpoints. A step function has the form

    f(x) = { value     if x_lo <= x <= x_hi
           { 0.        otherwise

    Note that the inequalities can be strict or non-strict; the
    integral is not affected.

    Inputs
    ------
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    x_lo, x_hi, value : float
      parameters of function

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    # contribution of Heaviside(x_lo)
    v_step_up   = np.maximum(0., x[1:] - np.maximum(x[:-1],x_lo))
    # contribution of Heaviside(x_hi)
    v_step_down = np.maximum(0., x[1:] - np.maximum(x[:-1],x_hi))

    v = v_step_up - v_step_down
    v *= value

    return v

def integral_diracdelta(x, x_nonzero, value):
    """
    Compute the integral of a Dirac Delta function between the
    specified endpoints. A Delta function integrates to 'value' if the
    domain contains the point x_nonzero or zero otherwise.

    Inputs
    ------
    x : array of float
      array of monotonically increasing grid points; integration is
      performed between each successive pair of points
    x_nonzero, value : float
      parameters of function

    Returns
    -------
    array of |x|-1 values containing definite integral values
    between each successive pair of points in x

    """

    bins = ((x[:-1] <= x_nonzero) & (x[1:] >= x_nonzero))
    v = np.where(bins, value, 0.)

    return v
