"""calib -- exact rational Gaussian conditioning for calibration review.

A calibration analyst reviews readings that share a reference error:
values that are individually plausible can still be jointly impossible.
This module registers a Gaussian model on an affine constraint set and
evaluates linear Gaussian readings against it with exact Fraction
arithmetic, flagging unrealizable observation tuples as ``inconsistent``.

Model registration, ``compile(names, P, h, A, b)``:

    density proportional to  exp(-x^T P x / 2 + h^T x)
    on the affine set        {x : A x = b}   (at most 8 equations).

``P`` must be symmetric positive semidefinite and positive definite on
ker(A); consistent redundant rows and fully determined points are
accepted.  All numerics are int (not bool), Fraction, or rational
number strings.  The returned model is reusable; each analysis is
independent.

Analysis, ``evaluate(model, H, y, R, L)``:

    readings  y = H x + noise,  noise zero-mean Gaussian with
    covariance R (possibly correlated/singular), independent of the
    model.  ``H`` and ``L`` are ordered dicts mapping non-empty row
    names to {variable: coefficient} dicts (missing entries are zero;
    unknown variables are rejected).  ``H`` has 0-8 rows, ``L`` has
    1-8 rows, ``y`` has the same keys as ``H``, and ``R`` is the
    symmetric PSD covariance in H order (empty H/y/R means no
    observations).

    Returns ``(status, row_names, means, covariance)`` where the last
    three follow L order as tuples of Fractions (covariance is the full
    tuple of row tuples).  If the observations are not realizable the
    status is ``"inconsistent"`` and both numeric items are empty
    tuples; otherwise the status is ``"ok"``.

Only the Python standard library is used.
"""

from fractions import Fraction
from itertools import combinations
from collections.abc import Mapping, Sequence

__all__ = ["compile", "evaluate"]

_MAX_NAMES = 8
_MAX_EQS = 8
_MAX_OBS = 8
_MAX_OUT = 8

_ZERO = Fraction(0)
_ONE = Fraction(1)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def _num(v, what):
    """Parse one declared numeric: int (not bool), Fraction, or rational str."""
    if isinstance(v, bool):
        raise ValueError("%s: booleans are not valid numerics" % what)
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, str):
        try:
            return Fraction(v)
        except (ValueError, ZeroDivisionError):
            raise ValueError("%s: %r is not a rational number string"
                             % (what, v)) from None
    raise ValueError("%s: unsupported numeric type %s"
                     % (what, type(v).__name__))


def _is_seq(v):
    return isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))


def _vec(v, n, what):
    if not _is_seq(v) or len(v) != n:
        raise ValueError("%s must be a sequence of %d numerics" % (what, n))
    return [_num(x, "%s[%d]" % (what, i)) for i, x in enumerate(v)]


def _mat(M, rows, cols, what):
    if not _is_seq(M) or len(M) != rows:
        raise ValueError("%s must be a sequence of %d rows" % (what, rows))
    return [_vec(r, cols, "%s[%d]" % (what, i)) for i, r in enumerate(M)]


def _mat_rows(M, cols, max_rows, what):
    if not _is_seq(M):
        raise ValueError("%s must be a sequence of rows" % what)
    if len(M) > max_rows:
        raise ValueError("%s must have at most %d rows" % (what, max_rows))
    return [_vec(r, cols, "%s[%d]" % (what, i)) for i, r in enumerate(M)]


def _check_symmetric(M, what):
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j] != M[j][i]:
                raise ValueError("%s must be symmetric" % what)


def _rows(obj, lo, hi, index, what):
    """Validate an ordered row-name -> {variable: coefficient} mapping."""
    if not isinstance(obj, Mapping):
        raise ValueError("%s must be an ordered mapping of row name to a "
                         "variable-coefficient mapping" % what)
    keys = list(obj.keys())
    if not lo <= len(keys) <= hi:
        raise ValueError("%s must have between %d and %d rows"
                         % (what, lo, hi))
    mat = []
    for k in keys:
        if not isinstance(k, str) or not k:
            raise ValueError("%s row names must be non-empty strings" % what)
        coefs = obj[k]
        if not isinstance(coefs, Mapping):
            raise ValueError("%s row %r must map to a variable-coefficient "
                             "mapping" % (what, k))
        row = [_ZERO] * len(index)
        for vn, c in coefs.items():
            if not isinstance(vn, str) or vn not in index:
                raise ValueError("%s row %r names unknown variable %r"
                                 % (what, k, vn))
            row[index[vn]] = _num(c, "%s row %r coefficient %r"
                                  % (what, k, vn))
        mat.append(row)
    return mat, keys


# ---------------------------------------------------------------------------
# exact rational linear algebra
# ---------------------------------------------------------------------------

def _dot(a, b):
    s = _ZERO
    for x, y in zip(a, b):
        s += x * y
    return s


def _matvec(M, v):
    return [_dot(row, v) for row in M]


def _transpose(M):
    if not M:
        return []
    return [list(col) for col in zip(*M)]


def _matmul(A, B):
    Bt = _transpose(B)
    return [[_dot(row, col) for col in Bt] for row in A]


def _rref(rows):
    """Reduced row echelon form over the rationals.

    Returns (rref_rows, pivot_column_indices).
    """
    M = [list(r) for r in rows]
    m = len(M)
    ncols = len(M[0]) if m else 0
    pivots = []
    r = 0
    for c in range(ncols):
        if r >= m:
            break
        piv = -1
        for i in range(r, m):
            if M[i][c] != 0:
                piv = i
                break
        if piv < 0:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        pivots.append(c)
        r += 1
    return M, pivots


def _nullspace(A, n):
    """Basis of {v in Q^n : A v = 0} as a list of column tuples."""
    M, pivots = _rref(A)
    pivset = set(pivots)
    basis = []
    for f in range(n):
        if f in pivset:
            continue
        v = [_ZERO] * n
        v[f] = _ONE
        for i, pc in enumerate(pivots):
            v[pc] = -M[i][f]
        basis.append(tuple(v))
    return basis


def _colbasis(A):
    """Pivot columns of A: a basis of its column space."""
    if not A:
        return []
    _, pivots = _rref(A)
    return [tuple(row[c] for row in A) for c in pivots]


def _solve(A, b, n):
    """One solution of A x = b (free variables zero), or None if inconsistent."""
    aug = [list(row) + [b[i]] for i, row in enumerate(A)]
    M, pivots = _rref(aug)
    for row in M:
        if row[n] != 0 and all(row[j] == 0 for j in range(n)):
            return None
    x = [_ZERO] * n
    for i, pc in enumerate(pivots):
        if pc < n:
            x[pc] = M[i][n]
    return x


def _det(M):
    n = len(M)
    A = [list(r) for r in M]
    det = _ONE
    for c in range(n):
        piv = -1
        for i in range(c, n):
            if A[i][c] != 0:
                piv = i
                break
        if piv < 0:
            return _ZERO
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        pv = A[c][c]
        det *= pv
        for i in range(c + 1, n):
            if A[i][c] != 0:
                f = A[i][c] / pv
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return det


def _is_psd(M):
    """Symmetric M is PSD iff every principal minor is nonnegative."""
    n = len(M)
    idx = range(n)
    for k in range(1, n + 1):
        for S in combinations(idx, k):
            sub = [[M[i][j] for j in S] for i in S]
            if _det(sub) < 0:
                return False
    return True


def _is_pd(M):
    """Sylvester's criterion: all leading principal minors positive."""
    for k in range(1, len(M) + 1):
        if _det([row[:k] for row in M[:k]]) <= 0:
            return False
    return True


def _inv(M):
    n = len(M)
    A = [list(M[i]) + [_ONE if i == j else _ZERO for j in range(n)]
         for i in range(n)]
    for c in range(n):
        piv = -1
        for i in range(c, n):
            if A[i][c] != 0:
                piv = i
                break
        if piv < 0:
            raise ValueError("matrix is singular")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return [row[n:] for row in A]


# ---------------------------------------------------------------------------
# model registration
# ---------------------------------------------------------------------------

class _Model:
    """Reusable registered model; immutable, analyses stay independent."""

    __slots__ = ("names", "x0", "Z", "mu_z", "Sigma_z")

    def __init__(self, names, x0, Z, mu_z, Sigma_z):
        self.names = names        # tuple of variable names, length n
        self.x0 = x0              # particular solution of A x = b, length n
        self.Z = Z                # tuple of d kernel basis columns (length n)
        self.mu_z = mu_z          # prior mean in z coordinates, length d
        self.Sigma_z = Sigma_z    # prior covariance in z coordinates, d x d


def compile(names, P, h, A, b):
    """Register a reusable Gaussian model on an affine constraint set."""
    # names: 1-8 distinct non-empty strings
    if not _is_seq(names):
        raise ValueError("names must be a sequence of 1-8 distinct "
                         "non-empty strings")
    names = tuple(names)
    if not 1 <= len(names) <= _MAX_NAMES:
        raise ValueError("names must contain between 1 and %d entries"
                         % _MAX_NAMES)
    for nm in names:
        if not isinstance(nm, str) or not nm:
            raise ValueError("variable names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("variable names must be distinct")
    n = len(names)

    Pm = _mat(P, n, n, "P")
    hv = _vec(h, n, "h")
    Am = _mat_rows(A, n, _MAX_EQS, "A")
    bv = _vec(b, len(Am), "b")

    _check_symmetric(Pm, "P")
    if not _is_psd(Pm):
        raise ValueError("P must be positive semidefinite")

    x0 = _solve(Am, bv, n)
    if x0 is None:
        raise ValueError("A x = b is inconsistent")

    Z = _nullspace(Am, n)
    d = len(Z)
    if d:
        # P must be positive definite on ker(A)
        Lam = [[_dot(zi, _matvec(Pm, zj)) for zj in Z] for zi in Z]
        if not _is_pd(Lam):
            raise ValueError("P must be positive definite on the kernel of A")
        Px0 = _matvec(Pm, x0)
        eta = [_dot(zi, [hv[k] - Px0[k] for k in range(n)]) for zi in Z]
        Sig = _inv(Lam)
        mu = _matvec(Sig, eta)
    else:
        # A determines x uniquely: a point mass at x0.
        Sig = []
        mu = []

    return _Model(names, tuple(x0), tuple(Z), tuple(mu),
                  tuple(tuple(row) for row in Sig))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _split_constraints(G, c, d):
    """Reduce G z = c to an independent row set.

    Returns (G1, c1) with G1 of full row rank, or None if inconsistent.
    """
    if not G:
        return [], []
    aug = [list(row) + [c[i]] for i, row in enumerate(G)]
    M, _ = _rref(aug)
    G1, c1 = [], []
    for row in M:
        lead = None
        for j in range(d + 1):
            if row[j] != 0:
                lead = j
                break
        if lead is None:
            continue
        if lead == d:
            return None
        G1.append(row[:d])
        c1.append(row[d])
    return G1, c1


def evaluate(model, H, y, R, L):
    """Condition the model on readings and report posterior L-functionals."""
    if not isinstance(model, _Model):
        raise ValueError("model must be produced by calib.compile")
    names = model.names
    n = len(names)
    index = {nm: i for i, nm in enumerate(names)}

    Hm, hrows = _rows(H, 0, _MAX_OBS, index, "H")
    m = len(hrows)

    if not isinstance(y, Mapping):
        raise ValueError("y must be a mapping with the same keys as H")
    if set(y.keys()) != set(hrows):
        raise ValueError("y must have the same keys as H")
    yv = [_num(y[k], "y[%r]" % k) for k in hrows]

    Rm = _mat(R, m, m, "R")
    _check_symmetric(Rm, "R")
    if not _is_psd(Rm):
        raise ValueError("R must be positive semidefinite")

    Lm, lrows = _rows(L, 1, _MAX_OUT, index, "L")
    lnames = tuple(lrows)

    # Observation equation in kernel coordinates:  y' = F z + noise
    x0 = model.x0
    Z = model.Z
    d = len(Z)
    Fcols = [[_dot(Hm[i], Z[j]) for i in range(m)] for j in range(d)]
    yp = [yv[i] - _dot(Hm[i], x0) for i in range(m)]

    # Split the noise: U spans range(R) (stochastic), W spans ker(R)
    # (deterministic linear constraints on z).
    U = _colbasis(Rm)
    W = _nullspace(Rm, m)
    G = [[_dot(w, Fcols[j]) for j in range(d)] for w in W]
    c = [_dot(w, yp) for w in W]
    split = _split_constraints(G, c, d)
    if split is None:
        return ("inconsistent", lnames, (), ())
    G1, c1 = split

    mu = list(model.mu_z)
    Sig = [list(row) for row in model.Sigma_z]

    if d and G1:
        # Condition z ~ N(mu, Sig) on the exact constraints G1 z = c1.
        SGt = _matmul(Sig, _transpose(G1))          # d x q
        M = _inv(_matmul(G1, SGt))                  # q x q, positive definite
        diff = [c1[i] - _dot(G1[i], mu) for i in range(len(G1))]
        adj = _matvec(SGt, _matvec(M, diff))
        mu = [mu[i] + adj[i] for i in range(d)]
        T = _matmul(SGt, M)                         # d x q
        q = len(G1)
        Sig = [[Sig[i][j] - _dot(T[i], [SGt[j][k] for k in range(q)])
                for j in range(d)] for i in range(d)]

    if d and U:
        # Bayesian update with the stochastic readings (positive definite D).
        E = [[_dot(u, Fcols[j]) for j in range(d)] for u in U]   # r x d
        wobs = [_dot(u, yp) for u in U]
        D = [[_dot(u, _matvec(Rm, v)) for v in U] for u in U]    # r x r, PD
        SEt = _matmul(Sig, _transpose(E))           # d x r
        S = _matmul(E, SEt)
        for i in range(len(U)):
            for j in range(len(U)):
                S[i][j] += D[i][j]
        Sinv = _inv(S)
        gain = _matvec(Sinv, [wobs[i] - _dot(E[i], mu)
                              for i in range(len(U))])
        mu = [mu[i] + _dot(SEt[i], gain) for i in range(d)]
        K = _matmul(SEt, Sinv)                      # d x r
        Sig = [[Sig[i][j] - _dot(K[i], [SEt[j][k] for k in range(len(U))])
                for j in range(d)] for i in range(d)]

    # Back to x coordinates: x = x0 + Z z.
    mux = [x0[i] + sum(Z[j][i] * mu[j] for j in range(d)) for i in range(n)]
    if d:
        Zmat = [[Z[j][i] for j in range(d)] for i in range(n)]
        Sigx = _matmul(_matmul(Zmat, Sig), _transpose(Zmat))
    else:
        Sigx = [[_ZERO] * n for _ in range(n)]

    means = tuple(_matvec(Lm, mux))
    cov = _matmul(_matmul(Lm, Sigx), _transpose(Lm))
    return ("ok", lnames, means, tuple(tuple(row) for row in cov))
