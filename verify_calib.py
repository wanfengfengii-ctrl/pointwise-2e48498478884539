#!/usr/bin/env python3
"""Verification battery for calib.py.

Run: python3 verify_calib.py
Exits 0 when every check passes, 1 otherwise.
"""

import sys
from fractions import Fraction as F

import calib

_failures = []
_count = 0


def check(label, fn):
    global _count
    _count += 1
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report and continue
        _failures.append("%s: %s" % (label, exc))
        print("FAIL %s: %s" % (label, exc))
    else:
        print("PASS %s" % label)


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# headline scenario: shared reference error, P=diag(1,0), h=(1,0), Y=1
# ---------------------------------------------------------------------------

def make_headline_model():
    return calib.compile(["X", "Y"],
                         [[1, 0], [0, 0]],
                         [1, 0],
                         [[0, 1]],
                         [1])


HEAD_H = {"ox": {"X": 1}, "oy1": {"Y": 1}, "oy2": {"Y": 1}}
HEAD_R = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
HEAD_L = {"X": {"X": 1}, "Y": {"Y": 1}}


def t_headline_ok():
    m = make_headline_model()
    st, rows, mean, cov = calib.evaluate(
        m, HEAD_H, {"ox": 3, "oy1": 2, "oy2": 2}, HEAD_R, HEAD_L)
    assert st == "ok", st
    assert rows == ("X", "Y"), rows
    assert mean == (F(2), F(1)), mean
    assert cov == ((F(0), F(0)), (F(0), F(0))), cov
    assert all(isinstance(v, F) for v in mean)
    assert all(isinstance(v, F) for row in cov for v in row)


def t_headline_inconsistent():
    m = make_headline_model()
    st, rows, mean, cov = calib.evaluate(
        m, HEAD_H, {"ox": 3, "oy1": 2, "oy2": 3}, HEAD_R, HEAD_L)
    assert st == "inconsistent", st
    assert rows == ("X", "Y"), rows
    assert mean == () and cov == (), (mean, cov)


def t_headline_no_observation():
    m = make_headline_model()
    st, rows, mean, cov = calib.evaluate(m, {}, {}, [], HEAD_L)
    assert st == "ok", st
    assert mean == (F(1), F(1)), mean          # X ~ N(1, 1), Y == 1
    assert cov == ((F(1), F(0)), (F(0), F(0))), cov


def t_headline_string_numerics():
    m = calib.compile(["X", "Y"],
                      [["1", "0"], ["0", "0"]],
                      ["1", "0"],
                      [["0", "1"]],
                      ["1"])
    H = {"ox": {"X": "1"}, "oy1": {"Y": "1"}, "oy2": {"Y": "1"}}
    y = {"ox": "3", "oy1": "2", "oy2": "2"}
    R = [["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]]
    st, _, mean, cov = calib.evaluate(m, H, y, R, HEAD_L)
    assert st == "ok" and mean == (F(2), F(1)), (st, mean)
    assert cov == ((F(0), F(0)), (F(0), F(0))), cov


def t_headline_fraction_numerics():
    m = calib.compile(["X", "Y"],
                      [[F(1), F(0)], [F(0), F(0)]],
                      [F(1), F(0)],
                      [[F(0), F(1)]],
                      [F(1)])
    st, _, mean, _ = calib.evaluate(
        m, HEAD_H, {"ox": F(3), "oy1": F(2), "oy2": F(2)}, HEAD_R, HEAD_L)
    assert st == "ok" and mean == (F(2), F(1)), (st, mean)


def t_model_reuse_independent():
    m = make_headline_model()
    r1 = calib.evaluate(m, HEAD_H, {"ox": 3, "oy1": 2, "oy2": 2}, HEAD_R, HEAD_L)
    r2 = calib.evaluate(m, HEAD_H, {"ox": 3, "oy1": 2, "oy2": 3}, HEAD_R, HEAD_L)
    r3 = calib.evaluate(m, HEAD_H, {"ox": 3, "oy1": 2, "oy2": 2}, HEAD_R, HEAD_L)
    assert r1[0] == "ok" and r2[0] == "inconsistent", (r1[0], r2[0])
    assert r1 == r3, "reused model must give identical independent results"


# ---------------------------------------------------------------------------
# scalar models
# ---------------------------------------------------------------------------

def t_scalar_common_noise():
    # X ~ N(0,1); two readings share one noise (R all ones): one effective
    # observation of value 2 with variance 1.
    m = calib.compile(["X"], [[1]], [0], [], [])
    H = {"a": {"X": 1}, "b": {"X": 1}}
    st, _, mean, cov = calib.evaluate(
        m, H, {"a": 2, "b": 2}, [[1, 1], [1, 1]], {"X": {"X": 1}})
    assert st == "ok", st
    assert mean == (F(1),), mean
    assert cov == ((F(1, 2),),), cov


def t_scalar_common_noise_contradiction():
    m = calib.compile(["X"], [[1]], [0], [], [])
    H = {"a": {"X": 1}, "b": {"X": 1}}
    st, _, mean, cov = calib.evaluate(
        m, H, {"a": 2, "b": 3}, [[1, 1], [1, 1]], {"X": {"X": 1}})
    assert st == "inconsistent" and mean == () and cov == (), st


def t_scalar_noiseless():
    m = calib.compile(["X"], [[1]], [0], [], [])
    st, _, mean, cov = calib.evaluate(
        m, {"a": {"X": 1}}, {"a": 5}, [[0]], {"X": {"X": 1}})
    assert st == "ok" and mean == (F(5),) and cov == ((F(0),),), (st, mean, cov)


def t_scalar_mixed_noise():
    # Block noise: readings a,b share one error, c independent.
    m = calib.compile(["X"], [[1]], [0], [], [])
    H = {"a": {"X": 1}, "b": {"X": 1}, "c": {"X": 1}}
    R = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    st, _, mean, cov = calib.evaluate(
        m, H, {"a": 2, "b": 2, "c": 3}, R, {"X": {"X": 1}})
    assert st == "ok", st
    assert mean == (F(5, 3),), mean
    assert cov == ((F(1, 3),),), cov
    st2, _, mean2, cov2 = calib.evaluate(
        m, H, {"a": 2, "b": 3, "c": 3}, R, {"X": {"X": 1}})
    assert st2 == "inconsistent" and mean2 == () and cov2 == (), st2


# ---------------------------------------------------------------------------
# multivariate models
# ---------------------------------------------------------------------------

def t_simple_bayes():
    # x ~ N(0, I2); observe X with unit noise, reading 2.
    m = calib.compile(["X", "Y"], [[1, 0], [0, 1]], [0, 0], [], [])
    st, rows, mean, cov = calib.evaluate(
        m, {"o": {"X": 1}}, {"o": 2}, [[1]],
        {"X": {"X": 1}, "Y": {"Y": 1}})
    assert st == "ok" and rows == ("X", "Y"), (st, rows)
    assert mean == (F(1), F(0)), mean
    assert cov == ((F(1, 2), F(0)), (F(0), F(1))), cov


def t_correlated_full_rank_R():
    # x ~ N(0, I2); observe both coordinates, R = [[2,1],[1,2]], y = (1,1).
    m = calib.compile(["X", "Y"], [[1, 0], [0, 1]], [0, 0], [], [])
    H = {"a": {"X": 1}, "b": {"Y": 1}}
    st, _, mean, cov = calib.evaluate(
        m, H, {"a": 1, "b": 1}, [[2, 1], [1, 2]],
        {"X": {"X": 1}, "Y": {"Y": 1}})
    assert st == "ok", st
    assert mean == (F(1, 4), F(1, 4)), mean
    assert cov == ((F(5, 8), F(1, 8)), (F(1, 8), F(5, 8))), cov


def t_determined_point():
    # A = I fixes x = (1, 2); P = 0 is PSD and vacuously PD on ker A = {0}.
    m = calib.compile(["X", "Y"], [[0, 0], [0, 0]], [0, 0],
                      [[1, 0], [0, 1]], [1, 2])
    st, _, mean, cov = calib.evaluate(
        m, {}, {}, [], {"X": {"X": 1}, "Y": {"Y": 1}})
    assert st == "ok" and mean == (F(1), F(2)), (st, mean)
    assert cov == ((F(0), F(0)), (F(0), F(0))), cov
    # Noiseless reading must match the determined point exactly.
    st2 = calib.evaluate(m, {"o": {"X": 1}}, {"o": 1}, [[0]],
                         {"X": {"X": 1}})[0]
    assert st2 == "ok", st2
    st3, _, mean3, cov3 = calib.evaluate(m, {"o": {"X": 1}}, {"o": 9}, [[0]],
                                         {"X": {"X": 1}})
    assert st3 == "inconsistent" and mean3 == () and cov3 == (), st3


def t_redundant_consistent_A():
    # Rows [1,1]x=3 and [2,2]x=6 are redundant but consistent.
    m = calib.compile(["X", "Y"], [[1, 0], [0, 1]], [0, 0],
                      [[1, 1], [2, 2]], [3, 6])
    st, _, mean, cov = calib.evaluate(
        m, {}, {}, [], {"X": {"X": 1}, "Y": {"Y": 1}})
    assert st == "ok", st
    assert mean == (F(3, 2), F(3, 2)), mean
    assert cov == ((F(1, 2), F(-1, 2)), (F(-1, 2), F(1, 2))), cov


def t_mixed_deterministic_and_stochastic():
    # x ~ N(0, I2); readings s = X+Y+n, d = X-Y+n with common noise n.
    m = calib.compile(["X", "Y"], [[1, 0], [0, 1]], [0, 0], [], [])
    H = {"s": {"X": 1, "Y": 1}, "d": {"X": 1, "Y": -1}}
    st, _, mean, cov = calib.evaluate(
        m, H, {"s": 4, "d": 2}, [[1, 1], [1, 1]],
        {"X": {"X": 1}, "Y": {"Y": 1}})
    assert st == "ok", st
    assert mean == (F(3, 2), F(1)), mean      # Y fixed exactly, X | data
    assert cov == ((F(1, 2), F(0)), (F(0), F(0))), cov


def t_linear_combination_rows():
    m = make_headline_model()
    L = {"sum": {"X": 1, "Y": 1}, "diff": {"X": 1, "Y": -1},
         "lone": {"Y": 1}}
    st, rows, mean, cov = calib.evaluate(m, {}, {}, [], L)
    assert st == "ok" and rows == ("sum", "diff", "lone"), (st, rows)
    assert mean == (F(2), F(0), F(1)), mean
    assert cov == ((F(1), F(1), F(0)),
                   (F(1), F(1), F(0)),
                   (F(0), F(0), F(0))), cov


def t_pure_noise_rows():
    # Zero-coefficient rows observe pure noise; noiseless zero rows must be 0.
    m = calib.compile(["X"], [[1]], [0], [], [])
    H = {"n1": {}, "n2": {}}
    st, _, mean, cov = calib.evaluate(
        m, H, {"n1": 0, "n2": 0}, [[0, 0], [0, 0]], {"X": {"X": 1}})
    assert st == "ok" and mean == (F(0),) and cov == ((F(1),),), (st, mean, cov)
    st2 = calib.evaluate(m, H, {"n1": 1, "n2": 0}, [[0, 0], [0, 0]],
                         {"X": {"X": 1}})[0]
    assert st2 == "inconsistent", st2


def t_maximum_sizes():
    # 8 variables, 8 equations, 8 observations, 8 output rows.
    names = ["v%d" % i for i in range(8)]
    P = [[F(1 if i == j else 0) for j in range(8)] for i in range(8)]
    A = [[F(1 if j == i else 0) for j in range(8)] for i in range(8)]
    b = [F(i) for i in range(8)]
    m = calib.compile(names, P, [F(0)] * 8, A, b)
    H = {"r%d" % i: {"v%d" % i: 1} for i in range(8)}
    y = {"r%d" % i: i for i in range(8)}
    R = [[F(1 if i == j else 0) for j in range(8)] for i in range(8)]
    L = {"o%d" % i: {"v%d" % i: 1} for i in range(8)}
    st, rows, mean, cov = calib.evaluate(m, H, y, R, L)
    assert st == "ok" and rows == tuple("o%d" % i for i in range(8)), st
    assert mean == tuple(F(i) for i in range(8)), mean
    assert cov == tuple(tuple(F(0) for _ in range(8)) for _ in range(8)), cov


# ---------------------------------------------------------------------------
# invalid declarations -> ValueError
# ---------------------------------------------------------------------------

def t_invalid_compile():
    raises_value_error(lambda: calib.compile([], [[1]], [0], [], []))
    raises_value_error(lambda: calib.compile(["a"] * 2, [[1, 0], [0, 1]],
                                             [0, 0], [], []))
    raises_value_error(lambda: calib.compile([""], [[1]], [0], [], []))
    raises_value_error(lambda: calib.compile([1], [[1]], [0], [], []))
    raises_value_error(lambda: calib.compile(["v%d" % i for i in range(9)],
                                             [[1 if i == j else 0
                                               for j in range(9)]
                                              for i in range(9)],
                                             [0] * 9, [], []))
    # P malformed / asymmetric / not PSD / wrong types
    raises_value_error(lambda: calib.compile(["X"], [[1, 0]], [0], [], []))
    raises_value_error(lambda: calib.compile(["X", "Y"], [[1, 2], [0, 1]],
                                             [0, 0], [], []))
    raises_value_error(lambda: calib.compile(["X", "Y"], [[1, 2], [2, 1]],
                                             [0, 0], [], []))
    raises_value_error(lambda: calib.compile(["X"], [[True]], [0], [], []))
    raises_value_error(lambda: calib.compile(["X"], [[1.5]], [0], [], []))
    raises_value_error(lambda: calib.compile(["X"], [["abc"]], [0], [], []))
    raises_value_error(lambda: calib.compile(["X"], [["1/0"]], [0], [], []))
    # P not positive definite on ker(A)
    raises_value_error(lambda: calib.compile(["X"], [[0]], [0], [], []))
    raises_value_error(lambda: calib.compile(["X", "Y"],
                                             [[1, 0], [0, 0]], [0, 0], [], []))
    # A inconsistent / malformed / too many rows
    raises_value_error(lambda: calib.compile(["X"], [[1]], [0],
                                             [[1], [1]], [1, 2]))
    raises_value_error(lambda: calib.compile(["X"], [[1]], [0],
                                             [[1, 2]], [1]))
    raises_value_error(lambda: calib.compile(["X"], [[1]], [0],
                                             [[1]] * 9, [1] * 9))
    raises_value_error(lambda: calib.compile(["X"], [[1]], [0], [[1]], [1, 2]))
    raises_value_error(lambda: calib.compile(["X"], [[1]], [0, 0], [], []))


def t_invalid_evaluate():
    m = make_headline_model()
    L = {"X": {"X": 1}}
    raises_value_error(lambda: calib.evaluate("nope", {}, {}, [], L))
    raises_value_error(lambda: calib.evaluate(m, [], {}, [], L))
    raises_value_error(lambda: calib.evaluate(m, {"": {"X": 1}}, {}, [[1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"Z": 1}}, {"o": 1},
                                              [[1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"X": 1}}, {}, [[1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"X": 1}},
                                              {"o": 1, "p": 2}, [[1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"X": 1}}, {"o": 1.5},
                                              [[1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"X": 1}}, {"o": 1},
                                              [[1, 0], [0, 1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"o": {"X": 1}}, {"o": 1},
                                              [[-1]], L))
    raises_value_error(lambda: calib.evaluate(m, {"a": {"X": 1},
                                                  "b": {"Y": 1}},
                                              {"a": 1, "b": 1},
                                              [[1, 2], [2, 1]], L))
    raises_value_error(lambda: calib.evaluate(m, {}, {}, [], {}))
    raises_value_error(lambda: calib.evaluate(m, {}, {}, [],
                                              {"r%d" % i: {"X": 1}
                                               for i in range(9)}))
    raises_value_error(lambda: calib.evaluate(m, {}, {}, [],
                                              {"o": {"Z": 1}}))
    raises_value_error(lambda: calib.evaluate(m, {}, {}, [],
                                              {"o": {"X": "1/2/3"}}))
    raises_value_error(lambda: calib.evaluate(
        m, {"r%d" % i: {"X": 1} for i in range(9)},
        {"r%d" % i: 0 for i in range(9)},
        [[1 if i == j else 0 for j in range(9)] for i in range(9)], L))


def main():
    tests = [
        ("headline ok", t_headline_ok),
        ("headline inconsistent", t_headline_inconsistent),
        ("headline no observations", t_headline_no_observation),
        ("headline string numerics", t_headline_string_numerics),
        ("headline Fraction numerics", t_headline_fraction_numerics),
        ("model reuse stays independent", t_model_reuse_independent),
        ("scalar common noise", t_scalar_common_noise),
        ("scalar common-noise contradiction", t_scalar_common_noise_contradiction),
        ("scalar noiseless reading", t_scalar_noiseless),
        ("scalar mixed noise", t_scalar_mixed_noise),
        ("simple bayes update", t_simple_bayes),
        ("correlated full-rank R", t_correlated_full_rank_R),
        ("determined point", t_determined_point),
        ("redundant consistent A", t_redundant_consistent_A),
        ("mixed deterministic+stochastic", t_mixed_deterministic_and_stochastic),
        ("linear combination rows", t_linear_combination_rows),
        ("pure noise rows", t_pure_noise_rows),
        ("maximum sizes", t_maximum_sizes),
        ("invalid compile declarations", t_invalid_compile),
        ("invalid evaluate declarations", t_invalid_evaluate),
    ]
    for label, fn in tests:
        check(label, fn)
    print("-" * 60)
    if _failures:
        print("%d/%d checks FAILED" % (len(_failures), _count))
        return 1
    print("all %d checks passed" % _count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
