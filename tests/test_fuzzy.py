"""Fuzzy-inference correctness: membership functions, and that green time
responds sensibly to the queues.
"""

from _harness import check, summary

from cadence.fuzzy import (
    FuzzyGreenController, triangular, shoulder_low, shoulder_high,
)


def test_membership_functions():
    print("-- membership functions behave")
    check("triangle peaks at 1 at its centre", triangular(10, 4, 10, 16) == 1.0)
    check("triangle is 0 at its feet", triangular(4, 4, 10, 16) == 0.0
          and triangular(16, 4, 10, 16) == 0.0)
    check("triangle is 0.5 halfway up", abs(triangular(7, 4, 10, 16) - 0.5) < 1e-9)
    check("low shoulder is full below its knee", shoulder_low(2, 4, 10) == 1.0)
    check("high shoulder is full above its knee", shoulder_high(20, 10, 18) == 1.0)
    check("shoulders cross over the transition",
          0 < shoulder_low(7, 4, 10) < 1 and 0 < shoulder_high(14, 10, 18) < 1)


def test_green_responds_to_queues():
    print("-- green time rises with own queue, falls with the opposing queue")
    f = FuzzyGreenController()
    check("longer own queue -> longer green",
          f.infer(16, 4) > f.infer(8, 4) > f.infer(2, 4),
          f"{f.infer(16,4)}, {f.infer(8,4)}, {f.infer(2,4)}")
    check("longer opposing queue -> shorter green (own fixed)",
          f.infer(10, 16) <= f.infer(10, 2),
          f"{f.infer(10,16)} vs {f.infer(10,2)}")
    check("output stays within the green-time range",
          all(10 <= f.infer(a, b) <= 45 for a in (0, 8, 20) for b in (0, 8, 20)))


def test_defuzzification_default():
    print("-- with no queues the controller returns a sane default")
    f = FuzzyGreenController()
    g = f.infer(0, 0)
    check("empty intersection yields the short green", g == f.GREEN["short"],
          str(g))


def main() -> int:
    for t in (test_membership_functions, test_green_responds_to_queues,
              test_defuzzification_default):
        t()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
