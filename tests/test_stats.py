import pytest

from safety_signal.stats import Table, Thresholds, chi_square, evaluate, prr


def test_prr_known_example():
    # a/(a+b) = 0.2, c/(c+d) = 0.1
    assert prr(Table(20, 80, 100, 900)) == pytest.approx(2.0)


def test_chi_square_known_example():
    assert chi_square(Table(20, 80, 100, 900)) == pytest.approx(9.3537, rel=1e-3)


def test_independent_table_has_prr_one_and_zero_chi_square():
    t = Table(10, 90, 100, 900)
    assert prr(t) == pytest.approx(1.0)
    assert chi_square(t) == pytest.approx(0.0, abs=1e-9)


def test_scaling_all_cells_keeps_prr_and_scales_chi_square():
    base, scaled = Table(20, 80, 100, 900), Table(200, 800, 1000, 9000)
    assert prr(scaled) == pytest.approx(prr(base))
    assert chi_square(scaled) == pytest.approx(10 * chi_square(base))


def test_undefined_values_return_none_not_errors():
    assert prr(Table(5, 5, 0, 100)) is None
    assert prr(Table(0, 0, 5, 5)) is None
    assert chi_square(Table(0, 0, 5, 5)) is None
    assert evaluate(Table(5, 5, 0, 100)).flagged is False


def test_negative_cells_rejected():
    with pytest.raises(ValueError):
        Table(-1, 1, 1, 1)


def test_flag_requires_minimum_case_count():
    assert evaluate(Table(2, 8, 10, 980)).flagged is False
    assert evaluate(Table(3, 7, 10, 980)).flagged is True


def test_flag_requires_prr_threshold():
    t = Table(20, 80, 100, 900)  # PRR 2.0, chi-square about 9.35
    assert evaluate(t).flagged is True
    assert evaluate(t, Thresholds(prr_min=3.0)).flagged is False


def test_flag_requires_chi_square_threshold():
    t = Table(20, 80, 100, 900)
    assert evaluate(t, Thresholds(chi2_min=50.0)).flagged is False
