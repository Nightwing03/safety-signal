import io

from safety_signal.cli import main
from safety_signal.errors import DataError
from tests.fakes import make_events


def run(argv, factory):
    buf = io.StringIO()
    code = main(argv, source_factory=factory, out=buf)
    return code, buf.getvalue()


def test_output_has_drug_as_of_date_caveat_and_flag():
    code, text = run(["alpha", "--top", "5"], lambda args: make_events())
    assert code == 0
    assert "alpha" in text
    assert "data as of: 2026-01-01" in text
    assert "voluntary" in text
    assert "RASH" in text and "SIGNAL" in text
    assert "DRUG INEFFECTIVE" not in text


def test_unknown_drug_prints_a_message_not_a_crash():
    code, text = run(["nonexistent"], lambda args: make_events())
    assert code == 0
    assert "No reactions found" in text


def test_library_errors_become_exit_code_one(capsys):
    def factory(args):
        raise DataError("boom")

    code, _ = run(["alpha"], factory)
    assert code == 1
    assert "error: boom" in capsys.readouterr().err
