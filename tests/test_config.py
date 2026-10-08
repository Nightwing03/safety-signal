from safety_signal.config import load_api_key


def test_environment_wins_over_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("OPENFDA_API_KEY=fromfile\n")
    assert load_api_key({"OPENFDA_API_KEY": "fromenv"}, str(f)) == "fromenv"


def test_reads_file_and_strips_quotes(tmp_path):
    f = tmp_path / ".env"
    f.write_text('OTHER=1\nOPENFDA_API_KEY="quoted"\n')
    assert load_api_key({}, str(f)) == "quoted"


def test_missing_file_or_key_gives_none(tmp_path):
    assert load_api_key({}, str(tmp_path / "nope")) is None
    f = tmp_path / ".env"
    f.write_text("OPENFDA_API_KEY=\n")
    assert load_api_key({}, str(f)) is None
