from src.data import load_and_clean


def test_cleaning(tmp_path):
    row = "63,1,4,145,233,1,2,150,0,2.3,3,0,6,0\n67,1,4,160,286,0,2,108,1,1.5,2,3,3,2\n"
    p = tmp_path / "sample.data"
    p.write_text(row)
    df = load_and_clean(p)
    assert "target" in df.columns
    assert set(df.target) == {0, 1}
    assert len(df) == 2
