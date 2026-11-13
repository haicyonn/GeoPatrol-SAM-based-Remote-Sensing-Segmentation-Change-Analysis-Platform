from pathlib import Path
import hashlib,sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


def test_native_cache_is_identical_and_never_silently_reuses_corruption(tmp_path):
    from sql_benchmark_io import cached_database
    source=tmp_path/'source.sqlite';source.write_bytes(b'known database bytes')
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    result=cached_database(source,expected,tmp_path/'cache')
    assert result.read_bytes()==source.read_bytes() and result!=source
    result.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Cache fingerprint'):
        cached_database(source,expected,tmp_path/'cache')


def test_source_change_is_rejected_before_cache_copy(tmp_path):
    from sql_benchmark_io import cached_database
    source=tmp_path/'source.sqlite';source.write_bytes(b'different')
    with pytest.raises(ValueError,match='Source fingerprint'):
        cached_database(source,'expected',tmp_path/'cache')
