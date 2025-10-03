import types

from vital_reader import _create_easyocr_reader


class _FakeReader:
    def __init__(self, languages, gpu=False, verbose=False):
        self.languages = languages
        self.gpu = gpu
        self.verbose = verbose


class _FakeEasyOCR(types.SimpleNamespace):
    Reader = _FakeReader


def test_create_easyocr_reader_without_torch_uses_cpu():
    reader = _create_easyocr_reader(_FakeEasyOCR, None)
    assert isinstance(reader, _FakeReader)
    assert reader.gpu is False
    assert reader.languages == ['en', 'ja']


def test_create_easyocr_reader_handles_broken_cuda(monkeypatch):
    class _FakeCuda:
        def is_available(self):  # pragma: no cover - executed in test
            raise RuntimeError('boom')

    class _FakeTorch:
        cuda = _FakeCuda()

    reader = _create_easyocr_reader(_FakeEasyOCR, _FakeTorch())
    assert isinstance(reader, _FakeReader)
    assert reader.gpu is False


def test_create_easyocr_reader_returns_none_when_reader_fails(monkeypatch):
    class _BrokenReader:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('no backend')

    broken_easyocr = types.SimpleNamespace(Reader=_BrokenReader)

    assert _create_easyocr_reader(broken_easyocr, None) is None
