import sys
from types import ModuleType, SimpleNamespace
import hashlib
import dcf_xet_stream


class R2:
    def __init__(self): self.uploads=[]; self.aborted=False; self.completed=False
    def create_multipart_upload(self, **kw): return {"UploadId":"u"}
    def upload_part(self, PartNumber, Body, **kw):
        self.uploads.append((PartNumber, bytes(Body))); return {"ETag":str(PartNumber)}
    def complete_multipart_upload(self, **kw): self.completed=True
    def abort_multipart_upload(self, **kw): self.aborted=True


def _install(monkeypatch, chunks, size=10):
    class Stream:
        def __iter__(self): return iter(chunks)
        def cancel(self): pass
    class Group:
        def download_stream(self, info): return Stream()
    class Session:
        def new_download_stream_group(self, **kw): return Group()
    hx=ModuleType("hf_xet"); hx.XetFileInfo=lambda h,s:(h,s)
    monkeypatch.setitem(sys.modules,"hf_xet",hx)
    fd=ModuleType("huggingface_hub.file_download")
    fd.hf_hub_url=lambda *a,**k:"url"
    fd.get_hf_file_metadata=lambda *a,**k:SimpleNamespace(size=size,xet_file_data=SimpleNamespace(file_hash="hash",refresh_route="refresh"))
    monkeypatch.setitem(sys.modules,"huggingface_hub.file_download",fd)
    hd=ModuleType("huggingface_hub.utils._headers"); hd.build_hf_headers=lambda **k:{}
    monkeypatch.setitem(sys.modules,"huggingface_hub.utils._headers",hd)
    xu=ModuleType("huggingface_hub.utils._xet")
    xu.get_xet_session=lambda:Session()
    xu.refresh_xet_connection_info=lambda **k:SimpleNamespace(endpoint="e",access_token="t",expiration_unix_epoch=1)
    xu.xet_headers_without_auth=lambda h:{}
    monkeypatch.setitem(sys.modules,"huggingface_hub.utils._xet",xu)


def _artifact(data):
    return {"repository":"r","repository_path":"f","revision":"rev","r2_key":"k",
            "expected_size_bytes":len(data),"expected_sha256":hashlib.sha256(data).hexdigest()}


def test_stream_to_multipart(monkeypatch):
    data=b"abcdefghij"; _install(monkeypatch,[b"ab",b"cdefg",b"hij"],len(data))
    r2=R2(); size,digest=dcf_xet_stream.transfer_xet_stream_to_r2(_artifact(data),r2,"b",None,part_size=4,upload_workers=2)
    assert size==10 and digest==hashlib.sha256(data).hexdigest()
    assert r2.uploads==[(1,b"abcd"),(2,b"efgh"),(3,b"ij")]
    assert r2.completed and not r2.aborted


def test_corrupt_stream_aborts(monkeypatch):
    data=b"abcdefghij"; _install(monkeypatch,[b"abc"],len(data))
    r2=R2()
    import pytest
    with pytest.raises(dcf_xet_stream.XetStreamError):
        dcf_xet_stream.transfer_xet_stream_to_r2(_artifact(data),r2,"b",None,part_size=4)
    assert r2.aborted and not r2.completed
