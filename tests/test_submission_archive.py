import io
import os
import zipfile

from cua_speedrun.commands.dashboard_client import _submission_files_zip
from cua_speedrun.service.enqueue import zip_submission


def test_archives_accept_reproducible_build_timestamps(tmp_path):
    contents = {"init.py": b"print('init')\n", "agent.py": b"print('agent')\n"}
    for name, data in contents.items():
        path = tmp_path / name
        path.write_bytes(data)
        os.utime(path, (0, 0))
    archives = [
        _submission_files_zip(tmp_path / "init.py", tmp_path / "agent.py"),
        zip_submission(tmp_path),
    ]
    for data in archives:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert set(archive.namelist()) == set(contents)
            for name, expected in contents.items():
                assert archive.read(name) == expected
                assert archive.getinfo(name).date_time == (1980, 1, 1, 0, 0, 0)
