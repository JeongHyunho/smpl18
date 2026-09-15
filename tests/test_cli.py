import pytest

from smpl24 import __version__
from smpl24.cli import main


def test_version_flag_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"smpl24 {__version__}"


def test_no_arguments_is_a_no_op_exit_zero() -> None:
    assert main([]) == 0
