import os
import shutil
import subprocess
from pathlib import Path

_READY = False


def ensure_compile_env() -> None:
    global _READY
    if _READY or os.name != "nt" or shutil.which("cl"):
        _READY = True
        return

    vcvars = _find_vcvars64()
    if vcvars is not None:
        _load_vcvars(vcvars)
    _READY = True


def _find_vcvars64() -> Path | None:
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        install_dir = _vswhere_install_dir(
            Path(program_files_x86)
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
        if install_dir is not None:
            vcvars = install_dir / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.exists():
                return vcvars

    for root in _visual_studio_roots():
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            vcvars = root / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.exists():
                return vcvars
    return None


def _vswhere_install_dir(vswhere: Path) -> Path | None:
    if not vswhere.exists():
        return None
    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    return Path(path) if result.returncode == 0 and path else None


def _visual_studio_roots() -> list[Path]:
    roots = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        path = os.environ.get(env_name)
        if path:
            roots.append(Path(path) / "Microsoft Visual Studio" / "2022")
    return roots


def _load_vcvars(vcvars: Path) -> None:
    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        capture_output=True,
        shell=True,
        text=True,
    )
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            os.environ[name] = value
