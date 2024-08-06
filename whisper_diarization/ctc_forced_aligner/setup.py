from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.config.expand import find_packages
import os

output_directory = os.path.join(os.path.dirname(__file__), "ctc_forced_aligner")


ext_modules = [
    Pybind11Extension(
        ".ctc_forced_aligner",
        ["whisper_diarization/ctc_forced_aligner/ctc_forced_aligner/forced_align_impl.cpp"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="whisper-diarization",
    version="0.1.0",
    packages=find_packages(include=["whisper_diarization", "whisper_diarization.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    ext_package=".whisper_diarization.ctc_forced_aligner.ctc_forced_aligner",
)

os.makedirs(output_directory, exist_ok=True)
