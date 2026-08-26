from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allow installation to continue when no native compiler is available."""

    def run(self):
        try:
            super().run()
        except Exception as error:
            print(f"WARNING: optional Cython extension was not built: {error}")

    def build_extension(self, extension):
        try:
            super().build_extension(extension)
        except Exception as error:
            print(f"WARNING: optional Cython extension was not built: {error}")


try:
    from pybind11.setup_helpers import Pybind11Extension

    extensions = [
        Pybind11Extension(
            "c5tree._splitter_fast",
            ["c5tree/_splitter_fast.cpp"],
            cxx_std=17,
        )
    ]
except Exception as error:
    print(f"WARNING: optional C++ extension is unavailable: {error}")
    extensions = []


setup(ext_modules=extensions, cmdclass={"build_ext": OptionalBuildExt})
