try:
    from skbuild import setup
except ImportError:
    print('pip must be updated to the latest version for the installer to work.\nRun "pip3 install --user --upgrade pip" to do this.')
    raise
try:
    from psutil import cpu_count
    psutil_found = True
except ImportError:
    psutil_found = False
import os

# Versioning information
# Only bump for stable releases or API breaks
major = ("COALDECODER_VERSION_MAJOR", "0")

# Bump for functionality additions that DO NOT break past APIs
minor = ("COALDECODER_VERSION_MINOR", "001")

# Bump for bugfixes that DO NOT break past APIs
patch = ("COALDECODER_VERSION_PATCH", "0")

name = ("COALDECODER_VERSION_NAME", "Agate")

# Check for cores and use the physical cores
if psutil_found:
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count(logical=False))

# Arguments to pass to CMake
cmake_major = "-D" + major[0] + "=" + major[1]
cmake_minor = "-D" + minor[0] + "=" + minor[1]
cmake_patch = "-D" + patch[0] + "=" + patch[1]
cmake_name = "-D" + name[0] + "=" + name[1]
cmake_args = [cmake_major, cmake_minor, cmake_patch, cmake_name]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="coaldecoder",
    version=major[1]+"."+minor[1]+"."+patch[1],
    author="Nate Pope",
    author_email="natep@uoregon.edu",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    project_urls={
        'Documentation': 'TODO',
        'Source': 'TODO',
        'Tracker': 'TODO',
    },
    packages=["coaldecoder"],
    package_dir={"": "src"},
    classifiers=[
        "TODO"
    ],
    python_requires='>=3.6.0',
    keywords='TODO',
    cmake_args=cmake_args,
    cmake_install_dir="src/coaldecoder",
    setup_requires=["setuptools", "wheel",
                      "scikit-build", "cmake", "ninja", "psutil"]
)

