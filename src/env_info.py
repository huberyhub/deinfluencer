import platform
import sys
import numpy
import scipy
import networkx
import matplotlib
import pkg_resources

installed_packages = pkg_resources.working_set
packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

for package in packages_list:
    print(package)

# Gather software environment details
environment_details = {
    "Operating System": f"{platform.system()} {platform.release()}",
    "Python Version": sys.version,
    "NumPy Version": numpy.__version__,
    "SciPy Version": scipy.__version__,
    "NetworkX Version": networkx.__version__,
    "Matplotlib Version": matplotlib.__version__
}

for key, value in environment_details.items():
    print(f"{key}: {value}")
