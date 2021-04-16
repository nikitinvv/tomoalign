from pkg_resources import get_distribution, DistributionNotFound

from tomoalign.methods import *
# from tomoalign.gendata import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
