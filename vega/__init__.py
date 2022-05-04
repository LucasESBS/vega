"""VEGA: VAE Enhanced by Gene Annotations"""
import scvi

use_old = True

if scvi.__version__!='0.9.0':
    use_old = False

if use_old:
    from .old.vega_model import VEGA
    from .old.vega_count import VegaSCVI
    from .old.utils import *
    from .  import data
    from .old.plotting import *
else:
    from . import model, utils, data

__version__ = '0.0.2'
