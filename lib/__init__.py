import os
from warnings import simplefilter
os.environ["NUMEXPR_MAX_THREADS"] = "32"
simplefilter(action="ignore", category=FutureWarning, append = True)
simplefilter(action="ignore", category=RuntimeWarning, append = True)