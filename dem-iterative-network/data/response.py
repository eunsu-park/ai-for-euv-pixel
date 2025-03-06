import os

os.environ['HOME'] = "C:\\Users\\eunsupark"

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import hissw



temperature = np.linspace(0.1, 100, 1000) * u.MK
flags = ['temp', 'dn', 'timedepend_date', 'evenorm']
inputs = {'flags': flags, 'temperature': temperature}

ssw = hissw.Environment(ssw_packages=['sdo/aia'], ssw_paths=['aia'])
ssw_resp = ssw.run('calc_aia_response.pro', args=inputs)