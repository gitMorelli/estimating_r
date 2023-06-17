#!/usr/bin/env python
# coding: utf-8

import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
import pandas as pd
import healpy as hp
from camb import model, initialpower
import useful_functions as uf
import pysm3
from fgbuster import (CMB, Dust, Synchrotron, basic_comp_sep,get_observation, get_instrument)
from fgbuster.visualization import corner_norm

instrument = get_instrument('LiteBIRD')
nside=16
kind_of_map='d1s1'
sync_freq_maps = get_observation(instrument, kind_of_map, noise=False, nside=nside, unit='uK_CMB')
home_dir="/home/amorelli/temporary_storage/"
np.savez(home_dir+kind_of_map+"_file",sync_freq_maps=sync_freq_maps)
