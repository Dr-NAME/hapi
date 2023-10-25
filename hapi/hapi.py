# -*- coding: utf-8 -*-

'''
This module provides an access to the HITRAN data.
Data is downloaded and cached.

This module serves as a simple database manager front end.

API is aimed to be RESTful, which means that interaction
between local API and remote data-server will be held 
via sending RESTful queries (API->remote) and
receiving data preferably in text format (remote->API).

Object are supposed to be implemented by structures/dicts
as they are present in almost any programming language.

Trying to retain functional style for this API. 
'''

import sys
import json
import os, os.path
import re
from os import listdir
import numpy as np
from numpy import zeros, array, setdiff1d, ndarray, arange
from numpy import place, where, real, polyval
from numpy import sqrt, abs, exp, pi, log, sin, cos, tan
from numpy import convolve
from numpy import flipud
from numpy.fft import fft, fftshift
from numpy import linspace, floor
from numpy import any, minimum, maximum
from numpy import sort as npsort
from bisect import bisect
from warnings import warn, simplefilter
from time import time
from .tips import PYTIPS
from .dtype import ComplexType, IntegerType, FloatType64
from .constants import cZero, cBolts, cc, hh, cSqrtLn2divSqrtPi, cLn2, cSqrtLn2, cSqrt2Ln2
from .iso import ISO, ISO_ID, ISO_INDEX, ISO_ID_INDEX
# Enable warning repetitions
simplefilter('always', UserWarning)

# Python 3 compatibility
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

if 'io' in sys.modules: # define open using Linux-style line endings
    import io
    def open_(*args, **argv):
        argv.update(dict(newline='\n'))
        return io.open(*args, **argv)
else:
    open_ = open

# initialize global variables
VARIABLES = {}

VARIABLES['DEBUG'] = False
if VARIABLES['DEBUG']: warn('DEBUG is set to True!')

GLOBAL_DEBUG = False
if GLOBAL_DEBUG: warn('GLOBAL_DEBUG is set to True!')

FLAG_DEBUG_PROFILE = False
FLAG_DEBUG_LADDER = False

LOCAL_HOST = 'http://localhost'

# DEBUG switch
if GLOBAL_DEBUG:
   GLOBAL_HOST = LOCAL_HOST+':8000' # localhost
else:
   GLOBAL_HOST = 'http://hitran.org'

VARIABLES['PROXY'] = {}
# EXAMPLE OF PROXY:
# VARIABLES['PROXY'] = {'http': '127.0.0.1:80'}
   
# make it changeable
VARIABLES['GLOBAL_HOST'] = GLOBAL_HOST

# display the fetch URL (debug)
VARIABLES['DISPLAY_FETCH_URL'] = False

# In this "robust" version of arange the grid doesn't suffer 
# from the shift of the nodes due to error accumulation.
# This effect is pronounced only if the step is sufficiently small.
def arange_(lower, upper, step):
    npnt = floor((upper - lower)/step) + 1
    npnt = int(npnt) # cast to integer to avoid type errors
    upper_new = lower + step*(npnt - 1)
    if abs((upper-upper_new)-step) < 1e-10:
        upper_new += step
        npnt += 1    
    return linspace(lower, upper_new, npnt)

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# LOCAL DATABASE MANAGEMENT SYSTEM
# ---------------------------------------------------------------
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# DATABASE BACKEND: simple text files, parsed into a python lists
# Use a directory as a database. Each table is stored in a 
# separate text file. Parameters in text are position-fixed.

BACKEND_DATABASE_NAME_DEFAULT = '.'

VARIABLES['BACKEND_DATABASE_NAME'] = BACKEND_DATABASE_NAME_DEFAULT

# For this node local DB is schema-dependent!
LOCAL_TABLE_CACHE = {
   'sampletab' : { # table
      'header' : { # header
         'order' : ('column1', 'column2', 'column3'),
         'format' : {
            'column1' : '%10d',
            'column2' : '%20f',
            'column3' : '%30s' 
         },
         'default' : {
            'column1' : 0,
            'column2' : 0.0,
            'column3' : ''
         },
         'number_of_rows' : 3,
         'size_in_bytes' : None,
         'table_name' : 'sampletab',
         'table_type' : 'strict'
      }, # /header
      'data' : {
         'column1' : [1, 2, 3],
         'column2' : [10.5, 11.5, 12.5],
         'column3' : ['one', 'two', 'three']
      }, # /data
   } # /table
} # hash-map of tables

# FORMAT CONVERSION LAYER

# converts between TRANSPORT_FORMAT and OBJECT_FORMAT
HITRAN_FORMAT_160 = {
   'M'          : {'pos' :   1,   'len' :  2,   'format' : '%2d' },
   'I'          : {'pos' :   3,   'len' :  1,   'format' : '%1d' },
   'nu'         : {'pos' :   4,   'len' : 12,   'format' : '%12f'},
   'S'          : {'pos' :  16,   'len' : 10,   'format' : '%10f'},
   'R'          : {'pos' :  26,   'len' :  0,   'format' : '%0f' },
   'A'          : {'pos' :  26,   'len' : 10,   'format' : '%10f'},
   'gamma_air'  : {'pos' :  36,   'len' :  5,   'format' : '%5f' },
   'gamma_self' : {'pos' :  41,   'len' :  5,   'format' : '%5f' },
   'E_'         : {'pos' :  46,   'len' : 10,   'format' : '%10f'},
   'n_air'      : {'pos' :  56,   'len' :  4,   'format' : '%4f' },
   'delta_air'  : {'pos' :  60,   'len' :  8,   'format' : '%8f' },
   'V'          : {'pos' :  68,   'len' : 15,   'format' : '%15s'},
   'V_'         : {'pos' :  83,   'len' : 15,   'format' : '%15s'},
   'Q'          : {'pos' :  98,   'len' : 15,   'format' : '%15s'},
   'Q_'         : {'pos' : 113,   'len' : 15,   'format' : '%15s'},
   'Ierr'       : {'pos' : 128,   'len' :  6,   'format' : '%6s' },
   'Iref'       : {'pos' : 134,   'len' : 12,   'format' : '%12s'},
   'flag'       : {'pos' : 146,   'len' :  1,   'format' : '%1s' },
   'g'          : {'pos' : 147,   'len' :  7,   'format' : '%7f' },
   'g_'         : {'pos' : 154,   'len' :  7,   'format' : '%7f' }
}

# This should be generating from the server's response
HITRAN_DEFAULT_HEADER = {
  "table_type": "column-fixed", 
  "size_in_bytes": -1, 
  "table_name": "###", 
  "number_of_rows": -1, 
  "order": [
    "molec_id", 
    "local_iso_id", 
    "nu", 
    "sw", 
    "a", 
    "gamma_air", 
    "gamma_self", 
    "elower", 
    "n_air", 
    "delta_air", 
    "global_upper_quanta", 
    "global_lower_quanta", 
    "local_upper_quanta", 
    "local_lower_quanta", 
    "ierr", 
    "iref", 
    "line_mixing_flag", 
    "gp", 
    "gpp"
  ],
  "format": {
    "a": "%10.3E", 
    "gamma_air": "%5.4f", 
    "gp": "%7.1f", 
    "local_iso_id": "%1d", 
    "molec_id": "%2d", 
    "sw": "%10.3E", 
    "local_lower_quanta": "%15s", 
    "local_upper_quanta": "%15s", 
    "gpp": "%7.1f", 
    "elower": "%10.4f", 
    "n_air": "%4.2f", 
    "delta_air": "%8.6f", 
    "global_upper_quanta": "%15s", 
    "iref": "%12s", 
    "line_mixing_flag": "%1s", 
    "ierr": "%6s", 
    "nu": "%12.6f", 
    "gamma_self": "%5.3f", 
    "global_lower_quanta": "%15s"
  }, 
  "default": {
    "a": 0.0, 
    "gamma_air": 0.0, 
    "gp": "FFF", 
    "local_iso_id": 0, 
    "molec_id": 0, 
    "sw": 0.0, 
    "local_lower_quanta": "000", 
    "local_upper_quanta": "000", 
    "gpp": "FFF", 
    "elower": 0.0, 
    "n_air": 0.0, 
    "delta_air": 0.0, 
    "global_upper_quanta": "000", 
    "iref": "EEE", 
    "line_mixing_flag": "EEE", 
    "ierr": "EEE", 
    "nu": 0.0, 
    "gamma_self": 0.0, 
    "global_lower_quanta": "000"
  },
  "description": {
    "a": "Einstein A-coefficient in s-1", 
    "gamma_air": "Air-broadened Lorentzian half-width at half-maximum at p = 1 atm and T = 296 K", 
    "gp": "Upper state degeneracy", 
    "local_iso_id": "Integer ID of a particular Isotopologue, unique only to a given molecule, in order or abundance (1 = most abundant)", 
    "molec_id": "The HITRAN integer ID for this molecule in all its isotopologue forms", 
    "sw": "Line intensity, multiplied by isotopologue abundance, at T = 296 K", 
    "local_lower_quanta": "Rotational, hyperfine and other quantum numbers and labels for the lower state of a transition", 
    "local_upper_quanta": "Rotational, hyperfine and other quantum numbers and labels for the upper state of a transition", 
    "gpp": "Lower state degeneracy", 
    "elower": "Lower-state energy", 
    "n_air": "Temperature exponent for the air-broadened HWHM", 
    "delta_air": "Pressure shift induced by air, referred to p=1 atm", 
    "global_upper_quanta": "Electronic and vibrational quantum numbers and labels for the upper state of a transition", 
    "iref": "Ordered list of reference identifiers for transition parameters", 
    "line_mixing_flag": "A flag indicating the presence of additional data and code relating to line-mixing", 
    "ierr": "Ordered list of indices corresponding to uncertainty estimates of transition parameters", 
    "nu": "Transition wavenumber", 
    "gamma_self": "Self-broadened HWHM at 1 atm pressure and 296 K", 
    "global_lower_quanta": "Electronic and vibrational quantum numbers and labels for the lower state of a transition"
  },
  "position": {
    "molec_id": 0,
    "local_iso_id": 2,
    "nu": 3,
    "sw": 15,
    "a": 25,
    "gamma_air": 35,
    "gamma_self": 40,
    "elower": 45,
    "n_air": 55,
    "delta_air": 59,
    "global_upper_quanta": 67,
    "global_lower_quanta": 82,
    "local_upper_quanta": 97,
    "local_lower_quanta": 112,
    "ierr": 127,
    "iref": 133,
    "line_mixing_flag": 145,
    "gp": 146,
    "gpp": 153,
  },
  'cast': {
    "molec_id": "uint8",
    "local_iso_id": "uint8",
    "nu": "float32",
    "sw": "float62",
    "a": "float62",
    "gamma_air": "float16",
    "gamma_self": "float16",
    "elower": "float32",
    "n_air": "float16",
    "delta_air": "float16",
    "global_upper_quanta": "str",
    "global_lower_quanta": "str",
    "local_upper_quanta": "str",
    "local_upper_quanta": "str",
    "ierr": "str",
    "iref": "str",
    "line_mixing_flag": "str",
    "gp": "int16",
    "gpp": "int16",  
  }
}

class CaseInsensitiveDict(dict):
    """
    https://gist.github.com/m000/acbb31b9eca92c1da795 (c) Manolis Stamatogiannakis.
    """
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()
    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)
    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))
    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))
    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(self.__class__._k(key))
    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)
    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)
    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)
    def update(self, E, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))
    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)

CaselessDict = CaseInsensitiveDict
            
PARAMETER_META = CaselessDict(
{
  "global_iso_id" : {
    "default_fmt" : "%5d",
  },
  "molec_id" : {
    "default_fmt" : "%2d",
  },
  "local_iso_id" : {
    "default_fmt" : "%1d",
  },
  "nu" : {
    "default_fmt" : "%12.6f",
  },
  "sw" : {
    "default_fmt" : "%10.3e",
  },
  "a" : {
    "default_fmt" : "%10.3e",
  },
  "gamma_air" : {
    "default_fmt" : "%6.4f",
  },
  "gamma_self" : {
    "default_fmt" : "%5.3f",
  },
  "n_air" : {
    "default_fmt" : "%7.4f",
  },
  "delta_air" : {
    "default_fmt" : "%9.6f",
  },
  "elower" : {
    "default_fmt" : "%10.4f",
  },
  "gp" : {
    "default_fmt" : "%5d",
  },
  "gpp" : {
    "default_fmt" : "%5d",
  },
  "global_upper_quanta" : {
    "default_fmt" : "%15s",
  },
  "global_lower_quanta" : {
    "default_fmt" : "%15s",
  },
  "local_upper_quanta" : {
    "default_fmt" : "%15s",
  },
  "local_lower_quanta" : {
    "default_fmt" : "%15s",
  },
  "line_mixing_flag" : {
    "default_fmt" : "%1s",
  },
  "ierr" : {
    "default_fmt" : "%s",
  },
  "iref" : {
    "default_fmt" : "%s",
  },
  "deltap_air" : {
    "default_fmt" : "%10.3e",
  },
  "n_self" : {
    "default_fmt" : "%7.4f",
  },
  "delta_self" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_self" : {
    "default_fmt" : "%10.3e",
  },
  "SD_air" : {
    "default_fmt" : "%9.6f",
  },
  "SD_self" : {
    "default_fmt" : "%9.6f",
  },
  "beta_g_air" : {
    "default_fmt" : "%9.6f",
  },
  "y_self" : {
    "default_fmt" : "%10.3e",
  },
  "y_air" : {
    "default_fmt" : "%10.3e",
  },
  "statep" : {
    "default_fmt" : "%256s",
  },
  "statepp" : {
    "default_fmt" : "%256s",
  },
  "beta_g_self" : {
    "default_fmt" : "%9.6f",
  },
  "trans_id" : {
    "default_fmt" : "%12d",
  },
  "par_line" : {
    "default_fmt" : "%160s",
  },
  "gamma_H2" : {
    "default_fmt" : "%6.4f",
  },
  "n_H2" : {
    "default_fmt" : "%7.4f",
  },
  "delta_H2" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_H2" : {
    "default_fmt" : "%10.3e",
  },
  "gamma_He": {
    "default_fmt" : "%6.4f",
  },
  "n_He" : {
    "default_fmt" : "%7.4f",
  },
  "delta_He" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_CO2" : {
    "default_fmt" : "%6.4f",
  },
  "n_CO2" : {
    "default_fmt" : "%7.4f",
  },
  "delta_CO2" : {
    "default_fmt" : "%9.6f",
  }, 
  "gamma_HT_0_self_50" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_self_50" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_self_50" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_self_50" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_self_50" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_self_50" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_self_150" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_self_150" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_self_150" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_self_150" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_self_150" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_self_150" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_self_296" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_self_296" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_self_700" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_self_700" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_self_700" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_self_700" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_self_700" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_self_700" : {
    "default_fmt" : "%9.6f",
  },
  "nu_HT_self" : {
    "default_fmt" : "%6.4f",
  },
  "kappa_HT_self" : {
    "default_fmt" : "%9.6f",
  },
  "eta_HT_self" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_air_50" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_air_50" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_air_50" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_air_50" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_air_50" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_air_50" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_air_150" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_air_150" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_air_150" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_air_150" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_air_150" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_air_150" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_air_296" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_air_296" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_0_air_700" : {
    "default_fmt" : "%6.4f",
  },
  "n_HT_air_700" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_HT_2_air_700" : {
    "default_fmt" : "%6.4f",
  },
  "delta_HT_0_air_700" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_HT_air_700" : {
    "default_fmt" : "%9.6f",
  },
  "delta_HT_2_air_700" : {
    "default_fmt" : "%9.6f",
  },
  "nu_HT_air" : {
    "default_fmt" : "%6.4f",
  },
  "kappa_HT_air" : {
    "default_fmt" : "%9.6f",
  },
  "eta_HT_air" : {
    "default_fmt" : "%9.6f",
  },  
  "gamma_H2O" : {
    "default_fmt" : "%6.4f",
  },
  "n_H2O" : {
    "default_fmt" : "%9.6f",
  },
  "Y_SDV_air_296" : {
    "default_fmt" : "%10.3e",
  },
  "Y_SDV_self_296" : {
    "default_fmt" : "%10.3e",
  },
  "n_Y_SDV_air_296" : {
    "default_fmt" : "%6.4e",
  },
  "n_Y_SDV_self_296" : {
    "default_fmt" : "%6.4e",
  },
  "Y_HT_air_296" : {
    "default_fmt" : "%10.3e",
  },
  "Y_HT_self_296" : {
    "default_fmt" : "%10.3e",
  },
  "gamma_SDV_0_air_296" : {
    "default_fmt" : "%6.4f",
  },
  "gamma_SDV_0_self_296" : {
    "default_fmt" : "%6.4f",
  },
  "n_SDV_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "n_SDV_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "gamma_SDV_2_air_296" : {
    "default_fmt" : "%6.4f",
  },
  "gamma_SDV_2_self_296" : {
    "default_fmt" : "%6.4f",
  },
  "n_gamma_SDV_2_air_296" : {
    "default_fmt" : "%6.4f",
  },
  "n_gamma_SDV_2_self_296" : {
    "default_fmt" : "%6.4f",
  },
  "delta_SDV_0_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "delta_SDV_0_self_296" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_SDV_air_296" : {
    "default_fmt" : "%9.6f",
  },
  "deltap_SDV_self_296" : {
    "default_fmt" : "%9.6f",
  },
})

def getFullTableAndHeaderName(TableName, ext='data'):
    flag_abspath = os.path.isabs(TableName) # check if the supplied table name already contains absolute path
    fullpath_data = TableName + '.' + ext
    if not flag_abspath:
        fullpath_data = os.path.join(VARIABLES['BACKEND_DATABASE_NAME'], fullpath_data)
    if not os.path.isfile(fullpath_data):
        fullpath_data = VARIABLES['BACKEND_DATABASE_NAME'] + '/' + TableName + '.par'
        if not os.path.isfile(fullpath_data) and TableName!='sampletab':
            raise Exception('Lonely header \"%s\"' % fullpath_data)
    fullpath_header = TableName + '.header'
    if not flag_abspath:
        fullpath_header = os.path.join(VARIABLES['BACKEND_DATABASE_NAME'], fullpath_header)
    return fullpath_data, fullpath_header

def getParameterFormat(ParameterName, TableName):
    return LOCAL_TABLE_CACHE[TableName]['header']['format']

def getTableHeader(TableName):
    return LOCAL_TABLE_CACHE[TableName]['header']

def getRowObject(RowID, TableName):
    """return RowObject from TableObject in CACHE"""
    RowObject = []
    for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_value = LOCAL_TABLE_CACHE[TableName]['data'][par_name][RowID]
        par_format = LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        RowObject.append((par_name, par_value, par_format))
    return RowObject

# INCREASE ROW COUNT
def addRowObject(RowObject, TableName):
    for par_name, par_value, par_format in RowObject:
        LOCAL_TABLE_CACHE[TableName]['data'][par_name].append(par_value)

def setRowObject(RowID, RowObject, TableName):
    number_of_rows = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    if RowID >= 0 and RowID < number_of_rows:
       for par_name, par_value, par_format in RowObject:
           LOCAL_TABLE_CACHE[TableName]['data'][par_name][RowID] = par_value
    else:
       # !!! XXX ATTENTION: THIS IS A TEMPORARY INSERTION XXX !!!
       LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows'] += 1
       addRowObject(RowObject, TableName)

def getDefaultRowObject(TableName):
    """get a default RowObject from a table"""
    RowObject = []
    for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_value = LOCAL_TABLE_CACHE[TableName]['header']['default'][par_name]
        par_format = LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        RowObject.append((par_name, par_value, par_format))
    return RowObject

def subsetOfRowObject(ParameterNames, RowObject):
    """
    return a subset of RowObject according to 
    RowObjectNew = []
    for par_name, par_value, par_format in RowObject:
         if par_name in ParameterNames:
            RowObjectNew.append((par_name, par_value, par_format))
    return RowObjectNew
    """
    dct = {}
    for par_name, par_value, par_format in RowObject:
        dct[par_name] = (par_name, par_value, par_format)
    RowObjectNew = []
    for par_name in ParameterNames:
        RowObjectNew.append(dct[par_name])
    return RowObjectNew

FORMAT_PYTHON_REGEX = '^\%(\d*)(\.(\d*))?([edfsEDFS])$'

# Fortran string formatting
#  based on a pythonic format string
def formatString(par_format, par_value, lang='FORTRAN'):
    # Fortran format rules:
    #  %M.NP
    #        M - total field length (optional)
    #             (minus sign included in M)
    #        . - decimal ceparator (optional)
    #        N - number of digits after . (optional)
    #        P - [dfs] int/float/string
    # PYTHON RULE: if N is abcent, default value is 6
    regex = FORMAT_PYTHON_REGEX
    (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
    if type(par_value) is np.ma.core.MaskedConstant:
        result = '%%%ds' % lng % '#'
        return result
    result = par_format % par_value
    if ty.lower() in set(['f', 'e']):
       lng = int(lng) if lng else 0
       lngpnt = int(lngpnt) if lngpnt else 0
       result = par_format % par_value
       res = result.strip()
       if lng == lngpnt + 1:
          if res[0:1] == '0':
             result = '%%%ds' % lng % res[1:]
       if par_value < 0:
          if res[1:2] == '0':
             result = '%%%ds' % lng % (res[0:1]+res[2:])
    return result

def putRowObjectToString(RowObject):
    # serialize RowObject to string
    # TODO: support different languages (C, Fortran)
    output_string = ''
    for par_name, par_value, par_format in RowObject:
        # Python formatting
        #output_string += par_format % par_value
        # Fortran formatting
        #print 'par_name, par_value, par_format: '+str((par_name, par_value, par_format))
        output_string += formatString(par_format, par_value)
    return output_string

# Parameter nicknames are hard-coded.
PARAMETER_NICKNAMES = {
    "a": "A", 
    "gamma_air": "gair", 
    "gp": "g", 
    "local_iso_id": "I", 
    "molec_id": "M", 
    "sw": "S", 
    "local_lower_quanta": "Q_", 
    "local_upper_quanta": "Q", 
    "gpp": "g_", 
    "elower": "E_", 
    "n_air": "nair", 
    "delta_air": "dair", 
    "global_upper_quanta": "V", 
    "iref": "Iref", 
    "line_mixing_flag": "f", 
    "ierr": "ierr", 
    "nu": "nu", 
    "gamma_self": "gsel", 
    "global_lower_quanta": "V_"
}  

def putTableHeaderToString(TableName):
    output_string = ''
    regex = FORMAT_PYTHON_REGEX
    for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_format = LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
        fmt = '%%%ss' % lng
        try:
            par_name_short = PARAMETER_NICKNAMES[par_name]
        except:
            par_name_short = par_name
        #output_string += fmt % par_name
        output_string += (fmt % par_name_short)[:int(lng)]
    return output_string

def getRowObjectFromString(input_string, TableName):
    # restore RowObject from string, get formats and names in TableName
    #print 'getRowObjectFromString:'
    pos = 0
    RowObject = []
    for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_format = LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        regex = FORMAT_PYTHON_REGEX
        (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
        lng = int(lng)
        par_value = input_string[pos:(pos+lng)]
        if ty == 'd': # integer value
           par_value = int(par_value)
        elif ty.lower() in set(['e', 'f']): # float value
           par_value = float(par_value)
        elif ty == 's': # string value
           pass # don't strip string value
        else:
           print('err1')
           raise Exception('Format \"%s\" is unknown' % par_format)
        RowObject.append((par_name, par_value, par_format))
        pos += lng
    # Do the same but now for extra (comma-separated) parameters
    if 'extra' in set(LOCAL_TABLE_CACHE[TableName]['header']):
        csv_chunks = input_string.split(LOCAL_TABLE_CACHE[TableName]['header'].\
                                        get('extra_separator', ', '))
        # Disregard the first "column-fixed" container if it presents:
        if LOCAL_TABLE_CACHE[TableName]['header'].get('order', []):
            pos = 1
        else:
            pos = 0
        for par_name in LOCAL_TABLE_CACHE[TableName]['header']['extra']:
            par_format = LOCAL_TABLE_CACHE[TableName]['header']['extra_format'][par_name]
            regex = FORMAT_PYTHON_REGEX
            (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
            lng = int(lng) 
            par_value = csv_chunks[pos]
            if ty == 'd': # integer value
                try:
                    par_value = int(par_value)
                except ValueError:
                    par_value = np.nan
            elif ty.lower() in set(['e', 'f']): # float value
                try:
                    par_value = float(par_value)
                except ValueError:
                    #par_value = 0.0
                    par_value = np.nan
            elif ty == 's': # string value
                pass # don't strip string value
            else:
                print('err')
                raise Exception('Format \"%s\" is unknown' % par_format)
            RowObject.append((par_name, par_value, par_format))
            pos += 1   
    return RowObject

# Conversion between OBJECT_FORMAT and STORAGE_FORMAT
# This will substitute putTableToStorage and getTableFromStorage
def cache2storage(TableName):
    try:
       os.mkdir(VARIABLES['BACKEND_DATABASE_NAME'])
    except:
       pass

    fullpath_data = VARIABLES['BACKEND_DATABASE_NAME'] + '/' + TableName + '.data' # bugfix
    fullpath_header = VARIABLES['BACKEND_DATABASE_NAME'] + '/' + TableName + '.header' # bugfix
    OutfileData = open(fullpath_data, 'w')
    OutfileHeader = open(fullpath_header, 'w')
    # write table data
    line_count = 1
    line_number = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    for RowID in range(0, line_number):
        line_count += 1
        RowObject = getRowObject(RowID, TableName)
        raw_string = putRowObjectToString(RowObject)
        OutfileData.write(raw_string+'\n')
    # write table header
    TableHeader = getTableHeader(TableName)
    OutfileHeader.write(json.dumps(TableHeader, indent=2))
    
def storage2cache(TableName, cast=True, ext=None, nlines=None, pos=None):
    """ edited by NHL
    TableName: name of the HAPI table to read in
    ext: file extension
    nlines: number of line in the block; if None, read all line at once 
    pos: file position to seek
    """
    if nlines is not None:
        print('WARNING: storage2cache is reading the block of maximum %d lines'%nlines)
    fullpath_data, fullpath_header = getFullTableAndHeaderName(TableName, ext)
    if TableName in LOCAL_TABLE_CACHE and \
       'filehandler' in LOCAL_TABLE_CACHE[TableName] and \
       LOCAL_TABLE_CACHE[TableName]['filehandler'] is not None:
        InfileData = LOCAL_TABLE_CACHE[TableName]['filehandler']
    else:
        InfileData = open_(fullpath_data, 'r')
    InfileHeader = open(fullpath_header, 'r')
    #try:
    header_text = InfileHeader.read()
    try:
        Header = json.loads(header_text)
    except:
        print('HEADER:')
        print(header_text)
        raise Exception('Invalid header')
    LOCAL_TABLE_CACHE[TableName] = {}
    LOCAL_TABLE_CACHE[TableName]['header'] = Header
    LOCAL_TABLE_CACHE[TableName]['data'] = CaselessDict()
    LOCAL_TABLE_CACHE[TableName]['filehandler'] = InfileData
    # Check if Header['order'] and Header['extra'] contain
    #  parameters with same names, raise exception if true.
    #intersct = set(Header['order']).intersection(set(Header.get('extra', [])))
    intersct = set(Header.get('order', [])).intersection(set(Header.get('extra', [])))
    if intersct:
        raise Exception('Parameters with the same names: {}'.format(intersct))
    # initialize empty data to avoid problems
    glob_order = []; glob_format = {}; glob_default = {}
    if "order" in LOCAL_TABLE_CACHE[TableName]['header'].keys():
        glob_order += LOCAL_TABLE_CACHE[TableName]['header']['order']
        glob_format.update(LOCAL_TABLE_CACHE[TableName]['header']['format'])
        glob_default.update(LOCAL_TABLE_CACHE[TableName]['header']['default'])
        for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
            LOCAL_TABLE_CACHE[TableName]['data'][par_name] = []
    if "extra" in LOCAL_TABLE_CACHE[TableName]['header'].keys():
        glob_order += LOCAL_TABLE_CACHE[TableName]['header']['extra']
        glob_format.update(LOCAL_TABLE_CACHE[TableName]['header']['extra_format'])
        for par_name in LOCAL_TABLE_CACHE[TableName]['header']['extra']:
            glob_default[par_name] = PARAMETER_META[par_name]['default_fmt']
            LOCAL_TABLE_CACHE[TableName]['data'][par_name] = []
    
    header = LOCAL_TABLE_CACHE[TableName]['header']
    if 'extra' in header and header['extra']:
        line_count = 0
        flag_EOF = False
        while True:
            if nlines is not None and line_count >= nlines: break
            line = InfileData.readline()
            if line == '': # end of file is represented by an empty string
                flag_EOF = True
                break 
            try:
                RowObject = getRowObjectFromString(line, TableName)
                line_count += 1
            except:
                continue
            addRowObject(RowObject, TableName)

        LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows'] = line_count
    else:
        quantities = header['order']
        formats = [header['format'][qnt] for qnt in quantities]
        types = {'d':int, 'f':float, 'E':float, 's':str}
        converters = []
        end = 0
        for qnt, fmt in zip(quantities, formats):
            # pre-defined positions are needed to skip the existing parameters in headers (new feature)
            if 'position' in header:
                start = header['position'][qnt]
            else:
                start = end
            dtype = types[fmt[-1]]
            aux = fmt[fmt.index('%')+1:-1]
            if '.' in aux:
                aux = aux[:aux.index('.')]
            size = int(aux)
            end = start + size
            def cfunc(line, dtype=dtype, start=start, end=end, qnt=qnt):
                if dtype == float:
                    try:
                        return dtype(line[start:end])
                    except ValueError: # possible D exponent instead of E 
                        try:
                            return dtype(line[start:end].replace('D', 'E'))
                        except ValueError: # this is a special case and it should not be in the main version tree!
                            # Dealing with the weird and unparsable intensity format such as "2.700-164, i.e with no E or D characters.
                            res = re.search('(\d\.\d\d\d)\-(\d\d\d)', line[start:end])
                            if res:
                                return dtype(res.group(1)+'E-'+res.group(2))
                            else:
                                raise Exception('PARSE ERROR: unknown format of the par value (%s)'%line[start:end])
                elif dtype == int and qnt == 'local_iso_id':
                    if line[start:end] == '0':
                        return 10
                    try:
                        return dtype(line[start:end])
                    except ValueError:
                        # convert letters to numbers: A->11, B->12, etc... ; .par file must be in ASCII or Unicode.
                        return 11 + ord(line[start:end]) - ord('A')
                else:
                    return dtype(line[start:end])

            converters.append(cfunc)

        flag_EOF = False
        line_count = 0
        data_matrix = []
        while True:
            if nlines is not None and line_count >= nlines: break
            line = InfileData.readline()
            if line == '': # end of file is represented by an empty string
                flag_EOF = True
                break 
            data_matrix.append([cvt(line) for cvt in converters])
            line_count += 1
        data_columns = zip(*data_matrix)
        for qnt, col in zip(quantities, data_columns):
            if type(col[0]) in {int, float}:
                LOCAL_TABLE_CACHE[TableName]['data'][qnt] = np.array(col) # new code
            else:
                LOCAL_TABLE_CACHE[TableName]['data'][qnt].extend(col) # old code

        header['number_of_rows'] = line_count = (len(LOCAL_TABLE_CACHE[TableName]['data'][quantities[0]]))
            
    # Convert all columns to numpy arrays
    par_names = LOCAL_TABLE_CACHE[TableName]['header']['order']
    if 'extra' in header and header['extra']:
        par_names += LOCAL_TABLE_CACHE[TableName]['header']['extra']
    for par_name in par_names:
        column = LOCAL_TABLE_CACHE[TableName]['data'][par_name]
        LOCAL_TABLE_CACHE[TableName]['data'][par_name] = np.array(column)                    
            
    # Additionally: convert numeric arrays in "extra" part of the LOCAL_TABLE_CACHE to masked arrays.
    # This is done to avoid "nan" values in the arithmetic operations involving these columns.
    if 'extra' in header and header['extra']:
        for par_name in LOCAL_TABLE_CACHE[TableName]['header']['extra']:
            par_format = LOCAL_TABLE_CACHE[TableName]['header']['extra_format'][par_name]
            regex = FORMAT_PYTHON_REGEX
            (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
            if ty.lower() in ['d', 'e', 'f']:
                column = LOCAL_TABLE_CACHE[TableName]['data'][par_name]
                colmask = np.isnan(column)
                LOCAL_TABLE_CACHE[TableName]['data'][par_name] = np.ma.array(column, mask=colmask)
    
    # Delete all character-separated values, treat them as column-fixed.
    try:
        del LOCAL_TABLE_CACHE[TableName]['header']['extra']
        del LOCAL_TABLE_CACHE[TableName]['header']['extra_format']
        del LOCAL_TABLE_CACHE[TableName]['header']['extra_separator']
    except:
        pass
    # Update header.order/format with header.extra/format if exist.
    LOCAL_TABLE_CACHE[TableName]['header']['order'] = glob_order
    LOCAL_TABLE_CACHE[TableName]['header']['format'] = glob_format
    LOCAL_TABLE_CACHE[TableName]['header']['default'] = glob_default
    if flag_EOF:
        InfileData.close()
        LOCAL_TABLE_CACHE[TableName]['filehandler'] = None
    InfileHeader.close()
    print('                     Lines parsed: %d' % line_count)
    return flag_EOF    
    
# / FORMAT CONVERSION LAYER    
    
def getTableNamesFromStorage(StorageName):
    file_names = listdir(StorageName)
    table_names = []
    for file_name in file_names:
        matchObject = re.search('(.+)\.header$', file_name)
        if matchObject:
           table_names.append(matchObject.group(1))
    return table_names

# FIX POSSIBLE BUG: SIMILAR NAMES OF .PAR AND .DATA FILES
# BUG FIXED BY INTRODUCING A PRIORITY:
#   *.data files have more priority than *.par files
#   See getFullTableAndHeaderName function for explanation
def scanForNewParfiles(StorageName):
    file_names = listdir(StorageName)
    headers = {} # without extensions!
    parfiles_without_header = []
    for file_name in file_names:
        # create dictionary of unique headers
        try:
            fname, fext = re.search('(.+)\.(\w+)', file_name).groups()
        except:
            continue
        if fext == 'header':
            headers[fname] = True
    for file_name in file_names:
        # check if extension is 'par' and the header is absent
        try:
            fname, fext = re.search('(.+)\.(\w+)', file_name).groups()
        except:
            continue
        if fext == 'par' and fname not in headers:
            parfiles_without_header.append(fname)
    return parfiles_without_header

def createHeader(TableName):
    fname = TableName + '.header'
    fp = open(VARIABLES['BACKEND_DATABASE_NAME']+'/'+fname, 'w')
    if os.path.isfile(TableName):
        raise Exception('File \"%s\" already exists!' % fname)
    fp.write(json.dumps(HITRAN_DEFAULT_HEADER, indent=2))
    fp.close()

def loadCache():
    print('Using '+VARIABLES['BACKEND_DATABASE_NAME']+'\n')
    LOCAL_TABLE_CACHE = {}
    table_names = getTableNamesFromStorage(VARIABLES['BACKEND_DATABASE_NAME'])
    parfiles_without_header = scanForNewParfiles(VARIABLES['BACKEND_DATABASE_NAME'])
    # create headers for new parfiles
    for tab_name in parfiles_without_header:
        # get name without 'par' extension
        createHeader(tab_name)
        table_names.append(tab_name)
    for TableName in table_names:
        print(TableName)
        storage2cache(TableName)

def saveCache():
    try:
        # delete query buffer
        del LOCAL_TABLE_CACHE[QUERY_BUFFER]
    except:
        pass
    for TableName in LOCAL_TABLE_CACHE:
        print(TableName)
        cache2storage(TableName)

# DB backend level, start transaction
def databaseBegin(db=None):
    if db:
       VARIABLES['BACKEND_DATABASE_NAME'] = db
    else:
       VARIABLES['BACKEND_DATABASE_NAME'] = BACKEND_DATABASE_NAME_DEFAULT
    if not os.path.exists(VARIABLES['BACKEND_DATABASE_NAME']):
       os.mkdir(VARIABLES['BACKEND_DATABASE_NAME'])
    loadCache()

# DB backend level, end transaction
def databaseCommit():
    saveCache()

# ----------------------------------------------------
# ----------------------------------------------------
# CONDITIONS
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
# hierarchic query.condition language:
# Conditions: CONS = ('and', ('=', 'p1', 'p2'), ('<', 'p1', 13))
# String literals are distinguished from variable names 
#  by using the operation ('STRING', 'some_string')
# ----------------------------------------------------

# necessary conditions for hitranonline:
SAMPLE_CONDITIONS = ('AND', ('SET', 'internal_iso_id', [1, 2, 3, 4, 5, 6]), ('>=', 'nu', 0), ('<=', 'nu', 100))

# sample hitranonline protocol
# http://hitran.cloudapp.net/lbl/5?output_format_id=1&iso_ids_list=5&numin=0&numax=100&access=api&key=e20e4bd3-e12c-4931-99e0-4c06e88536bd

CONDITION_OPERATIONS = set(['AND', 'OR', 'NOT', 'RANGE', 'IN', '<', '>', '<=', '>=', ' == ', '!=', 'LIKE', 'STR', '+', '-', '*', '/', 'MATCH', 'SEARCH', 'FINDALL'])

# Operations used in Condition verification
# Basic scheme: operationXXX(args),
# where args - list/array of arguments (>=1)

def operationAND(args):
    # any number if arguments
    for arg in args:
        if not arg:
           return False
    return True

def operationOR(args):
    # any number of arguments
    for arg in args:
        if arg:
           return True
    return False

def operationNOT(arg):
    # one argument
    return not arg

def operationRANGE(x, x_min, x_max):
    return x_min <= x <= x_max
    
def operationSUBSET(arg1, arg2):
    # True if arg1 is subset of arg2
    # arg1 is an element
    # arg2 is a set
    return arg1 in arg2

def operationLESS(args):
    # any number of args
    for i in range(1, len(args)):
        if args[i-1] >= args[i]:
           return False
    return True

def operationMORE(args):
    # any number of args
    for i in range(1, len(args)):
        if args[i-1] <= args[i]:
           return False
    return True

def operationLESSOREQUAL(args):
    # any number of args
    for i in range(1, len(args)):
        if args[i-1] > args[i]:
           return False
    return True

def operationMOREOREQUAL(args):
    # any number of args
    for i in range(1, len(args)):
        if args[i-1] < args[i]:
           return False
    return True

def operationEQUAL(args):
    # any number of args
    for i in range(1, len(args)):
        if args[i] != args[i-1]:
           return False
    return True

def operationNOTEQUAL(arg1, arg2):
    return arg1 != arg2
    
def operationSUM(args):
    # any numbers of arguments
    if type(args[0]) in set([int, float]):
       result = 0
    elif type(args[0]) in set([str, unicode]):
       result = ''
    else:
       raise Exception('SUM error: unknown arg type')
    for arg in args:
        result += arg
    return result

def operationDIFF(arg1, arg2):
    return arg1-arg2

def operationMUL(args):
    # any numbers of arguments
    if type(args[0]) in set([int, float]):
       result = 1
    else:
       raise Exception('MUL error: unknown arg type')
    for arg in args:
        result *= arg
    return result

def operationDIV(arg1, arg2):
    return arg1/arg2

def operationSTR(arg):
    # transform arg to str
    if type(arg)!=str:
       raise Exception('Type mismatch: STR')
    return arg

def operationSET(arg):
    # transform arg to list
    if type(arg) not in set([list, tuple, set]):
        raise Exception('Type mismatch: SET')
    return list(arg)

def operationMATCH(arg1, arg2):
    # Match regex (arg1) and string (arg2)
    #return bool(re.match(arg1, arg2)) # works wrong
    return bool(re.search(arg1, arg2))

def operationSEARCH(arg1, arg2):
    # Search regex (arg1) in string (arg2)
    # Output list of entries
    group = re.search(arg1, arg2).groups()
    result = []
    for item in group:
        result.append(('STR', item))
    return result

def operationFINDALL(arg1, arg2):
    # Search all groups of a regex
    # Output a list of groups of entries
    # XXX: If a group has more than 1 entry,
    #    there could be potential problems
    list_of_groups = re.findall(arg1, arg2)
    result = []
    for item in list_of_groups:
        result.append(('STR', item))
    return result

def operationLIST(args):
    # args is a list: do nothing (almost)
    return list(args)

# /operations

# GROUPING ---------------------------------------------- 

GROUP_INDEX = {}
# GROUP_INDEX has the following structure:
#  GROUP_INDEX[KEY] = VALUE
#    KEY = table line values
#    VALUE = {'FUNCTIONS':DICT, 'FLAG':LOGICAL, 'ROWID':INTEGER}
#      FUNCTIONS = {'FUNC_NAME':DICT}
#            FUNC_NAME = {'FLAG':LOGICAL, 'NAME':STRING}

# name and default value
GROUP_FUNCTION_NAMES = { 'COUNT' :  0,
                         'SUM'   :  0,
                         'MUL'   :  1,
                         'AVG'   :  0,
                         'MIN'   : +1e100,
                         'MAX'   : -1e100,
                         'SSQ'   : 0,
                       }

def clearGroupIndex():
    #GROUP_INDEX = {}
    for key in GROUP_INDEX.keys():
        del GROUP_INDEX[key]

def getValueFromGroupIndex(GroupIndexKey, FunctionName):
    # If no such index_key, create it and return a value
    if FunctionName not in GROUP_FUNCTION_NAMES:
       raise Exception('No such function \"%s\"' % FunctionName)
    # In the case if NewRowObjectDefault is requested
    if not GroupIndexKey:
       return GROUP_FUNCTION_NAMES[FunctionName]
    if FunctionName not in GROUP_INDEX[GroupIndexKey]['FUNCTIONS']:
       GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName] = {}
       GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['FLAG'] = True
       GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['VALUE'] = \
         GROUP_FUNCTION_NAMES[FunctionName]
    return GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['VALUE']

def setValueToGroupIndex(GroupIndexKey, FunctionName, Value):
    GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['VALUE'] = Value

GROUP_DESC = {}
def initializeGroup(GroupIndexKey):
    if GroupIndexKey not in GROUP_INDEX:
        print('GROUP_DESC[COUNT]='+str(GROUP_DESC['COUNT']))
        GROUP_INDEX[GroupIndexKey] = {}
        GROUP_INDEX[GroupIndexKey]['FUNCTIONS'] = {}
        GROUP_INDEX[GroupIndexKey]['ROWID'] = len(GROUP_INDEX) - 1
    for FunctionName in GROUP_FUNCTION_NAMES:
        # initialize function flags (UpdateFlag)
        if FunctionName in GROUP_INDEX[GroupIndexKey]['FUNCTIONS']:
           GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['FLAG'] = True
    print('initializeGroup: GROUP_INDEX='+str(GROUP_INDEX))

def groupCOUNT(GroupIndexKey):
    FunctionName = 'COUNT'
    Value = getValueFromGroupIndex(GroupIndexKey, FunctionName)
    if GroupIndexKey:
       if GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['FLAG']:
          GROUP_INDEX[GroupIndexKey]['FUNCTIONS'][FunctionName]['FLAG'] = False
          Value = Value + 1
          setValueToGroupIndex(GroupIndexKey, FunctionName, Value)
    return Value

def groupSUM():
    pass

def groupMUL():
    pass

def groupAVG():
    pass

def groupMIN():
    pass

def groupMAX():
    pass

def groupSSQ():
    pass

OPERATORS = {\
# List
'LIST' : lambda args : operationLIST(args),
# And
'&' : lambda args : operationAND(args),
'&&' : lambda args : operationAND(args),
'AND' : lambda args : operationAND(args),
# Or
'|' : lambda args : operationOR(args),
'||' : lambda args : operationOR(args),
'OR' : lambda args : operationOR(args),
# Not
'!' : lambda args : operationNOT(args[0]),
'NOT' : lambda args : operationNOT(args[0]),
# Between
'RANGE' : lambda args : operationRANGE(args[0], args[1], args[2]),
'BETWEEN' : lambda args : operationRANGE(args[0], args[1], args[2]),
# Subset
'IN' : lambda args : operationSUBSET(args[0], args[1]),
'SUBSET': lambda args : operationSUBSET(args[0], args[1]),
# Less
'<' : lambda args : operationLESS(args),
'LESS' : lambda args : operationLESS(args),
'LT'  : lambda args : operationLESS(args),
# More
'>' : lambda args : operationMORE(args),
'MORE' : lambda args : operationMORE(args),
'MT'   : lambda args : operationMORE(args),
# Less or equal
'<=' : lambda args : operationLESSOREQUAL(args),
'LESSOREQUAL' : lambda args : operationLESSOREQUAL(args),
'LTE' : lambda args : operationLESSOREQUAL(args),
# More or equal
'>=' : lambda args : operationMOREOREQUAL(args),
'MOREOREQUAL' : lambda args : operationMOREOREQUAL(args),
'MTE' : lambda args : operationMOREOREQUAL(args),
# Equal
'=' : lambda args : operationEQUAL(args),
'==' : lambda args : operationEQUAL(args),
'EQ' : lambda args : operationEQUAL(args),
'EQUAL' : lambda args : operationEQUAL(args),
'EQUALS' : lambda args : operationEQUAL(args),
# Not equal
'!=' : lambda args : operationNOTEQUAL(args[0], args[1]),
'<>' : lambda args : operationNOTEQUAL(args[0], args[1]),
'~=' : lambda args : operationNOTEQUAL(args[0], args[1]),
'NE' : lambda args : operationNOTEQUAL(args[0], args[1]),
'NOTEQUAL' : lambda args : operationNOTEQUAL(args[0], args[1]),
# Plus
'+' : lambda args : operationSUM(args),
'SUM' : lambda args : operationSUM(args),
# Minus
'-' : lambda args : operationDIFF(args[0], args[1]),
'DIFF' : lambda args : operationDIFF(args[0], args[1]),
# Mul
'*' : lambda args : operationMUL(args),
'MUL' : lambda args : operationMUL(args),
# Div
'/' : lambda args : operationDIV(args[0], args[1]),
'DIV' : lambda args : operationDIV(args[0], args[1]),
# Regexp match
'MATCH' : lambda args : operationMATCH(args[0], args[1]),
'LIKE' : lambda args : operationMATCH(args[0], args[1]),
# Regexp search
'SEARCH' : lambda args : operationSEARCH(args[0], args[1]),
# Regexp findal
'FINDALL' : lambda args : operationFINDALL(args[0], args[1]),
# Group count
'COUNT' : lambda args : groupCOUNT(args[0]),
}
    
# new evaluateExpression function,
#  accounting for groups
"""
def evaluateExpression(root, VarDictionary, GroupIndexKey=None):
    # input = local tree root
    # XXX: this could be very slow due to passing
    #      every time VarDictionary as a parameter
    # Two special cases: 1) root=varname
    #                    2) root=list/tuple
    # These cases must be processed in a separate way
    if type(root) in set([list, tuple]):
       # root is not a leaf
       head = root[0].upper()
       # string constants are treated specially
       if head in set(['STR', 'STRING']): # one arg
          return operationSTR(root[1])
       elif head in set(['SET']):
          return operationSET(root[1])
       tail = root[1:]
       args = []
       # evaluate arguments recursively
       for element in tail: # resolve tree by recursion
           args.append(evaluateExpression(element, VarDictionary, GroupIndexKey))
       # call functions with evaluated arguments
       if head in set(['LIST']): # list arg
          return operationLIST(args)
       elif head in set(['&', '&&', 'AND']): # many args
          return operationAND(args)
       elif head in set(['|', '||', 'OR']): # many args
          return operationOR(args)
       elif head in set(['!', 'NOT']): # one args
          return operationNOT(args[0])
       elif head in set(['RANGE', 'BETWEEN']): # three args
          return operationRANGE(args[0], args[1], args[2])
       elif head in set(['IN', 'SUBSET']): # two args
          return operationSUBSET(args[0], args[1])
       elif head in set(['<', 'LESS', 'LT']): # many args
          return operationLESS(args)
       elif head in set(['>', 'MORE', 'MT']): # many args
          return operationMORE(args)
       elif head in set(['<=', 'LESSOREQUAL', 'LTE']): # many args
          return operationLESSOREQUAL(args)
       elif head in set(['>=', 'MOREOREQUAL', 'MTE']): # many args
          return operationMOREOREQUAL(args)
       elif head in set(['=', '==', 'EQ', 'EQUAL', 'EQUALS']): # many args
          return operationEQUAL(args)
       elif head in set(['!=', '<>', '~=', 'NE', 'NOTEQUAL']): # two args
          return operationNOTEQUAL(args[0], args[1])
       elif head in set(['+', 'SUM']): # many args
          return operationSUM(args)
       elif head in set(['-', 'DIFF']): # two args
          return operationDIFF(args[0], args[1])
       elif head in set(['*', 'MUL']): # many args
          return operationMUL(args)
       elif head in set(['/', 'DIV']): # two args
          return operationDIV(args[0], args[1])
       elif head in set(['MATCH', 'LIKE']): # two args
          return operationMATCH(args[0], args[1])
       elif head in set(['SEARCH']): # two args
          return operationSEARCH(args[0], args[1])
       elif head in set(['FINDALL']): # two args
          return operationFINDALL(args[0], args[1])
       # --- GROUPING OPERATIONS ---
       elif head in set(['COUNT']):
          return groupCOUNT(GroupIndexKey)
       else:
          raise Exception('Unknown operator: %s' % root[0])
    elif type(root) == str:
       # root is a par_name
       return VarDictionary[root]
    else: 
       # root is a non-string constant
       return root
"""

def evaluateExpression(root, VarDictionary, GroupIndexKey=None):
    # input = local tree root
    # XXX: this could be very slow due to passing
    #      every time VarDictionary as a parameter
    # Two special cases: 1) root=varname
    #                    2) root=list/tuple
    # These cases must be processed in a separate way
    if type(root) in set([list, tuple]):
        # root is not a leaf
        head = root[0].upper()
        # string constants are treated specially
        if head in set(['STR', 'STRING']): # one arg
            return operationSTR(root[1])
        elif head in set(['SET']):
            return operationSET(root[1])
        tail = root[1:]
        args = []
        # evaluate arguments recursively
        for element in tail: # resolve tree by recursion
            args.append(evaluateExpression(element, VarDictionary, GroupIndexKey))
        # call functions with evaluated arguments
        try:
            return OPERATORS[head](args)
        except KeyError:
            raise Exception('Unknown operator: %s' % head)
    elif type(root) == str:
       # root is a par_name
       return VarDictionary[root]
    else: 
       # root is a non-string constant
       return root

def getVarDictionary(RowObject):
    # get VarDict from RowObject
    # VarDict: par_name => par_value
    VarDictionary = {}
    for par_name, par_value, par_format in RowObject:
        VarDictionary[par_name] = par_value
    return VarDictionary

def checkRowObject(RowObject, Conditions, VarDictionary):
    #VarDictionary = getVarDictionary(RowObject)   
    if Conditions:
       Flag = evaluateExpression(Conditions, VarDictionary)
    else:
       Flag=True
    return Flag

# ----------------------------------------------------
# /CONDITIONS
# ----------------------------------------------------


# ----------------------------------------------------
# PARAMETER NAMES (includeing creation of new ones)
# ----------------------------------------------------

# Bind an expression to a new parameter
#   in a form: ('BIND', 'new_par', ('some_exp', ...))
def operationBIND(parname, Expression, VarDictionary):
    pass

# This section is for more detailed processing of parlists. 

# Table creation must include not only subsets of 
#   existing parameters, but also new parameters
#   derived from functions on a special prefix language
# For this reason subsetOfRowObject(..) must be substituted
#   by newRowObject(ParameterNames, RowObject)

# For parsing use the function evaluateExpression

# Get names from expression.
#  Must merge this one with evaluateExrpression.
# This is VERY LIMITED version of what will be 
#  when make the language parser is implemented.
# For more ideas and info see LANGUAGE_REFERENCE

# more advansed version of expression evaluator
def evaluateExpressionPAR(ParameterNames, VarDictionary=None):
    # RETURN: 1) Upper-level Expression names
    #         2) Upper-level Expression values
    # Is it reasonable to pass a Context to every parse function?
    # For now the function does the following:
    #   1) iterates through all UPPER-LEVEL list elements
    #   2) if element is a par name: return par name
    #      if element is an BIND expression: return bind name
    #              (see operationBIND)
    #   3) if element is an anonymous expression: return #N(=1, 2, 3...)
    # N.B. Binds can be only on the 0-th level of Expression    
    pass

def getContextFormat(RowObject):
    # Get context format from the whole RowObject
    ContextFormat = {}
    for par_name, par_value, par_format in RowObject:
        ContextFormat[par_name] = par_format
    return ContextFormat

def getDefaultFormat(Type):
    if Type is int:
       return '%10d'
    elif Type is float:
       return '%25.15E'
    elif Type is str:
       return '%20s'
    elif Type is bool:
       return '%2d'
    else:
       raise Exception('Unknown type')
     
def getDefaultValue(Type):
    if Type is int:
       return 0
    elif Type is float:
       return 0.0
    elif Type is str:
       return ''
    elif Type is bool:
       return False
    else:
       raise Exception('Unknown type')

# VarDictionary = Context (this name is more suitable)

# GroupIndexKey is a key to special structure/dictionary GROUP_INDEX.
# GROUP_INDEX contains information needed to calculate streamed group functions
#  such as COUNT, AVG, MIN, MAX etc...

def newRowObject(ParameterNames, RowObject, VarDictionary, ContextFormat, GroupIndexKey=None):
    # Return a subset of RowObject according to 
    # ParameterNames include either par names
    #  or expressions containing par names literals
    # ContextFormat contains format for ParNames
    anoncount = 0
    RowObjectNew = []
    for expr in ParameterNames:
        if type(expr) in set([list, tuple]): # bind
           head = expr[0]
           if head in set(['let', 'bind', 'LET', 'BIND']):
              par_name = expr[1]
              par_expr = expr[2]
           else:
              par_name = "#%d" % anoncount
              anoncount += 1
              par_expr = expr
           par_value = evaluateExpression(par_expr, VarDictionary, GroupIndexKey)
           try:
              par_format = expr[3]
           except:
              par_format = getDefaultFormat(type(par_value))
        else: # parname
           par_name = expr
           par_value = VarDictionary[par_name]
           par_format = ContextFormat[par_name]
        RowObjectNew.append((par_name, par_value, par_format))
    return RowObjectNew

# ----------------------------------------------------
# /PARAMETER NAMES
# ----------------------------------------------------


# ----------------------------------------------------
# OPERATIONS ON TABLES
# ----------------------------------------------------

QUERY_BUFFER = '__BUFFER__'

def getTableList():
    return LOCAL_TABLE_CACHE.keys()

def describeTable(TableName):
    """
    INPUT PARAMETERS: 
        TableName: name of the table to describe
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Print information about table, including 
        parameter names, formats and wavenumber range.
    ---
    EXAMPLE OF USAGE:
        describeTable('sampletab')
    ---
    """
    print('-----------------------------------------')
    print(TableName+' summary:')
    try:
       print('-----------------------------------------')
       print('Comment: \n'+LOCAL_TABLE_CACHE[TableName]['header']['comment'])
    except:
       pass
    print('Number of rows: '+str(LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']))
    print('Table type: '+str(LOCAL_TABLE_CACHE[TableName]['header']['table_type']))
    print('-----------------------------------------')
    print('            PAR_NAME           PAR_FORMAT')
    print('')
    for par_name in LOCAL_TABLE_CACHE[TableName]['header']['order']:
        par_format = LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]
        print('%20s %20s' % (par_name, par_format))
    print('-----------------------------------------')

# Write a table to File or STDOUT
def outputTable(TableName, Conditions=None, File=None, Header=True):
    # Display or record table with condition checking
    if File:
       Header = False
       OutputFile = open(File, 'w')
    if Header:
       headstr = putTableHeaderToString(TableName)
       if File:
          OutputFile.write(headstr)
       else:
          print(headstr)
    for RowID in range(0, LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']):
        RowObject = getRowObject(RowID, TableName)
        VarDictionary = getVarDictionary(RowObject)
        VarDictionary['LineNumber'] = RowID
        if not checkRowObject(RowObject, Conditions, VarDictionary):
           continue
        raw_string = putRowObjectToString(RowObject)
        if File:
           OutputFile.write(raw_string+'\n')
        else:
           print(raw_string)

# Create table "prototype-based" way
def createTable(TableName, RowObjectDefault):
    # create a Table based on a RowObjectDefault
    LOCAL_TABLE_CACHE[TableName] = {}
    header_order = []
    header_format = {}
    header_default = {}
    data = {}
    for par_name, par_value, par_format in RowObjectDefault:
        header_order.append(par_name)
        header_format[par_name] = par_format
        header_default[par_name] = par_value
        data[par_name] = []
    #header_order = tuple(header_order) # XXX ?
    LOCAL_TABLE_CACHE[TableName]['header']={}
    LOCAL_TABLE_CACHE[TableName]['header']['order'] = header_order 
    LOCAL_TABLE_CACHE[TableName]['header']['format'] = header_format
    LOCAL_TABLE_CACHE[TableName]['header']['default'] = header_default
    LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows'] = 0
    LOCAL_TABLE_CACHE[TableName]['header']['size_in_bytes'] = 0
    LOCAL_TABLE_CACHE[TableName]['header']['table_name'] = TableName
    LOCAL_TABLE_CACHE[TableName]['header']['table_type'] = 'column-fixed'
    LOCAL_TABLE_CACHE[TableName]['data'] = data
    

# simple "drop table" capability
def dropTable(TableName):
    """
    INPUT PARAMETERS: 
        TableName:  name of the table to delete
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Deletes a table from local database.
    ---
    EXAMPLE OF USAGE:
        dropTable('some_dummy_table')
    ---
    """
    # delete Table from both Cache and Storage
    try:
       #LOCAL_TABLE_CACHE[TableName] = {}
       del LOCAL_TABLE_CACHE[TableName]
    except:
       pass
    # delete from storage
    pass # TODO

# Returns a column corresponding to parameter name
def getColumn(TableName, ParameterName):
    """
    INPUT PARAMETERS: 
        TableName:      source table name     (required)
        ParameterName:  name of column to get (required)
    OUTPUT PARAMETERS: 
        ColumnData:     list of values from specified column 
    ---
    DESCRIPTION:
        Returns a column with a name ParameterName from
        table TableName. Column is returned as a list of values.
    ---
    EXAMPLE OF USAGE:
        p1 = getColumn('sampletab', 'p1')
    ---
    """
    return LOCAL_TABLE_CACHE[TableName]['data'][ParameterName]

# Returns a list of columns corresponding to parameter names
def getColumns(TableName, ParameterNames):
    """
    INPUT PARAMETERS: 
        TableName:       source table name           (required)
        ParameterNames:  list of column names to get (required)
    OUTPUT PARAMETERS: 
        ListColumnData:   tuple of lists of values from specified column 
    ---
    DESCRIPTION:
        Returns columns with a names in ParameterNames from
        table TableName. Columns are returned as a tuple of lists.
    ---
    EXAMPLE OF USAGE:
        p1, p2, p3 = getColumns('sampletab', ('p1', 'p2', 'p3'))
    ---
    """
    Columns = []
    for par_name in ParameterNames:
        Columns.append(LOCAL_TABLE_CACHE[TableName]['data'][par_name])
    return Columns

def addColumn(TableName, ParameterName, Before=None, Expression=None, Type=None, Default=None, Format=None):
    if ParameterName in LOCAL_TABLE_CACHE[TableName]['header']['format']:
       raise Exception('Column \"%s\" already exists' % ParameterName)
    if not Type: Type = float
    if not Default: Default = getDefaultValue(Type)
    if not Format: Format = getDefaultFormat(Type)
    number_of_rows = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    # Mess with data
    if not Expression:
       LOCAL_TABLE_CACHE[TableName]['data'][ParameterName]=[Default for i in range(0, number_of_rows)]
    else:
       data = []
       for RowID in range(0, number_of_rows):
           RowObject = getRowObject(RowID, TableName)
           VarDictionary = getVarDictionary(RowObject)
           VarDictionary['LineNumber'] = RowID
           par_value = evaluateExpression(Expression, VarDictionary)
           data.append(par_value)
           LOCAL_TABLE_CACHE[TableName]['data'][ParameterName] = data
    # Mess with header
    header_order = LOCAL_TABLE_CACHE[TableName]['header']['order']
    if not Before: 
       header_order.append(ParameterName)
    else:
       #i = 0
       #for par_name in header_order:
       #    if par_name == Before: break
       #    i += 1
       i = header_order.index(Before)
       header_order = header_order[:i] + [ParameterName, ] + header_order[i:]
    LOCAL_TABLE_CACHE[TableName]['header']['order'] = header_order
    LOCAL_TABLE_CACHE[TableName]['header']['format'][ParameterName] = Format
    LOCAL_TABLE_CACHE[TableName]['header']['default'][ParameterName] = Default
   

def deleteColumn(TableName, ParameterName):
    if ParameterName not in LOCAL_TABLE_CACHE[TableName]['header']['format']:
       raise Exception('No such column \"%s\"' % ParameterName)
    # Mess with data
    i = LOCAL_TABLE_CACHE[TableName]['header']['order'].index(ParameterName)
    del LOCAL_TABLE_CACHE[TableName]['header']['order'][i]
    del LOCAL_TABLE_CACHE[TableName]['header']['format'][ParameterName]
    del LOCAL_TABLE_CACHE[TableName]['header']['default'][ParameterName]
    if not LOCAL_TABLE_CACHE[TableName]['header']['order']:
       LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows'] = 0
    # Mess with header
    del LOCAL_TABLE_CACHE[TableName]['data'][ParameterName]

def deleteColumns(TableName, ParameterNames):
    if type(ParameterNames) not in set([list, tuple, set]):
       ParameterNames = [ParameterNames]
    for ParameterName in ParameterNames:
        deleteColumn(TableName, ParameterName)

def renameColumn(TableName, OldParameterName, NewParameterName):
    pass

def insertRow():
    pass

def deleteRows(TableName, ParameterNames, Conditions):
    pass

# select from table to another table
def selectInto(DestinationTableName, TableName, ParameterNames, Conditions):
    # TableName must refer to an existing table in cache!!
    # Conditions = Restrictables in specific format
    # Sample conditions: cond = {'par1':{'range', [b_lo, b_hi]}, 'par2':b}
    # return structure similar to TableObject and put it to QUERY_BUFFER
    # if ParameterNames is '*' then all parameters are used
    #table_columns = LOCAL_TABLE_CACHE[TableName]['data'].keys()
    #table_length = len(TableObject['header']['number_of_rows'])
    #if ParameterNames == '*':
    #   ParameterNames = table_columns
    # check if Conditions contain elements which are not in the TableObject
    #condition_variables = getConditionVariables(Conditions)
    #strange_pars = set(condition_variables)-set(table_variables)
    #if strange_pars: 
    #   raise Exception('The following parameters are not in the table \"%s\"' % (TableName, list(strange_pars)))
    # do full scan each time
    if DestinationTableName == TableName:
       raise Exception('Selecting into source table is forbidden')
    table_length = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    row_count = 0
    for RowID in range(0, table_length):
        RowObject = getRowObject(RowID, TableName)
        VarDictionary = getVarDictionary(RowObject)
        VarDictionary['LineNumber'] = RowID
        ContextFormat = getContextFormat(RowObject)
        RowObjectNew = newRowObject(ParameterNames, RowObject, VarDictionary, ContextFormat)
        if checkRowObject(RowObject, Conditions, VarDictionary):
           addRowObject(RowObjectNew, DestinationTableName)
           row_count += 1
    LOCAL_TABLE_CACHE[DestinationTableName]['header']['number_of_rows'] += row_count

def length(TableName):
    tab_len = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    #print(str(tab_len)+' rows in '+TableName)
    return tab_len

# Select parameters from a table with certain conditions.
# Parameters can be the names or expressions.
# Conditions contain a list of expressions in a special language.
# Set Output to False to suppress output
# Set File=FileName to redirect output to a file.
def select(TableName, DestinationTableName=QUERY_BUFFER, ParameterNames=None, Conditions=None, Output=True, File=None):
    """
    INPUT PARAMETERS: 
        TableName:            name of source table              (required)
        DestinationTableName: name of resulting table           (optional)
        ParameterNames:       list of parameters or expressions (optional)
        Conditions:           list of logincal expressions      (optional)
        Output:   enable (True) or suppress (False) text output (optional)
        File:     enable (True) or suppress (False) file output (optional)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Select or filter the data in some table 
        either to standard output or to file (if specified)
    ---
    EXAMPLE OF USAGE:
        select('sampletab', DestinationTableName='outtab', ParameterNames=(p1, p2),
                Conditions=(('and', ('>=', 'p1', 1), ('<', ('*', 'p1', 'p2'), 20))))
        Conditions means (p1>=1 and p1*p2<20)
    ---
    """
    # TODO: Variables defined in ParameterNames ('LET') MUST BE VISIBLE IN Conditions !!
    # check if table exists
    if TableName not in LOCAL_TABLE_CACHE.keys():
        raise Exception('%s: no such table. Check tableList() for more info.' % TableName)
    if not ParameterNames: ParameterNames=LOCAL_TABLE_CACHE[TableName]['header']['order']
    LOCAL_TABLE_CACHE[DestinationTableName] = {} # clear QUERY_BUFFER for the new result
    RowObjectDefault = getDefaultRowObject(TableName)
    VarDictionary = getVarDictionary(RowObjectDefault)
    ContextFormat = getContextFormat(RowObjectDefault)
    RowObjectDefaultNew = newRowObject(ParameterNames, RowObjectDefault, VarDictionary, ContextFormat)
    dropTable(DestinationTableName) # redundant
    createTable(DestinationTableName, RowObjectDefaultNew)
    selectInto(DestinationTableName, TableName, ParameterNames, Conditions)
    if DestinationTableName!=QUERY_BUFFER:
        if File: outputTable(DestinationTableName, File=File)
    elif Output:
        outputTable(DestinationTableName, File=File)

# SORTING ===========================================================

def arrangeTable(TableName, DestinationTableName=None, RowIDList=None):
    #print 'AT/'
    #print 'AT: RowIDList = '+str(RowIDList)
    # make a subset of table rows according to RowIDList
    if not DestinationTableName:
       DestinationTableName = TableName
    if DestinationTableName != TableName:
       dropTable(DestinationTableName)
       LOCAL_TABLE_CACHE[DestinationTableName]['header']=LOCAL_TABLE_CACHE[TableName]['header']
       LOCAL_TABLE_CACHE[DestinationTableName]['data']={}
    LOCAL_TABLE_CACHE[DestinationTableName]['header']['number_of_rows'] = len(RowIDList)
    #print 'AT: RowIDList = '+str(RowIDList)
    for par_name in LOCAL_TABLE_CACHE[DestinationTableName]['header']['order']:
        par_data = LOCAL_TABLE_CACHE[TableName]['data'][par_name]
        LOCAL_TABLE_CACHE[DestinationTableName]['data'][par_name] = [par_data[i] for i in RowIDList]
    
def compareLESS(RowObject1, RowObject2, ParameterNames):
    #print 'CL/'
    # arg1 and arg2 are RowObjects
    # Compare them according to ParameterNames
    # Simple validity check:
    #if len(arg1) != len(arg2):
    #   raise Exception('Arguments have different lengths')
    #RowObject1Subset = subsetOfRowObject(ParameterNames, RowObject1)
    #RowObject2Subset = subsetOfRowObject(ParameterNames, RowObject2)
    #return RowObject1Subset < RowObject2Subset
    row1 = []
    row2 = []
    #n = len(RowObject1)
    #for i in range(0, n):
    #    par_name1 = RowObject1[i][0]
    #    if par_name1 in ParameterNames:
    #       par_value1 = RowObject1[i][1]
    #       par_value2 = RowObject2[i][1]
    #       row1 += [par_value1]
    #       row2 += [par_value2]
    VarDictionary1 = getVarDictionary(RowObject1)
    VarDictionary2 = getVarDictionary(RowObject2)
    for par_name in ParameterNames:
        par_value1 = VarDictionary1[par_name]
        par_value2 = VarDictionary2[par_name]
        row1 += [par_value1]
        row2 += [par_value2]
    Flag = row1 < row2
    return Flag

def quickSort(index, TableName, ParameterNames, Accending=True):
    # ParameterNames: names of parameters which are
    #  taking part in the sorting
    if index == []:
       return []
    else:
       PivotID = index[0]
       Pivot = getRowObject(PivotID, TableName)
       lesser_index = []
       greater_index = [];
       for RowID in index[1:]:
           RowObject = getRowObject(RowID, TableName)
           if compareLESS(RowObject, Pivot, ParameterNames):
              lesser_index += [RowID]
           else:
              greater_index += [RowID]
       lesser = quickSort(lesser_index, TableName, ParameterNames, Accending)
       greater = quickSort(greater_index, TableName, ParameterNames, Accending)
       if Accending:
          return lesser + [PivotID] + greater
       else:
          return greater + [PivotID] + lesser

# Sorting must work well on the table itself!
def sort(TableName, DestinationTableName=None, ParameterNames=None, Accending=True, Output=False, File=None):
    """
    INPUT PARAMETERS: 
        TableName:                name of source table          (required)
        DestinationTableName:     name of resulting table       (optional)
        ParameterNames:       list of parameters or expressions to sort by    (optional)
        Accending:       sort in ascending (True) or descending (False) order (optional)
        Output:   enable (True) or suppress (False) text output (optional)
        File:     enable (True) or suppress (False) file output (optional)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Sort a table by a list of it's parameters or expressions.
        The sorted table is saved in DestinationTableName (if specified).
    ---
    EXAMPLE OF USAGE:
        sort('sampletab', ParameterNames=(p1, ('+', p1, p2)))
    ---
    """
    number_of_rows = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    index = range(0, number_of_rows)
    if not DestinationTableName:
       DestinationTableName = TableName
    # if names are not provided use all parameters in sorting
    if not ParameterNames:
       ParameterNames = LOCAL_TABLE_CACHE[TableName]['header']['order']
    elif type(ParameterNames) not in set([list, tuple]):
       ParameterNames = [ParameterNames] # fix of stupid bug where ('p1', ) != ('p1')
    index_sorted = quickSort(index, TableName, ParameterNames, Accending)
    arrangeTable(TableName, DestinationTableName, index_sorted)
    if Output:
       outputTable(DestinationTableName, File=File)

# /SORTING ==========================================================
    

# GROUPING ==========================================================

# GROUP_INDEX global auxiliary structure is a Dictionary,
#   which has the following properties:
#      1) Each key is a composite variable:
#          [array of values of ParameterNames variable
#           STREAM_UPDATE_FLAG]
#      2) Each value is an index in LOCAL_TABLE_CACHE[TableName]['data'][...],
#          corresponding to this key
#   STREAM_UPDATE_FLAG = TRUE if value in GROUP_INDEX needs updating
#                      = FALSE otherwise
#   If no grouping variables are specified (GroupParameterNames == None)
#    than the following key is used: "__GLOBAL__"


def group(TableName, DestinationTableName=QUERY_BUFFER, ParameterNames=None, GroupParameterNames=None, File=None, Output=True):
    """
    INPUT PARAMETERS: 
        TableName:                name of source table          (required)
        DestinationTableName:     name of resulting table       (optional)
        ParameterNames:       list of parameters or expressions to take       (optional)
        GroupParameterNames:  list of parameters or expressions to group by   (optional)
        Accending:       sort in ascending (True) or descending (False) order (optional)
        Output:   enable (True) or suppress (False) text output (optional)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        none
    ---
    EXAMPLE OF USAGE:
        group('sampletab', ParameterNames=('p1', ('sum', 'p2')), GroupParameterNames=('p1'))
        ... makes grouping by p1, p2. For each group it calculates sum of p2 values.
    ---
    """
    # Implements such functions as:
    # count, sum, avg, min, max, ssq etc...
    # 1) ParameterNames can contain group functions
    # 2) GroupParameterNames can't contain group functions
    # 3) If ParameterNames contains parameters defined by LET directive,
    #    it IS visible in the sub-context of GroupParameterNames
    # 4) Parameters defined in GroupParameterNames are NOT visible in ParameterNames
    # 5) ParameterNames variable represents the structure of the resulting table/collection
    # 6) GroupParameterNames can contain either par_names or expressions with par_names
    # Clear old GROUP_INDEX value
    clearGroupIndex()
    # Consistency check
    if TableName == DestinationTableName:
       raise Exception('TableName and DestinationTableName must be different')
    #if not ParameterNames: ParameterNames=LOCAL_TABLE_CACHE[TableName]['header']['order']
    # Prepare the new DestinationTable
    RowObjectDefault = getDefaultRowObject(TableName)
    VarDictionary = getVarDictionary(RowObjectDefault)
    ContextFormat = getContextFormat(RowObjectDefault)
    RowObjectDefaultNew = newRowObject(ParameterNames, RowObjectDefault, VarDictionary, ContextFormat)
    dropTable(DestinationTableName) # redundant
    createTable(DestinationTableName, RowObjectDefaultNew)
    # Loop through rows of source Table
    # On each iteration group functions update GROUP_INDEX (see description above)
    number_of_rows = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']   
    # STAGE 1: CREATE GROUPS
    print('LOOP:')
    for RowID in range(0, number_of_rows):
        print('--------------------------------')
        print('RowID='+str(RowID))
        RowObject = getRowObject(RowID, TableName) # RowObject from source table
        VarDictionary = getVarDictionary(RowObject)
        print('VarDictionary='+str(VarDictionary))
        # This is a trick which makes evaluateExpression function
        #   not consider first expression as an operation
        GroupParameterNames_ = ['LIST'] + list(GroupParameterNames)
        GroupIndexKey = evaluateExpression(GroupParameterNames_, VarDictionary)
        # List is an unhashable type in Python!
        GroupIndexKey = tuple(GroupIndexKey)       
        initializeGroup(GroupIndexKey)
        print('GROUP_INDEX='+str(GROUP_INDEX))
        ContextFormat = getContextFormat(RowObject)
        RowObjectNew = newRowObject(ParameterNames, RowObject, VarDictionary, ContextFormat, GroupIndexKey)
        RowIDGroup = GROUP_INDEX[GroupIndexKey]['ROWID']
        setRowObject(RowIDGroup, RowObjectNew, DestinationTableName)
    # Output result if required
    if Output and DestinationTableName == QUERY_BUFFER:
       outputTable(DestinationTableName, File=File)

# /GROUPING =========================================================

# EXTRACTING ========================================================

REGEX_INTEGER = '[+-]?\d+'
REGEX_STRING = '[^\s]+'
REGEX_FLOAT_F = '[+-]?\d*\.?\d+'
REGEX_FLOAT_E = '[+-]?\d*\.?\d+[eEfF]?[+-]?\d+' 

REGEX_INTEGER_FIXCOL = lambda n: '\d{%d}' % n
REGEX_STRING_FIXCOL = lambda n: '[^\s]{%d}' % n
REGEX_FLOAT_F_FIXCOL = lambda n: '[\+\-\.\d]{%d}' % n
REGEX_FLOAT_E_FIXCOL = lambda n: '[\+\-\.\deEfF]{%d}' % n

# Extract sub-columns from string column
def extractColumns(TableName, SourceParameterName, ParameterFormats, ParameterNames=None, FixCol=False):
    """
    INPUT PARAMETERS: 
        TableName:             name of source table              (required)
        SourceParameterName:   name of source column to process  (required)
        ParameterFormats:      c formats of unpacked parameters  (required)
        ParameterNames:        list of resulting parameter names (optional)
        FixCol:      column-fixed (True) format of source column (optional)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Note, that this function is aimed to do some extra job on
        interpreting string parameters which is normally supposed
        to be done by the user.
    ---
    EXAMPLE OF USAGE:
        extractColumns('sampletab', SourceParameterName='p5',
                        ParameterFormats=('%d', '%d', '%d'),
                        ParameterNames=('p5_1', 'p5_2', 'p5_3'))
        This example extracts three integer parameters from
        a source column 'p5' and puts results in ('p5_1', 'p5_2', 'p5_3').
    ---
    """
    # ParameterNames = just the names without expressions
    # ParFormats contains python formats for par extraction
    # Example: ParameterNames=('v1', 'v2', 'v3')
    #          ParameterFormats=('%1s', '%1s', '%1s')
    # By default the format of parameters is column-fixed
    if type(LOCAL_TABLE_CACHE[TableName]['header']['default'][SourceParameterName]) not in set([str, unicode]):
       raise Exception('Source parameter must be a string')
    i=-1
    # bug when (a, ) != (a)
    if ParameterNames and type(ParameterNames) not in set([list, tuple]):
       ParameterNames = [ParameterNames]
    if ParameterFormats and type(ParameterFormats) not in set([list, tuple]):
       ParameterFormats = [ParameterFormats]
    # if ParameterNames is empty, fill it with #1-2-3-...
    if not ParameterNames:
       ParameterNames = []
       # using naming convension #i, i=0, 1, 2, 3...
       for par_format in ParameterFormats:
           while True:
                 i+=1
                 par_name = '#%d' % i
                 fmt = LOCAL_TABLE_CACHE[TableName]['header']['format'].get(par_name, None)
                 if not fmt: break
           ParameterNames.append(par_name)
    # check if ParameterNames are valid
    Intersection = set(ParameterNames).intersection(LOCAL_TABLE_CACHE[TableName]['header']['order'])
    if Intersection:
       raise Exception('Parameters %s already exist' % str(list(Intersection)))
    # loop over ParameterNames to prepare LOCAL_TABLE_CACHE
    i=0
    for par_name in ParameterNames:  
        par_format = ParameterFormats[i]     
        LOCAL_TABLE_CACHE[TableName]['header']['format'][par_name]=par_format
        LOCAL_TABLE_CACHE[TableName]['data'][par_name]=[] 
        i+=1
    # append new parameters in order list
    LOCAL_TABLE_CACHE[TableName]['header']['order'] += ParameterNames
    # cope with default values
    i=0
    format_regex = []
    format_types = []
    for par_format in ParameterFormats:
        par_name = ParameterNames[i]
        regex = FORMAT_PYTHON_REGEX
        (lng, trail, lngpnt, ty) = re.search(regex, par_format).groups()
        ty = ty.lower()
        if ty == 'd':
           par_type = int
           if FixCol:
              format_regex_part = REGEX_INTEGER_FIXCOL(lng)
           else:
              format_regex_part = REGEX_INTEGER
        elif ty == 's':
           par_type = str
           if FixCol:
              format_regex_part = REGEX_STRING_FIXCOL(lng)
           else:
              format_regex_part = REGEX_STRING
        elif ty == 'f':
           par_type = float
           if FixCol:
              format_regex_part = REGEX_FLOAT_F_FIXCOL(lng)
           else:
              format_regex_part = REGEX_FLOAT_F
        elif ty == 'e':
           par_type = float
           if FixCol:
              format_regex_part = REGEX_FLOAT_E_FIXCOL(lng)
           else:
              format_regex_part = REGEX_FLOAT_E
        else:
           raise Exception('Unknown data type')
        format_regex.append('('+format_regex_part+')')
        format_types.append(par_type)
        def_val = getDefaultValue(par_type)
        LOCAL_TABLE_CACHE[TableName]['header']['default'][par_name]=def_val
        i+=1
    format_regex = '\s*'.join(format_regex)
    # loop through values of SourceParameter
    for SourceParameterString in LOCAL_TABLE_CACHE[TableName]['data'][SourceParameterName]:
        try:
           ExtractedValues = list(re.search(format_regex, SourceParameterString).groups())
        except:
           raise Exception('Error with line \"%s\"' % SourceParameterString)
        i=0
        # loop through all parameters which are supposed to be extracted
        for par_name in ParameterNames:
            par_value = format_types[i](ExtractedValues[i])
            LOCAL_TABLE_CACHE[TableName]['data'][par_name].append(par_value)
            i+=1
    # explicitly check that number of rows are equal
    number_of_rows = LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
    number_of_rows2 = len(LOCAL_TABLE_CACHE[TableName]['data'][SourceParameterName])
    number_of_rows3 = len(LOCAL_TABLE_CACHE[TableName]['data'][ParameterNames[0]])
    if not (number_of_rows == number_of_rows2 == number_of_rows3):
       raise Exception('Error while extracting parameters: check your regexp')

# Split string columns into sub-columns with given names
def splitColumn(TableName, SourceParameterName, ParameterNames, Splitter):
    pass

# /EXTRACTING =======================================================

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# /LOCAL DATABASE MANAGEMENT SYSTEM
# ---------------------------------------------------------------
# ---------------------------------------------------------------


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# GLOBAL API FUNCTIONS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

def mergeParlist(*arg):
    # Merge parlists and remove duplicates.
    # Argument contains a list of lists/tuples.
    container = []
    for a in arg:
        container += list(a)
    result = []
    index = set()
    for par_name in container:
        if par_name not in index:
            index.add(par_name)
            result.append(par_name)
    return result

# Define parameter groups to simplify the usage of fetch_
# "Long term" core version includes templates for the Parlists instead of listing the broadeners explicitly.
# Each template parameter has '%s' in place of a broadener, i.e. 'gamma_%s' instead of 'gamma_air' 

# ======================================================
# CODE FOR GENERATING PARAMETER NAMES FOR LINE PROFILES
# NAME: GENERATE_PARLIST
# NOTE: THIS CODE DOESN'T COVER NON-PROFILE PARAMETERS 
#       SUCH AS QUANTA, LOCAL AND GLOBAL IDS ETC...
# NOTE2: THIS CODE DOESN'T GUARANTEE THAT THE GENERATED
#        PARAMETER NAMES WILL EXIST IN THE HITRANONLINE 
#        DATABASE!
#        TO GET THE REAL PARAMETER NAMES PLEASE EITHER
#        USE THE EXTENSION OF THE HITRAN APPLICATION
#        PROGRAMMING INTERFACE:
#             (http://github.org/hitranonline/hapi2) ...
#        ... OR LOOK FOR THE CUSTOM USER FORMAT WEB PAGE
#        ON HITRANONLINE WEBSITE (http://hitran.org).
# ======================================================

VOIGT_PROFILE_TEMPLATE = ['gamma_%s', 'n_%s', 'delta_%s', 'deltap_%s']

SDVOIGT_PROFILE_TEMPLATE = [
    'gamma_SDV_0_%s_%d', 'n_SDV_%s_%d', # HWHM AND ITS T-DEPENDENCE
    'gamma_SDV_2_%s_%d', 'n_gamma_SDV_2_%s_%d', # SPEED-DEPENDENCE OF HWHM AND ITS T-DEPENDENCE
    'delta_SDV_%s_%d', 'deltap_SDV_%s_%d', # SHIFT AND ITS T-DEPENDENCE
    'SD_%s' # UNITLESS SDV PARAMETER
    ]

HT_PROFILE_TEMPLATE = [
    'gamma_HT_0_%s_%d', 'n_HT_%s_%d', # HWHM AND ITS T-DEPENDENCE
    'gamma_HT_2_%s_%d', 'n_gamma_HT_2_%s_%d', # SPEED-DEPENDENCE OF HWHM AND ITS T-DEPENDENCE
    'delta_HT_%s_%d', 'deltap_HT_%s_%d', # SHIFT AND ITS T-DEPENDENCE
    ]

def apply_env(template, broadener, Tref):
    args = []
    if '%s' in template:
        args.append(broadener)
    if '%d'  in template:
        args.append(Tref)
    return template%tuple(args)

def generate_parlist(profile, broadener, Tref):
    PROFILE_MAP = {
        'voigt': VOIGT_PROFILE_TEMPLATE,
        'vp': VOIGT_PROFILE_TEMPLATE,
        'sdvoigt': SDVOIGT_PROFILE_TEMPLATE,
        'sdvp': SDVOIGT_PROFILE_TEMPLATE,
        'ht': HT_PROFILE_TEMPLATE,
        'htp': HT_PROFILE_TEMPLATE,
    }
    return [apply_env(template, broadener, Tref) \
        for template in PROFILE_MAP[profile.lower()]] 
    
# generate_parlist('Voigt', 'air', 296)  =>   gamma_air,
    
# ====================================================================        
# PARLISTS FOR EACH BROADENER EXPLICITLY (FOR BACKWARDS COMPATIBILITY)
# ====================================================================        

# Define parameter groups to simplify the usage of fetch_
PARLIST_DOTPAR = ['par_line', ]
PARLIST_ID = ['trans_id', ]
PARLIST_STANDARD = ['molec_id', 'local_iso_id', 'nu', 'sw', 'a', 'elower', 'gamma_air',
                    'delta_air', 'gamma_self', 'n_air', 'n_self', 'gp', 'gpp']
PARLIST_LABELS = ['statep', 'statepp']
#PARLIST_LINEMIXING = ['y_air', 'y_self']

PARLIST_VOIGT_AIR = ['gamma_air', 'delta_air', 'deltap_air', 'n_air']
PARLIST_VOIGT_SELF = ['gamma_self', 'delta_self', 'deltap_self', 'n_self']
PARLIST_VOIGT_H2 = ['gamma_H2', 'delta_H2', 'deltap_H2', 'n_H2']
PARLIST_VOIGT_CO2 = ['gamma_CO2', 'delta_CO2', 'n_CO2']
PARLIST_VOIGT_HE = ['gamma_He', 'delta_He', 'n_He']
PARLIST_VOIGT_H2O = ['gamma_H2O', 'n_H2O']
PARLIST_VOIGT_LINEMIXING_AIR = ['y_air']
PARLIST_VOIGT_LINEMIXING_SELF = ['y_self']
PARLIST_VOIGT_LINEMIXING_ALL = mergeParlist(PARLIST_VOIGT_LINEMIXING_AIR,
                                            PARLIST_VOIGT_LINEMIXING_SELF)
PARLIST_VOIGT_ALL = mergeParlist(PARLIST_VOIGT_AIR, PARLIST_VOIGT_SELF,
                                 PARLIST_VOIGT_H2, PARLIST_VOIGT_CO2,
                                 PARLIST_VOIGT_HE, PARLIST_VOIGT_H2O,
                                 PARLIST_VOIGT_LINEMIXING_ALL)

#PARLIST_SDVOIGT_AIR = ['gamma_air', 'delta_air', 'deltap_air', 'n_air', 'SD_air']
#PARLIST_SDVOIGT_AIR = ['gamma_SDV_0_air_296', 'n_SDV_air_296',
#                       'gamma_SDV_2_air_296', 'n_gamma_SDV_2_air_296', # n_SDV_2_air_296 ?
PARLIST_SDVOIGT_AIR = ['gamma_SDV_0_air_296',  # don't include temperature exponents while they are absent in the database
                       'gamma_SDV_2_air_296',  # don't include temperature exponents while they are absent in the database
                       'delta_SDV_0_air_296', 'deltap_SDV_air_296', 'SD_air']
#PARLIST_SDVOIGT_SELF = ['gamma_self', 'delta_self', 'deltap_self', 'n_self', 'SD_self']
#PARLIST_SDVOIGT_SELF = ['gamma_SDV_0_self_296', 'n_SDV_self_296',
#                       'gamma_SDV_2_self_296', 'n_gamma_SDV_2_self_296', # n_SDV_2_self_296 ?
PARLIST_SDVOIGT_SELF = ['gamma_SDV_0_self_296', # don't include temperature exponents while they are absent in the database
                       'gamma_SDV_2_self_296',  # don't include temperature exponents while they are absent in the database
                       'delta_SDV_0_self_296', 'deltap_SDV_self_296', 'SD_self']
PARLIST_SDVOIGT_H2 = []
PARLIST_SDVOIGT_CO2 = []
PARLIST_SDVOIGT_HE = []
#PARLIST_SDVOIGT_LINEMIXING_AIR = ['Y_SDV_air_296', 'n_Y_SDV_air_296']
PARLIST_SDVOIGT_LINEMIXING_AIR = ['Y_SDV_air_296'] # don't include temperature exponents while they are absent in the database
#PARLIST_SDVOIGT_LINEMIXING_SELF = ['Y_SDV_self_296', 'n_Y_SDV_self_296']
PARLIST_SDVOIGT_LINEMIXING_SELF = ['Y_SDV_self_296'] # don't include temperature exponents while they are absent in the database
PARLIST_SDVOIGT_LINEMIXING_ALL = mergeParlist(PARLIST_SDVOIGT_LINEMIXING_AIR,
                                              PARLIST_SDVOIGT_LINEMIXING_SELF)
PARLIST_SDVOIGT_ALL = mergeParlist(PARLIST_SDVOIGT_AIR, PARLIST_SDVOIGT_SELF,
                                   PARLIST_SDVOIGT_H2, PARLIST_SDVOIGT_CO2,
                                   PARLIST_SDVOIGT_HE, PARLIST_SDVOIGT_LINEMIXING_ALL)

PARLIST_GALATRY_AIR = ['gamma_air', 'delta_air', 'deltap_air', 'n_air', 'beta_g_air']
PARLIST_GALATRY_SELF = ['gamma_self', 'delta_self', 'deltap_self', 'n_self', 'beta_g_self']
PARLIST_GALATRY_H2 = []
PARLIST_GALATRY_CO2 = []
PARLIST_GALATRY_HE = []
PARLIST_GALATRY_ALL = mergeParlist(PARLIST_GALATRY_AIR, PARLIST_GALATRY_SELF,
                                   PARLIST_GALATRY_H2, PARLIST_GALATRY_CO2,
                                   PARLIST_GALATRY_HE)

PARLIST_HT_SELF = ['gamma_HT_0_self_50', 'n_HT_self_50', 'gamma_HT_2_self_50',
                   'delta_HT_0_self_50', 'deltap_HT_self_50', 'delta_HT_2_self_50',
                   'gamma_HT_0_self_150', 'n_HT_self_150', 'gamma_HT_2_self_150',
                   'delta_HT_0_self_150', 'deltap_HT_self_150', 'delta_HT_2_self_150',
                   'gamma_HT_0_self_296', 'n_HT_self_296', 'gamma_HT_2_self_296',
                   'delta_HT_0_self_296', 'deltap_HT_self_296', 'delta_HT_2_self_296',
                   'gamma_HT_0_self_700', 'n_HT_self_700', 'gamma_HT_2_self_700',
                   'delta_HT_0_self_700', 'deltap_HT_self_700', 'delta_HT_2_self_700',
                   'nu_HT_self', 'kappa_HT_self', 'eta_HT_self', 'Y_HT_self_296']
#PARLIST_HT_AIR = ['gamma_HT_0_air_50', 'n_HT_air_50', 'gamma_HT_2_air_50',
#                  'delta_HT_0_air_50', 'deltap_HT_air_50', 'delta_HT_2_air_50',
#                  'gamma_HT_0_air_150', 'n_HT_air_150', 'gamma_HT_2_air_150',
#                  'delta_HT_0_air_150', 'deltap_HT_air_150', 'delta_HT_2_air_150',
#                  'gamma_HT_0_air_296', 'n_HT_air_296', 'gamma_HT_2_air_296',
#                  'delta_HT_0_air_296', 'deltap_HT_air_296', 'delta_HT_2_air_296',
#                  'gamma_HT_0_air_700', 'n_HT_air_700', 'gamma_HT_2_air_700',
#                  'delta_HT_0_air_700', 'deltap_HT_air_700', 'delta_HT_2_air_700',
#                  'nu_HT_air', 'kappa_HT_air', 'eta_HT_air']
PARLIST_HT_AIR = ['gamma_HT_0_air_296', 'n_HT_air_296', 'gamma_HT_2_air_296',
                  'delta_HT_0_air_296', 'deltap_HT_air_296', 'delta_HT_2_air_296',
                  'nu_HT_air', 'kappa_HT_air', 'eta_HT_air', 'Y_HT_air_296']
PARLIST_HT_ALL = mergeParlist(PARLIST_HT_SELF, PARLIST_HT_AIR)
                                   
PARLIST_ALL = mergeParlist(PARLIST_ID, PARLIST_DOTPAR, PARLIST_STANDARD,
                           PARLIST_LABELS, PARLIST_VOIGT_ALL,
                           PARLIST_SDVOIGT_ALL, PARLIST_GALATRY_ALL,
                           PARLIST_HT_ALL)

# ====================================================================        
# PARLISTS FOR EACH BROADENER EXPLICITLY (FOR BACKWARDS COMPATIBILITY)
# ====================================================================        
                           
PARAMETER_GROUPS = {
  'par_line' : PARLIST_DOTPAR,
  '160-char' : PARLIST_DOTPAR,
  '.par' : PARLIST_DOTPAR,
  'id' : PARLIST_ID,
  'standard' : PARLIST_STANDARD,
  'labels' : PARLIST_LABELS,
  #'linemixing' : PARLIST_LINEMIXING,
  'voigt_air' : PARLIST_VOIGT_AIR,
  'voigt_self' : PARLIST_VOIGT_SELF,
  'voigt_h2' : PARLIST_VOIGT_H2,
  'voigt_co2' : PARLIST_VOIGT_CO2,
  'voigt_he' : PARLIST_VOIGT_HE,
  'voigt_h2o' : PARLIST_VOIGT_H2O,
  'voigt_linemixing_air': PARLIST_VOIGT_LINEMIXING_AIR,
  'voigt_linemixing_self': PARLIST_VOIGT_LINEMIXING_SELF,
  'voigt_linemixing': PARLIST_VOIGT_LINEMIXING_ALL,
  'voigt' : PARLIST_VOIGT_ALL,
  'sdvoigt_air' : PARLIST_SDVOIGT_AIR,
  'sdvoigt_self' : PARLIST_SDVOIGT_SELF,
  'sdvoigt_h2' : PARLIST_SDVOIGT_H2,
  'sdvoigt_co2' : PARLIST_SDVOIGT_CO2,
  'sdvoigt_he' : PARLIST_SDVOIGT_HE,
  'sdvoigt_linemixing_air': PARLIST_SDVOIGT_LINEMIXING_AIR,
  'sdvoigt_linemixing_self': PARLIST_SDVOIGT_LINEMIXING_SELF,
  'sdvoigt_linemixing': PARLIST_SDVOIGT_LINEMIXING_ALL,
  'sdvoigt' : PARLIST_SDVOIGT_ALL,
  'galatry_air' : PARLIST_GALATRY_AIR,
  'galatry_self' : PARLIST_GALATRY_SELF,
  'galatry_h2' : PARLIST_GALATRY_H2,
  'galatry_co2' : PARLIST_GALATRY_CO2,
  'galatry_he' : PARLIST_GALATRY_HE,
  'galatry' : PARLIST_GALATRY_ALL,
  'ht' : PARLIST_HT_ALL,
  'all' : PARLIST_ALL
}

def prepareParlist(pargroups=[], params=[], dotpar=True):
    # Apply defaults
    parlist_default = []
    if dotpar:
        parlist_default += ['par_line']
    #parlist_default += PARAMETER_GROUPS['id']
    
    # Make a dictionary of "assumed" parameters.
    ASSUMED_PARAMS = {}
    if 'par_line' in set(parlist_default):
        ASSUMED_PARAMS = HITRAN_DEFAULT_HEADER['format']
    
    parlist = parlist_default
    
    # Iterate over parameter groups.
    for pargroup in pargroups:
        pargroup = pargroup.lower()
        parlist += PARAMETER_GROUPS[pargroup]
        
    # Iterate over single parameters.
    for param in params:
        #param = param.lower()
        parlist.append(param)
        
    # Clean up parameter list.
    parlist = mergeParlist(parlist)
    result = []
    for param in parlist:
        if param not in ASSUMED_PARAMS:
            result.append(param)
    
    return result

def prepareHeader(parlist):
    HEADER = {'table_name':'', 'number_of_rows':-1, 'format':{},
              'default':{}, 'table_type':'column-fixed',
              'size_in_bytes':-1, 'order':[], 'description':{}}
    
    # Add column-fixed 160-character part, if specified in parlist.
    if 'par_line' in set(parlist):
        HEADER['order'] = HITRAN_DEFAULT_HEADER['order']
        HEADER['format'] = HITRAN_DEFAULT_HEADER['format']
        HEADER['default'] = HITRAN_DEFAULT_HEADER['default']
        HEADER['description'] = HITRAN_DEFAULT_HEADER['description']
        HEADER['position'] = HITRAN_DEFAULT_HEADER['position']

    # Insert all other parameters in the "extra" section of the header.
    plist = [v for v in parlist if v!='par_line']
    HEADER['extra'] = []
    HEADER['extra_format'] = {}
    HEADER['extra_separator'] = ', '
    for param in plist:
        param = param.lower()
        HEADER['extra'].append(param)
        HEADER['extra_format'][param] = PARAMETER_META[param]['default_fmt']
        
    return HEADER
        
def queryHITRAN(TableName, iso_id_list, numin, numax, pargroups=[], params=[], dotpar=True, head=False):
    ParameterList = prepareParlist(pargroups=pargroups, params=params, dotpar=dotpar)
    TableHeader = prepareHeader(ParameterList)
    TableHeader['table_name'] = TableName
    DataFileName = VARIABLES['BACKEND_DATABASE_NAME'] + '/' + TableName + '.data'
    HeaderFileName = VARIABLES['BACKEND_DATABASE_NAME'] + '/' + TableName + '.header'
    # create URL
    iso_id_list_str = [str(iso_id) for iso_id in iso_id_list]
    iso_id_list_str = ', '.join(iso_id_list_str)
    print('\nData is fetched from %s\n'%VARIABLES['GLOBAL_HOST'])
    if pargroups or params: # custom par search
        url = VARIABLES['GLOBAL_HOST'] + '/lbl/api?' + \
        'iso_ids_list=' + iso_id_list_str + '&' + \
        'numin=' + str(numin) + '&' + \
        'numax=' + str(numax) + '&' + \
        'head=' + str(head) + '&' + \
        'fixwidth=0&sep=[comma]&' +\
        'request_params=' + ', '.join(ParameterList)
    else: # old-fashioned .par search
        url = VARIABLES['GLOBAL_HOST'] + '/lbl/api?' + \
        'iso_ids_list=' + iso_id_list_str + '&' + \
        'numin=' + str(numin) + '&' + \
        'numax=' + str(numax)
    #raise Exception(url)
    # Download data by chunks.
    if VARIABLES['DISPLAY_FETCH_URL']: print(url+'\n')
    try:       
        # Proxy handling # https://stackoverflow.com/questions/1450132/proxy-with-urllib2
        if VARIABLES['PROXY']:
            print('Using proxy '+str(VARIABLES['PROXY']))
            proxy = urllib2.ProxyHandler(VARIABLES['PROXY'])
            opener = urllib2.build_opener(proxy)
            urllib2.install_opener(opener)            
        req = urllib2.urlopen(url)
    except urllib2.HTTPError:
        raise Exception('Failed to retrieve data for given parameters.')
    except urllib2.URLError:
        raise Exception('Cannot connect to %s. Try again or edit GLOBAL_HOST variable.' % GLOBAL_HOST)
    CHUNK = 64 * 1024
    print('BEGIN DOWNLOAD: '+TableName)
    with open_(DataFileName, 'w') as fp:
       while True:
          chunk = req.read(CHUNK)
          if not chunk: break
          fp.write(chunk.decode('utf-8'))
          print('  %d bytes written to %s' % (CHUNK, DataFileName))
    with open(HeaderFileName, 'w') as fp:
       fp.write(json.dumps(TableHeader, indent=2))
       print('Header written to %s' % HeaderFileName)
    print('END DOWNLOAD')
    # Set comment
    # Get this table to LOCAL_TABLE_CACHE
    storage2cache(TableName)
    print('PROCESSED')

def saveHeader(TableName):
    ParameterList = prepareParlist(dotpar=True)    
    TableHeader = prepareHeader(ParameterList)
    with open(TableName+'.header', 'w') as fp:
       fp.write(json.dumps(TableHeader, indent=2))
    
# ---------- DATABASE FRONTEND END -------------

# simple implementation of getting a line list from a remote server
def getLinelist(local_name, query, api_key):
    return makeQuery(local_name)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# / GLOBABL API FUNCTIONS
# -------------------------------------------------------------------
# -------------------------------------------------------------------



# ---------------- FILTER ---------------------------------------------

def filter(TableName, Conditions):
    select(TableName=TableName, Conditions=Conditions, Output=False)



# Get atmospheric (natural) abundance
# for a specified isotopologue
# M - molecule number
# I - isotopologue number
def abundance(M, I):
    """
    INPUT PARAMETERS: 
        M: HITRAN molecule number
        I: HITRAN isotopologue number
    OUTPUT PARAMETERS: 
        Abbundance: natural abundance
    ---
    DESCRIPTION:
        Return natural (Earth) abundance of HITRAN isotolopogue.
    ---
    EXAMPLE OF USAGE:
        ab = abundance(1, 1) # H2O
    ---
    """
    return ISO[(M, I)][ISO_INDEX['abundance']]

# Get molecular mass
# for a specified isotopologue
# M - molecule number
# I - isotopologue number
def molecularMass(M, I):
    """
    INPUT PARAMETERS: 
        M: HITRAN molecule number
        I: HITRAN isotopologue number
    OUTPUT PARAMETERS: 
        MolMass: molecular mass
    ---
    DESCRIPTION:
        Return molecular mass of HITRAN isotolopogue.
    ---
    EXAMPLE OF USAGE:
        mass = molecularMass(1, 1) # H2O
    ---
    """
    return ISO[(M, I)][ISO_INDEX['mass']]

# Get molecule name
# for a specified isotopologue
# M - molecule number
# I - isotopologue number
def moleculeName(M):
    """
    INPUT PARAMETERS: 
        M: HITRAN molecule number
    OUTPUT PARAMETERS: 
        MolName: molecular name
    ---
    DESCRIPTION:
        Return name of HITRAN molecule.
    ---
    EXAMPLE OF USAGE:
        molname = moleculeName(1) # H2O
    ---
    """
    return ISO[(M, 1)][ISO_INDEX['mol_name']]

# Get isotopologue name
# for a specified isotopologue
# M - molecule number
# I - isotopologue number
def isotopologueName(M, I):
    """
    INPUT PARAMETERS: 
        M: HITRAN molecule number
        I: HITRAN isotopologue number
    OUTPUT PARAMETERS: 
        IsoMass: isotopologue mass
    ---
    DESCRIPTION:
        Return name of HITRAN isotolopogue.
    ---
    EXAMPLE OF USAGE:
        isoname = isotopologueName(1, 1) # H2O
    ---
    """
    return ISO[(M, I)][ISO_INDEX['iso_name']]

# ----------------------- table list ----------------------------------
def tableList():
    """
    INPUT PARAMETERS: 
        none
    OUTPUT PARAMETERS: 
        TableList: a list of available tables
    ---
    DESCRIPTION:
        Return a list of tables present in database.
    ---
    EXAMPLE OF USAGE:
        lst = tableList()
    ---
    """

    return getTableList()

# ----------------------- describe ----------------------------------
def describe(TableName):
    """
    INPUT PARAMETERS: 
        TableName: name of the table to describe
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Print information about table, including 
        parameter names, formats and wavenumber range.
    ---
    EXAMPLE OF USAGE:
        describe('sampletab')
    ---
    """
    describeTable(TableName)



def db_begin(db=None):
    """
    INPUT PARAMETERS: 
        db: database name (optional)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Open a database connection. A database is stored 
        in a folder given in db input parameter.
        Default=data
    ---
    EXAMPLE OF USAGE:
        db_begin('bar')
    ---
    """
    databaseBegin(db)

def db_commit():
    """
    INPUT PARAMETERS: 
        none
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Commit all changes made to opened database.
        All tables will be saved in corresponding files.
    ---
    EXAMPLE OF USAGE:
        db_commit()
    ---
    """
    databaseCommit()

# ------------------ QUERY HITRAN ---------------------------------------

def comment(TableName, Comment):
    LOCAL_TABLE_CACHE[TableName]['header']['comment'] = Comment

def fetch_by_ids(TableName, iso_id_list, numin, numax, ParameterGroups=[], Parameters=[]):
    """
    INPUT PARAMETERS: 
        TableName:   local table name to fetch in (required)
        iso_id_list: list of isotopologue id's    (required)
        numin:       lower wavenumber bound       (required)
        numax:       upper wavenumber bound       (required)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Download line-by-line data from HITRANonline server
        and save it to local table. The input parameter iso_id_list
        contains list of "global" isotopologue Ids (see help on ISO_ID).
        Note: this function is required if user wants to download
        multiple species into single table.
    ---
    EXAMPLE OF USAGE:
        fetch_by_ids('water', [1, 2, 3, 4], 4000, 4100)
    ---
    """
    if type(iso_id_list) not in set([list, tuple]):
       iso_id_list = [iso_id_list]
    queryHITRAN(TableName, iso_id_list, numin, numax,
                pargroups=ParameterGroups, params=Parameters)
    iso_names = [ISO_ID[i][ISO_ID_INDEX['iso_name']] for i in iso_id_list]
    Comment = 'Contains lines for '+', '.join(iso_names)
    Comment += ('\n in %.3f-%.3f wavenumber range' % (numin, numax))
    comment(TableName, Comment)

#def queryHITRAN(TableName, iso_id_list, numin, numax):
def fetch(TableName, M, I, numin, numax, ParameterGroups=[], Parameters=[]):
    """
    INPUT PARAMETERS: 
        TableName:   local table name to fetch in (required)
        M:           HITRAN molecule number       (required)
        I:           HITRAN isotopologue number   (required)
        numin:       lower wavenumber bound       (required)
        numax:       upper wavenumber bound       (required)
    OUTPUT PARAMETERS: 
        none
    ---
    DESCRIPTION:
        Download line-by-line data from HITRANonline server
        and save it to local table. The input parameters M and I
        are the HITRAN molecule and isotopologue numbers.
        This function results in a table containing single 
        isotopologue specie. To have multiple species in a 
        single table use fetch_by_ids instead.
    ---
    EXAMPLE OF USAGE:
        fetch('HOH', 1, 1, 4000, 4100)
    ---
    """
    queryHITRAN(TableName, [ISO[(M, I)][ISO_INDEX['id']]], numin, numax,
                pargroups=ParameterGroups, params=Parameters)
    iso_name = ISO[(M, I)][ISO_INDEX['iso_name']]
    Comment = 'Contains lines for '+iso_name
    Comment += ('\n in %.3f-%.3f wavenumber range' % (numin, numax))
    comment(TableName, Comment)



# ------------------ LINESHAPES -----------------------------------------

# ------------------ complex probability function -----------------------
# define static data
zone = ComplexType(1.0e0 + 0.0e0j)
zi = ComplexType(0.0e0 + 1.0e0j)
tt = FloatType64([0.5e0, 1.5e0, 2.5e0, 3.5e0, 4.5e0, 5.5e0, 6.5e0, 7.5e0, 8.5e0, 9.5e0, 10.5e0, 11.5e0, 12.5e0, 13.5e0, 14.5e0])
pipoweronehalf = FloatType64(0.564189583547756e0)

# "naive" implementation for benchmarks
def cpf3(X, Y):

    # X, Y, WR, WI - numpy arrays
    if type(X) != ndarray: 
        if type(X) not in set([list, tuple]):
            X = array([X])
        else:
            X = array(X)
    if type(Y) != ndarray: 
        if type(Y) not in set([list, tuple]):
            Y = array([Y])
        else:
            Y = array(Y)

    zm1 = zone/ComplexType(X + zi*Y) # maybe redundant
    zm2 = zm1**2
    zsum = zone
    zterm=zone

    for tt_i in tt:
        zterm *= zm2*tt_i
        zsum += zterm
    
    zsum *= zi*zm1*pipoweronehalf
    
    return zsum.real, zsum.imag

T = FloatType64([0.314240376e0, 0.947788391e0, 1.59768264e0, 2.27950708e0, 3.02063703e0, 3.8897249e0])
U = FloatType64([1.01172805e0, -0.75197147e0, 1.2557727e-2, 1.00220082e-2, -2.42068135e-4, 5.00848061e-7])
S = FloatType64([1.393237e0, 0.231152406e0, -0.155351466e0, 6.21836624e-3, 9.19082986e-5, -6.27525958e-7])

# Complex probability function implementation (Humlicek)
def cpf(X, Y):

    # X, Y, WR, WI - numpy arrays
    if type(X) != ndarray: 
        if type(X) not in set([list, tuple]):
            X = array([X])
        else:
            X = array(X)
    if type(Y) != ndarray: 
        if type(Y) not in set([list, tuple]):
            Y = array([Y])
        else:
            Y = array(Y)
    
    # REGION3
    index_REGION3 = where(sqrt(X**2 + Y**2) > FloatType64(8.0e0))
    X_REGION3 = X[index_REGION3]
    Y_REGION3 = Y[index_REGION3]
    zm1 = zone/ComplexType(X_REGION3 + zi*Y_REGION3)
    zm2 = zm1**2
    zsum_REGION3 = zone
    zterm=zone
    for tt_i in tt:
        zterm *= zm2*tt_i
        zsum_REGION3 += zterm
    zsum_REGION3 *= zi*zm1*pipoweronehalf
    
    index_REGION12 = setdiff1d(array(arange(len(X))), array(index_REGION3))
    X_REGION12 = X[index_REGION12]
    Y_REGION12 = Y[index_REGION12]
    
    WR = FloatType64(0.0e0)
    WI = FloatType64(0.0e0)
    
    # REGION12
    Y1_REGION12 = Y_REGION12 + FloatType64(1.5e0)
    Y2_REGION12 = Y1_REGION12**2

    # REGION2    
    subindex_REGION2 = where((Y_REGION12 <= 0.85e0) & 
                             (abs(X_REGION12) >= (18.1e0*Y_REGION12 + 1.65e0)))
    
    index_REGION2 = index_REGION12[subindex_REGION2]
    
    X_REGION2 = X[index_REGION2]
    Y_REGION2 = Y[index_REGION2]
    Y1_REGION2 = Y1_REGION12[subindex_REGION2]
    Y2_REGION2 = Y2_REGION12[subindex_REGION2]
    Y3_REGION2 = Y_REGION2 + FloatType64(3.0e0)
    
    WR_REGION2 = WR
    WI_REGION2 = WI

    WR_REGION2 = zeros(len(X_REGION2))
    ii = abs(X_REGION2) < FloatType64(12.0e0)
    WR_REGION2[ii] = exp(-X_REGION2[ii]**2)
    WR_REGION2[~ii] = WR
    
    for I in range(6):
        R_REGION2 = X_REGION2 - T[I]
        R2_REGION2 = R_REGION2**2
        D_REGION2 = FloatType64(1.0e0) / (R2_REGION2 + Y2_REGION2)
        D1_REGION2 = Y1_REGION2 * D_REGION2
        D2_REGION2 = R_REGION2 * D_REGION2
        WR_REGION2 = WR_REGION2 + Y_REGION2 * (U[I]*(R_REGION2*D2_REGION2 - 1.5e0*D1_REGION2) + 
                                               S[I]*Y3_REGION2*D2_REGION2)/(R2_REGION2 + 2.25e0)
        R_REGION2 = X_REGION2 + T[I]
        R2_REGION2 = R_REGION2**2                
        D_REGION2 = FloatType64(1.0e0) / (R2_REGION2 + Y2_REGION2)
        D3_REGION2 = Y1_REGION2 * D_REGION2
        D4_REGION2 = R_REGION2 * D_REGION2
        WR_REGION2 = WR_REGION2 + Y_REGION2 * (U[I]*(R_REGION2*D4_REGION2 - 1.5e0*D3_REGION2) - 
                                               S[I]*Y3_REGION2*D4_REGION2)/(R2_REGION2 + 2.25e0)
        WI_REGION2 = WI_REGION2 + U[I]*(D2_REGION2 + D4_REGION2) + S[I]*(D1_REGION2 - D3_REGION2)

    # REGION3
    index_REGION1 = setdiff1d(array(index_REGION12), array(index_REGION2))
    X_REGION1 = X[index_REGION1]
    Y_REGION1 = X[index_REGION1]
    
    subindex_REGION1 = setdiff1d(array(arange(len(index_REGION12))), array(subindex_REGION2))
    Y1_REGION1 = Y1_REGION12[subindex_REGION1]
    Y2_REGION1 = Y2_REGION12[subindex_REGION1]
    
    WR_REGION1 = WR
    WI_REGION1 = WI  
    
    for I in range(6):
        R_REGION1 = X_REGION1 - T[I]
        D_REGION1 = FloatType64(1.0e0) / (R_REGION1**2 + Y2_REGION1)
        D1_REGION1 = Y1_REGION1 * D_REGION1
        D2_REGION1 = R_REGION1 * D_REGION1
        R_REGION1 = X_REGION1 + T[I]
        D_REGION1 = FloatType64(1.0e0) / (R_REGION1**2 + Y2_REGION1)
        D3_REGION1 = Y1_REGION1 * D_REGION1
        D4_REGION1 = R_REGION1 * D_REGION1
        
        WR_REGION1 = WR_REGION1 + U[I]*(D1_REGION1 + D3_REGION1) - S[I]*(D2_REGION1 - D4_REGION1)
        WI_REGION1 = WI_REGION1 + U[I]*(D2_REGION1 + D4_REGION1) + S[I]*(D1_REGION1 - D3_REGION1)

    # total result
    WR_TOTAL = zeros(len(X))
    WI_TOTAL = zeros(len(X))
    # REGION3
    WR_TOTAL[index_REGION3] = zsum_REGION3.real
    WI_TOTAL[index_REGION3] = zsum_REGION3.imag
    # REGION2
    WR_TOTAL[index_REGION2] = WR_REGION2
    WI_TOTAL[index_REGION2] = WI_REGION2
    # REGION1
    WR_TOTAL[index_REGION1] = WR_REGION1
    WI_TOTAL[index_REGION1] = WI_REGION1
    
    return WR_TOTAL, WI_TOTAL


hcpf = cpf # stub for initial cpf
    
# ------------------ Schreier CPF ------------------------

# "Optimized implementations of rational approximations 
#  for the Voigt and complex error function".
# Franz Schreier. JQSRT 112 (2011) 1010-10250
# doi:10.1016/j.jqsrt.2010.12.010

# Enable this if numpy.polyval doesn't perform well.
"""    
def polyval(p, x):
    y = zeros(x.shape, dtype=float)
    for i, v in enumerate(p):
        y *= x
        y += v
    return y
""";
    
def cef(x, y, N):
    # Computes the function w(z) = exp(-zA2) erfc(-iz) using a rational
    # series with N terms. It is assumed that Im(z) > 0 or Im(z) = 0.
    z = x + 1.0j*y
    M = 2*N; M2 = 2*M; k = arange(-M+1, M) #'; # M2 = no. of sampling points.
    L = sqrt(N/sqrt(2)); # Optimal choice of L.
    theta = k*pi/M; t = L*tan(theta/2); # Variables theta and t.
    #f = exp(-t.A2)*(LA2+t.A2); f = [0; f]; # Function to be transformed.
    f = zeros(len(t)+1); f[0] = 0
    f[1:] = exp(-t**2)*(L**2+t**2)
    #f = insert(exp(-t**2)*(L**2+t**2), 0, 0)
    a = real(fft(fftshift(f)))/M2; # Coefficients of transform.
    a = flipud(a[1:N+1]); # Reorder coefficients.
    Z = (L+1.0j*z)/(L-1.0j*z); p = polyval(a, Z); # Polynomial evaluation.
    w = 2*p/(L-1.0j*z)**2+(1/sqrt(pi))/(L-1.0j*z); # Evaluate w(z).
    return w

# weideman24 by default    
#weideman24 = lambda x, y: cef(x, y, 24)
weideman = lambda x, y, n: cef(x, y, n)

def hum1_wei(x, y, n=24):
    t = y-1.0j*x
    cerf=1/sqrt(pi)*t/(0.5+t**2)
    """
    z = x+1j*y
    cerf = 1j*z/sqrt(pi)/(z**2-0.5)
    """
    mask = abs(x)+y<15.0
    if any(mask):
        w24 = weideman(x[mask], y[mask], n)
        place(cerf, mask, w24)
    return cerf.real, cerf.imag

VARIABLES['CPF'] = hum1_wei
#VARIABLES['CPF'] = cpf
    
# ------------------ Hartmann-Tran Profile (HTP) ------------------------
def pcqsdhc(sg0, GamD, Gam0, Gam2, Shift0, Shift2, anuVC, eta, sg, Ylm=0.0):
    #-------------------------------------------------
    #      "pCqSDHC": partially-Correlated quadratic-Speed-Dependent Hard-Collision
    #      Subroutine to Compute the complex normalized spectral shape of an 
    #      isolated line by the pCqSDHC model
    #
    #      Reference:
    #      H. Tran, N.H. Ngo, J.-M. Hartmann.
    #      Efficient computation of some speed-dependent isolated line profiles.
    #      JQSRT, Volume 129, November 2013, Pages 199–203
    #      http://dx.doi.org/10.1016/j.jqsrt.2013.06.015
    #
    #      Input/Output Parameters of Routine (Arguments or Common)
    #      ---------------------------------
    #      T          : Temperature in Kelvin (Input).
    #      amM1       : Molar mass of the absorber in g/mol(Input).
    #      sg0        : Unperturbed line position in cm-1 (Input).
    #      GamD       : Doppler HWHM in cm-1 (Input)
    #      Gam0       : Speed-averaged line-width in cm-1 (Input).       
    #      Gam2       : Speed dependence of the line-width in cm-1 (Input).
    #      anuVC      : Velocity-changing frequency in cm-1 (Input).
    #      eta        : Correlation parameter, No unit (Input).
    #      Shift0     : Speed-averaged line-shift in cm-1 (Input).
    #      Shift2     : Speed dependence of the line-shift in cm-1 (Input)       
    #      sg         : Current WaveNumber of the Computation in cm-1 (Input).
    #      Ylm        : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    #
    #      Output Quantities (through Common Statements)
    #      -----------------
    #      LS_pCqSDHC_R: Real part of the normalized spectral shape (cm)
    #      LS_pCqSDHC_I: Imaginary part of the normalized spectral shape (cm)
    #
    #      Called Routines: 'CPF'      (Complex Probability Function)
    #      ---------------  'CPF3'      (Complex Probability Function for the region 3)
    #
    #      Called By: Main Program
    #      ---------
    #
    #     Double Precision Version
    #
    #-------------------------------------------------
    
    # sg is the only vector argument which is passed to function
    
    if type(sg) not in set([array, ndarray, list, tuple]):
        sg = array([sg])
    
    number_of_points = len(sg)
    Aterm_GLOBAL = zeros(number_of_points, dtype=ComplexType)
    Bterm_GLOBAL = zeros(number_of_points, dtype=ComplexType)

    cte=sqrt(log(2.0e0))/GamD
    rpi=sqrt(pi)
    iz = ComplexType(0.0e0 + 1.0e0j)

    c0 = ComplexType(Gam0 + 1.0e0j*Shift0)
    c2 = ComplexType(Gam2 + 1.0e0j*Shift2)
    c0t = ComplexType((1.0e0 - eta) * (c0 - 1.5e0 * c2) + anuVC)
    c2t = ComplexType((1.0e0 - eta) * c2)

    # PART1
    if abs(c2t) == 0.0e0:
        Z1 = (iz*(sg0 - sg) + c0t) * cte
        xZ1 = -Z1.imag
        yZ1 = Z1.real
        WR1, WI1 = VARIABLES['CPF'](xZ1, yZ1)
        Aterm_GLOBAL = rpi*cte*ComplexType(WR1 + 1.0e0j*WI1)
        index_Z1 = abs(Z1) <= 4.0e3
        index_NOT_Z1 = ~index_Z1
        if any(index_Z1):
            Bterm_GLOBAL = rpi*cte*((1.0e0 - Z1**2)*ComplexType(WR1 + 1.0e0j*WI1) + Z1/rpi)
        if any(index_NOT_Z1):
            Bterm_GLOBAL = cte*(rpi*ComplexType(WR1 + 1.0e0j*WI1) + 0.5e0/Z1 - 0.75e0/(Z1**3))
    else:
        # PART2, PART3 AND PART4   (PART4 IS A MAIN PART)

        # X - vector, Y - scalar
        X = (iz * (sg0 - sg) + c0t) / c2t
        Y = ComplexType(1.0e0 / ((2.0e0*cte*c2t))**2)
        csqrtY = (Gam2 - iz*Shift2) / (2.0e0*cte*(1.0e0-eta) * (Gam2**2 + Shift2**2))

        index_PART2 = abs(X) <= 3.0e-8 * abs(Y)
        index_PART3 = (abs(Y) <= 1.0e-15 * abs(X)) & ~index_PART2
        index_PART4 = ~ (index_PART2 | index_PART3)
        
        # PART4
        if any(index_PART4):
            X_TMP = X[index_PART4]
            Z1 = sqrt(X_TMP + Y) - csqrtY
            Z2 = Z1 + FloatType64(2.0e0) * csqrtY
            xZ1 = -Z1.imag
            yZ1 =  Z1.real
            xZ2 = -Z2.imag
            yZ2 =  Z2.real
            SZ1 = sqrt(xZ1**2 + yZ1**2)
            SZ2 = sqrt(xZ2**2 + yZ2**2)
            DSZ = abs(SZ1 - SZ2)
            SZmx = maximum(SZ1, SZ2)
            SZmn = minimum(SZ1, SZ2)
            length_PART4 = len(index_PART4)
            WR1_PART4 = zeros(length_PART4)
            WI1_PART4 = zeros(length_PART4)
            WR2_PART4 = zeros(length_PART4)
            WI2_PART4 = zeros(length_PART4)
            index_CPF3 = (DSZ <= 1.0e0) & (SZmx > 8.0e0) & (SZmn <= 8.0e0)
            index_CPF = ~index_CPF3 # can be removed
            if any(index_CPF3):
                WR1, WI1 = cpf3(xZ1[index_CPF3], yZ1[index_CPF3])
                WR2, WI2 = cpf3(xZ2[index_CPF3], yZ2[index_CPF3])
                WR1_PART4[index_CPF3] = WR1
                WI1_PART4[index_CPF3] = WI1
                WR2_PART4[index_CPF3] = WR2
                WI2_PART4[index_CPF3] = WI2
            if any(index_CPF):
                WR1, WI1 = VARIABLES['CPF'](xZ1[index_CPF], yZ1[index_CPF])
                WR2, WI2 = VARIABLES['CPF'](xZ2[index_CPF], yZ2[index_CPF])
                WR1_PART4[index_CPF] = WR1
                WI1_PART4[index_CPF] = WI1
                WR2_PART4[index_CPF] = WR2
                WI2_PART4[index_CPF] = WI2
            
            Aterm = rpi*cte*(ComplexType(WR1_PART4 + 1.0e0j*WI1_PART4) - ComplexType(WR2_PART4+1.0e0j*WI2_PART4))
            Bterm = (-1.0e0 +
                      rpi/(2.0e0*csqrtY)*(1.0e0 - Z1**2)*ComplexType(WR1_PART4 + 1.0e0j*WI1_PART4)-
                      rpi/(2.0e0*csqrtY)*(1.0e0 - Z2**2)*ComplexType(WR2_PART4 + 1.0e0j*WI2_PART4)) / c2t
            Aterm_GLOBAL[index_PART4] = Aterm
            Bterm_GLOBAL[index_PART4] = Bterm

        # PART2
        if any(index_PART2):
            X_TMP = X[index_PART2]
            Z1 = (iz*(sg0 - sg[index_PART2]) + c0t) * cte
            Z2 = sqrt(X_TMP + Y) + csqrtY
            xZ1 = -Z1.imag
            yZ1 = Z1.real
            xZ2 = -Z2.imag
            yZ2 = Z2.real
            WR1_PART2, WI1_PART2 = VARIABLES['CPF'](xZ1, yZ1)
            WR2_PART2, WI2_PART2 = VARIABLES['CPF'](xZ2, yZ2)
            Aterm = rpi*cte*(ComplexType(WR1_PART2 + 1.0e0j*WI1_PART2) - ComplexType(WR2_PART2 + 1.0e0j*WI2_PART2))
            Bterm = (-1.0e0 +
                      rpi/(2.0e0*csqrtY)*(1.0e0 - Z1**2)*ComplexType(WR1_PART2 + 1.0e0j*WI1_PART2)-
                      rpi/(2.0e0*csqrtY)*(1.0e0 - Z2**2)*ComplexType(WR2_PART2 + 1.0e0j*WI2_PART2)) / c2t
            Aterm_GLOBAL[index_PART2] = Aterm
            Bterm_GLOBAL[index_PART2] = Bterm
            
        # PART3
        if any(index_PART3):
            X_TMP = X[index_PART3]
            xZ1 = -sqrt(X_TMP + Y).imag
            yZ1 = sqrt(X_TMP + Y).real
            WR1_PART3, WI1_PART3 =  VARIABLES['CPF'](xZ1, yZ1)
            index_ABS = abs(sqrt(X_TMP)) <= 4.0e3
            index_NOT_ABS = ~index_ABS
            Aterm = zeros(len(index_PART3), dtype=ComplexType)
            Bterm = zeros(len(index_PART3), dtype=ComplexType)
            if any(index_ABS):
                xXb = -sqrt(X).imag
                yXb = sqrt(X).real
                WRb, WIb = VARIABLES['CPF'](xXb, yXb)
                Aterm[index_ABS] = (2.0e0*rpi/c2t)*(1.0e0/rpi - sqrt(X_TMP[index_ABS])*ComplexType(WRb + 1.0e0j*WIb))
                Bterm[index_ABS] = (1.0e0/c2t)*(-1.0e0+
                                  2.0e0*rpi*(1.0e0 - X_TMP[index_ABS]-2.0e0*Y)*(1.0e0/rpi-sqrt(X_TMP[index_ABS])*ComplexType(WRb + 1.0e0j*WIb))+
                                  2.0e0*rpi*sqrt(X_TMP[index_ABS] + Y)*ComplexType(WR1_PART3 + 1.0e0j*WI1_PART3))
            if any(index_NOT_ABS):
                Aterm[index_NOT_ABS] = (1.0e0/c2t)*(1.0e0/X_TMP[index_NOT_ABS] - 1.5e0/(X_TMP[index_NOT_ABS]**2))
                Bterm[index_NOT_ABS] = (1.0e0/c2t)*(-1.0e0 + (1.0e0 - X_TMP[index_NOT_ABS] - 2.0e0*Y)*
                                        (1.0e0/X_TMP[index_NOT_ABS] - 1.5e0/(X_TMP[index_NOT_ABS]**2))+
                                         2.0e0*rpi*sqrt(X_TMP[index_NOT_ABS] + Y)*ComplexType(WR1 + 1.0e0j*WI1))
            Aterm_GLOBAL[index_PART3] = Aterm
            Bterm_GLOBAL[index_PART3] = Bterm
            
    # common part
    # LINE MIXING PART NEEDS FURTHER TESTING!!!
    LS_pCqSDHC = (1.0e0/pi) * (Aterm_GLOBAL / (1.0e0 - (anuVC-eta*(c0-1.5e0*c2))*Aterm_GLOBAL + eta*c2*Bterm_GLOBAL))
    return LS_pCqSDHC.real + Ylm*LS_pCqSDHC.imag, LS_pCqSDHC.imag



# ------------------  CROSS-SECTIONS, XSECT.PY --------------------------------

# set interfaces for profiles

def PROFILE_HT(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, NuVC, Eta, WnGrid, YRosen=0.0, Sw=1.0):
    """
    #-------------------------------------------------
    #      "pCqSDHC": partially-Correlated quadratic-Speed-Dependent "Hard-Collision"
    #      Subroutine to Compute the complex normalized spectral shape of an 
    #      isolated line by the pCqSDHC model
    #
    #      References:
    #
    #      1) N.H. Ngo, D. Lisak, H. Tran, J.-M. Hartmann.
    #         An isolated line-shape model to go beyond the Voigt profile in 
    #         spectroscopic databases and radiative transfer codes.
    #         JQSRT, Volume 129, November 2013, Pages 89–100
    #         http://dx.doi.org/10.1016/j.jqsrt.2013.05.034
    #
    #      2) H. Tran, N.H. Ngo, J.-M. Hartmann.
    #         Efficient computation of some speed-dependent isolated line profiles.
    #         JQSRT, Volume 129, November 2013, Pages 199–203
    #         http://dx.doi.org/10.1016/j.jqsrt.2013.06.015
    #
    #      3) H. Tran, N.H. Ngo, J.-M. Hartmann.
    #         Erratum to “Efficient computation of some speed-dependent isolated line profiles”.
    #         JQSRT, Volume 134, February 2014, Pages 104
    #         http://dx.doi.org/10.1016/j.jqsrt.2013.10.015
    #
    #      Input/Output Parameters of Routine (Arguments or Common)
    #      ---------------------------------
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      Gamma2    : Speed dependence of the line-width in cm-1 (Input).
    #      NuVC      : Velocity-changing frequency in cm-1 (Input).
    #      Eta       : Correlation parameter, No unit (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      Delta2    : Speed dependence of the line-shift in cm-1 (Input)       
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    #
    #      The function has two outputs:
    #      -----------------
    #      (1): Real part of the normalized spectral shape (cm)
    #      (2): Imaginary part of the normalized spectral shape (cm)
    #
    #      Called Routines: 'CPF'       (Complex Probability Function)
    #      ---------------  'CPF3'      (Complex Probability Function for the region 3)
    #
    #      Based on a double precision Fortran version
    #
    #-------------------------------------------------
    """
    return Sw*pcqsdhc(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, NuVC, Eta, WnGrid, YRosen)[0]

PROFILE_HTP = PROFILE_HT # stub for backwards compatibility

def PROFILE_SDRAUTIAN(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, NuVC, WnGrid, YRosen=0.0, Sw=1.0):
    """
    # Speed dependent Rautian profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      Gamma2    : Speed dependence of the line-width in cm-1 (Input).
    #      NuVC      : Velocity-changing frequency in cm-1 (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      Delta2    : Speed dependence of the line-shift in cm-1 (Input)       
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    """
    return Sw*pcqsdhc(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, NuVC, cZero, WnGrid, YRosen)[0]

def PROFILE_RAUTIAN(Nu, GammaD, Gamma0, Delta0, NuVC, WnGrid, Ylm=0.0, Sw=1.0):
    """
    # Rautian profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      NuVC      : Velocity-changing frequency in cm-1 (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    """
    return Sw*pcqsdhc(Nu, GammaD, Gamma0, cZero, Delta0, cZero, NuVC, cZero, WnGrid, YRosen)[0]

def PROFILE_SDVOIGT(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, WnGrid, YRosen=0.0, Sw=1.0):
    """
    # Speed dependent Voigt profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      Gamma2    : Speed dependence of the line-width in cm-1 (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      Delta2    : Speed dependence of the line-shift in cm-1 (Input)       
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    """
    if FLAG_DEBUG_PROFILE: 
        print('PROFILE_SDVOIGT>>>', Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, WnGrid, YRosen, Sw)
    return Sw*pcqsdhc(Nu, GammaD, Gamma0, Gamma2, Delta0, Delta2, cZero, cZero, WnGrid, YRosen)[0]
    
def PROFILE_VOIGT(Nu, GammaD, Gamma0, Delta0, WnGrid, YRosen=0.0, Sw=1.0):
    """
    # Voigt profile based on HTP.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).       
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    """
    #return PROFILE_HTP(Nu, GammaD, Gamma0, cZero, cZero, cZero, cZero, cZero, WnGrid, YRosen)[0]
    if FLAG_DEBUG_PROFILE: 
        print('PROFILE_VOIGT>>>', Nu, GammaD, Gamma0, Delta0, WnGrid, YRosen, Sw)
    return Sw*pcqsdhc(Nu, GammaD, Gamma0, cZero, Delta0, cZero, cZero, cZero, WnGrid, YRosen)[0]

def PROFILE_LORENTZ(Nu, Gamma0, Delta0, WnGrid, YRosen=0.0, Sw=1.0):
    """
    # Lorentz profile.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      Gamma0    : Speed-averaged line-width in cm-1 (Input).
    #      Delta0    : Speed-averaged line-shift in cm-1 (Input).
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    #      YRosen    : 1st order (Rosenkranz) line mixing coefficients in cm-1 (Input)
    """
    # reduce the extra calculations in the case if YRosen is zero:
    if YRosen == 0.0:
        return Sw*Gamma0/(pi*(Gamma0**2+(WnGrid+Delta0-Nu)**2))
    else:
        return Sw*(Gamma0+YRosen*(WnGrid+Delta0-Nu))/(pi*(Gamma0**2+(WnGrid+Delta0-Nu)**2))

def PROFILE_DOPPLER(Nu, GammaD, WnGrid, Sw=1.0):
    """
    # Doppler profile.
    # Input parameters:
    #      Nu        : Unperturbed line position in cm-1 (Input).
    #      GammaD    : Doppler HWHM in cm-1 (Input)
    #      WnGrid    : Current WaveNumber of the Computation in cm-1 (Input).
    """
    return Sw*cSqrtLn2divSqrtPi*exp(-cLn2*((WnGrid-Nu)/GammaD)**2)/GammaD

# Volume concentration of all gas molecules at the pressure p and temperature T
def volumeConcentration(p, T):
    return (p/9.869233e-7)/(cBolts*T) # CGS

# ------------------------------- PARAMETER DEPENDENCIES --------------------------------

# THE LOGIC OF THIS SECTION IS THAT NOTHING (OR AT LEAST MINUMUM) SHOULD BE HARD-CODED INTO THE GENERIC ABSCOEF ROUTINES
# TRYING TO AVOID THE OBJECT ORIENTED APPROACH HERE IN ORDER TO CORRESPOND TO THE OVERALL STYLE OF THE PACKAGE

def ladder(parname, species, envdep_presets, TRANS, flag_exception=False): # priority search for the parameters
    INFO = {}  
    if FLAG_DEBUG_LADDER: print('\nladder>>> ======================')
    if FLAG_DEBUG_LADDER: print('ladder>>> Calculating %s for %s broadener'%(parname, species))
    if FLAG_DEBUG_LADDER: print('ladder>>> Envdep presets: ', envdep_presets)
    for profile, envdep in envdep_presets:
        try:
            if FLAG_DEBUG_LADDER: print('\nladder>>> Trying: ', profile, envdep)
            INFO, ARGS = PRESSURE_INDUCED_ENVDEP[profile][parname][envdep]['getargs'](species, TRANS)
            parval_species = PRESSURE_INDUCED_ENVDEP[profile][parname][envdep]['depfunc'](**ARGS)
            if FLAG_DEBUG_LADDER: print('ladder>>> success!\n')
            return INFO, parval_species
        #except KeyError as e:
        except Exception as e:
            if flag_exception:
                raise e
            else:
                INFO['status'] = e.__class__.__name__+': '+str(e)
                if FLAG_DEBUG_LADDER: print('ladder>>>', e.__class__.__name__+': '+str(e), ': ', parname, profile, envdep)
    if FLAG_DEBUG_LADDER: print('ladder>>> ')
    return INFO, 0

def calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO):
    """
    Default function for calculating the pressure-induced parameters.
    Use this function only if the final lineshape parameter needs summation
    over the mixture/diluent components.
    """
    parval = 0
    Diluent = TRANS['Diluent']
    calc_info_flag = False
    if type(CALC_INFO) is dict:
        calc_info_flag = True
        CALC_INFO[parname] = {'mixture':{}}
    for species in Diluent:
        abun = Diluent[species]
        INFO, parval_species = ladder(parname, species, envdep_presets, TRANS)
        parval += abun*parval_species
        if calc_info_flag: 
            CALC_INFO[parname]['mixture'][species] = {'args':INFO, 'value':parval_species}
    if calc_info_flag: 
        CALC_INFO[parname]['status'] = 'ok'
        CALC_INFO[parname]['value'] = parval
    return parval

def calculate_parameter_Nu(dummy, TRANS, CALC_INFO=None):
    nu = TRANS['nu']
    if type(CALC_INFO) is dict: 
        CALC_INFO['GammaD'] = {'value':nu, 'mixture':{'generic':{'args':{}}}}
    return nu

def calculate_parameter_Sw(dummy, TRANS, CALC_INFO=None):
    molec_id = TRANS['molec_id']
    local_iso_id = TRANS['local_iso_id']
    nu = TRANS['nu']
    sw = TRANS['sw']
    T = TRANS['T']
    Tref = TRANS['T_ref']
    SigmaT = TRANS['SigmaT']
    SigmaTref = TRANS['SigmaT_ref']
    elower = TRANS['elower']
    Sw_calc = EnvironmentDependency_Intensity(sw, T, Tref, SigmaT, SigmaTref, elower, nu)
    if 'Abundances' in TRANS:
        Sw_calc *= TRANS['Abundances'][(molec_id, local_iso_id)]/abundance(molec_id, local_iso_id)
    if type(CALC_INFO) is dict:
        CALC_INFO['Sw'] = {
            'value':Sw_calc, 
            'mixture':{
                'generic': {
                    'args': {
                        'SigmaT':{'value':SigmaT, 'source':'<calc>'}, 
                        'SigmaT_ref':{'value':SigmaTref, 'source':'<calc>'}, 
                        'elower':{'value':elower, 'source':'elower'}, 
                        'nu':{'value':nu, 'source':'nu'}, 
                    }
                }
            }
        }

    return Sw_calc
    #return {'value':TRANS['sw'], 'info':{}} # SHOULD BE REDONE TO INCLUDE THE ABUNDANCES OF RADIATIVELY ACTIVE SPECIES
    
def calculate_parameter_GammaD(dummy, TRANS, CALC_INFO=None):
    """
    Calculate Doppler broadening HWHM for given environment.
    """
    Diluent = TRANS['Diluent']
    T = TRANS['T']
    p = TRANS['p']
    MoleculeNumberDB = TRANS['molec_id']
    IsoNumberDB = TRANS['local_iso_id']
    LineCenterDB = TRANS['nu']
    cMassMol = 1.66053873e-27
    molmass = molecularMass(MoleculeNumberDB, IsoNumberDB)
    fSqrtMass = sqrt(molmass)
    cc_ = 2.99792458e8
    cBolts_ = 1.3806503e-23
    
    #GammaD = (cSqrt2Ln2/cc_)*sqrt(cBolts_/cMassMol)*sqrt(T) * LineCenterDB/fSqrtMass
    # OR 
    m = molmass * cMassMol * 1000
    GammaD = sqrt(2*cBolts*T*log(2)/m/cc**2)*LineCenterDB
    
    if type(CALC_INFO) is dict: 
        CALC_INFO['GammaD'] = {
            'value':GammaD, 
            'mixture':{
                'generic':{
                    'args':{
                        'T': {'value':T}, 
                        'p': {'value':p}, 
                        'molmass': {'value':molmass, 'source':'<calc>'}
                    }
                }
            }
        }
    return GammaD
    
def calculate_parameter_Gamma0(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate pressure-induced broadening HWHM for given Environment and TRANS.
    """
    parname = 'Gamma0'
    return calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO)

def calculate_parameter_Delta0(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate pressure-induced line shift for given Environment and TRANS.
    """
    parname = 'Delta0'
    return calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO)

def calculate_parameter_Gamma2(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate speed dependence of pressure-induced broadening HWHM for given Environment and TRANS.
    """
    parname = 'Gamma2'
    return calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO)

def calculate_parameter_Delta2(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate speed dependence of pressure-induced line shift for given Environment and TRANS.
    """
    parname = 'Delta2'
    return calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO)

def calculate_parameter_Eta(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate correlation parameter for given Environment and TRANS.
    """
    Diluent = TRANS['Diluent']
    if type(CALC_INFO) is not dict:
        return 0
    CALC_INFO['Eta'] = {'mixture':{}}
    Eta = 0
    Gamma2 = CALC_INFO['Gamma2']['value']
    Delta2 = CALC_INFO['Delta2']['value']
    Eta_denom = Gamma2-1j*Delta2
    for species in Diluent:
        abun = Diluent[species]
        EtaDB = TRANS.get('eta_HT_%s'%species, 0)
        Gamma2T = CALC_INFO['Gamma2']['mixture'][species]['value']
        Delta2T = CALC_INFO['Delta2']['mixture'][species]['value']
        Eta_species = EtaDB*abun*(Gamma2T-1j*Delta2T)
        if Eta_denom!=0: Eta_species/=Eta_denom
        CALC_INFO['Eta']['mixture'][species] = {
            'value':Eta_species, 
            'args':{
                'Gamma0':{'value':Gamma0, 'source':'<calc>'}, 
                'Delta0':{'value':Delta0, 'source':'<calc>'}, 
            }
        }
        Eta += Eta_species
    return Eta

def calculate_parameter_NuVC(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate velocity collision frequency for given Environment and TRANS.
    """
    Diluent = TRANS['Diluent']
    if type(CALC_INFO) is not dict:
        return 0
    CALC_INFO['NuVC'] = {'mixture':{}}
    Gamma0 = CALC_INFO['Gamma0']['value']
    Delta0 = CALC_INFO['Delta0']['value']
    Eta = CALC_INFO['Eta']['value']
    NuVC = Eta*(Gamma0-1j*Shift0)
    p = TRANS['p']
    T = TRANS['T']
    Tref = CALC_INFO['Tref']['value']
    for species in Diluent:

        abun = Diluent[species]

        # 1st NuVC component: weighted sum of NuVC_i
        NuVCDB = TRANS.get('nu_HT_%s'%species, 0)
        KappaDB = TRANS.get('kappa_HT_%s'%species, 0)
        NuVC += NuVCDB*(Tref/T)**KappaDB*p

        # 2nd NuVC component (with negative sign)
        Gamma0T = CALC_INFO['Gamma0']['mixture'][species]['value']
        Delta0T = CALC_INFO['Delta0']['mixture'][species]['value']
        EtaDB = TRANS.get('eta_HT_%s'%species, 0)
        NuVC -= EtaDB*abun*(Gamma0T-1j*Delta0T)

        CALC_INFO['NuVC']['mixture'][species] = {
            'value':NuVC_species, 
            'args':{
                'Gamma0':{'value':Gamma0, 'source':'<calc>'}, 
                'Delta0':{'value':Delta0, 'source':'<calc>'}, 
            }
        }

        NuVC += NuVC_species

    return NuVC
    
def calculate_parameter_YRosen(envdep_presets, TRANS, CALC_INFO=None):
    """
    Calculate pressure-induced 1st order line mixing parameter for given Environment and TRANS.
    """
    parname = 'YRosen'
    return calculate_parameter_PI(parname, envdep_presets, TRANS, CALC_INFO)
    
def calculateProfileParameters(envdep_presets, parameters, TRANS, CALC_INFO=None, exclude=None):
    """
    Get the Line context on the input, and return the dictionary with the "abstract" parameters.
    """
    PARAMS = {}
    for pname, pfunc in parameters:
        if exclude and pname in exclude:
            pval_default = 0
            PARAMS[pname] = pval_default # don't calculate parameter if it is present in exclude set
            if type(CALC_INFO) is dict:
                CALC_INFO[pname] = {
                    'value': pval_default, 
                    'status': 'excluded'
                }
        else:
            PARAMS[pname] = pfunc(envdep_presets, TRANS=TRANS, CALC_INFO=CALC_INFO)
    return PARAMS
    
def calculateProfileParametersDoppler(TRANS, CALC_INFO=None, exclude=None):
    """
    Get values for abstract profile parameters for Doppler profile.
    """
    envdep_presets = [('Doppler', 'default')]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('GammaD', calculate_parameter_GammaD), 
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)
    
def calculateProfileParametersLorentz(TRANS, CALC_INFO=None, exclude=None):
    """
    Get values for abstract profile parameters for Lorentz profile.
    """
    envdep_presets = [('Lorentz', 'default')]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('Gamma0', calculate_parameter_Gamma0), 
        ('Delta0', calculate_parameter_Delta0), 
        ('YRosen', calculate_parameter_YRosen), 
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)
    
def calculateProfileParametersVoigt(TRANS, CALC_INFO=None, exclude=None):
    """
    Get values for abstract profile parameters for Voigt profile.
    """
    envdep_presets = [('Voigt', 'default')]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('GammaD', calculate_parameter_GammaD), 
        ('Gamma0', calculate_parameter_Gamma0), 
        ('Delta0', calculate_parameter_Delta0), 
        ('YRosen', calculate_parameter_YRosen), 
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)

def calculateProfileParametersSDVoigt(TRANS, CALC_INFO=None, exclude=None):
    """
    Get values for abstract profile parameters for SDVoigt profile.
    """
    envdep_presets = [
        ('SDVoigt', 'default'), 
        ('SDVoigt', 'dimensionless'), 
        ('Voigt', 'default')
        ]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('GammaD', calculate_parameter_GammaD), 
        ('Gamma0', calculate_parameter_Gamma0), 
        ('Delta0', calculate_parameter_Delta0), 
        ('Gamma2', calculate_parameter_Gamma2), 
        ('Delta2', calculate_parameter_Delta2), 
        ('YRosen', calculate_parameter_YRosen),        
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)

def calculateProfileParametersHT(TRANS, CALC_INFO=None, exclude=None):
    """
    Get values for abstract profile parameters for HT profile.
    """
    envdep_presets = [
        ('HT', 'multitemp'), 
        ('HT', 'default'), 
        ('Voigt', 'default')
        ]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('GammaD', calculate_parameter_GammaD), 
        ('Gamma0', calculate_parameter_Gamma0), 
        ('Delta0', calculate_parameter_Delta0), 
        ('Gamma2', calculate_parameter_Gamma2), 
        ('Delta2', calculate_parameter_Delta2), 
        ('Eta', calculate_parameter_Eta), 
        ('NuVC', calculate_parameter_NuVC), 
        ('YRosen', calculate_parameter_YRosen), 
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)
    
def calculateProfileParametersFullPriority(TRANS, CALC_INFO=None, exclude=None):
    """
    Get the Line context on the input, and return the dictionary with the "abstract" parameters.
    """
    envdep_presets = [
        ('HT', 'multitemp'), 
        ('HT', 'default'), 
        ('SDVoigt', 'default'), 
        ('Voigt', 'default')
        ]
    parameters = [
        ('Nu', calculate_parameter_Nu), 
        ('Sw', calculate_parameter_Sw), 
        ('GammaD', calculate_parameter_GammaD), 
        ('Gamma0', calculate_parameter_Gamma0), 
        ('Delta0', calculate_parameter_Delta0), 
        ('Gamma2', calculate_parameter_Gamma2), 
        ('Delta2', calculate_parameter_Delta2), 
        ('Eta', calculate_parameter_Eta), 
        ('NuVC', calculate_parameter_NuVC), 
        ('YRosen', calculate_parameter_YRosen), 
    ]
    return calculateProfileParameters(envdep_presets, parameters, CALC_INFO=CALC_INFO, TRANS=TRANS, exclude=exclude)

VARIABLES['abscoef_debug'] = True

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ENVIRONMENT DEPENDENCES (GENERIC)

def environGetArguments(abstract_parnames, lookup_cases, 
        aux_args, TRANS):
    """
    Get the environment-dependent parameter names, along with auxiliary arguments 
    to use in the environment dependence function.
    INPUT:
        abstract_args: tuple containing the names of the abstract parameters 
                       for the environment dependence
        lookup_cases: list of dicts, each of which corresponds to the
                       mapping of abstract parname to particular value in the database.
                       The search goes accordingly to the 
                       order of the cases in the list, of none found then
                       the Exception is raised.
    OUTPUT:
        INFO: database names for some of the "abstract" arguments
                  for the environment dependence
        ARGS: values for the "abstract" arguments
    """    
    params_not_found = []
    
    for CASE in lookup_cases:
        
        casename = CASE['__case__']
    
        ARGS = aux_args
        INFO = {}
        
        flag = False
        for argname_abstract in set(CASE.keys())-set(['__case__']):
           
            argname_database = CASE[argname_abstract]['name']
            
            try:
                if argname_database not in TRANS or TRANS[argname_database] is np.ma.core.MaskedConstant:
                    if 'default' in CASE[argname_abstract]:
                        source = '<default>'
                        value = CASE[argname_abstract]['default']
                    else:
                        raise KeyError                    
                else:
                    source = argname_database
                    value = TRANS[argname_database]
                ARGS[argname_abstract] = value                   
                if VARIABLES['abscoef_debug']: 
                    INFO[argname_abstract]={'case':casename}
                    INFO[argname_abstract]['source'] = source
                    INFO[argname_abstract]['value'] = value
            except KeyError:                
                params_not_found.append(argname_database)
                flag = True
                
        if not flag: 
            return INFO, ARGS
        
    raise Exception('not found in DB: %s'%params_not_found)

# STANDARD ENVIRONMENT DEPENDENCE FUNCTIONS    
    
def environDependenceFn_PowerLaw(Par_ref, TempRatioPower, T, T_ref, p, p_ref):
    """
    Standard single power law environment dependence.
    """
    return Par_ref * ( T_ref/T )**TempRatioPower * p/p_ref
    
def environDependenceFn_LinearLaw(Par_ref, Coef, T, T_ref, p, p_ref):
    """
    Standard linear law environment dependence.    
    """ 
    return ( Par_ref + Coef*(T-T_ref) ) * p/p_ref
    
# ENVIRONMENT DEPENDENCES FOR LORENTZ PROFILE    
    
# Gamma0 =>
    
def environGetArguments_Lorentz_Gamma0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Gamma0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'Lorentz 1', 
            'Par_ref':{
                'name': 'gamma_%s'%broadener,                
            }, 
            'TempRatioPower':{
                'name': 'n_%s'%broadener, 
            }, 
        }, 
        {
            '__case__': 'Lorentz 2', 
            'Par_ref':{
                'name': 'gamma_%s'%broadener, 
            }, 
            'TempRatioPower':{
                'name': 'n_air', 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# Delta0 =>    
    
def environGetArguments_Lorentz_Delta0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Delta0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'Lorentz 1', 
            'Par_ref':{
                'name': 'delta_%s'%broadener, 
                'default': 0, 
            }, 
            'Coef':{
                'name': 'deltap_%s'%broadener, 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

# YRosen =>

def environGetArguments_Lorentz_YRosen_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for YRosen.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'Lorentz 1', 
            'Par_ref':{
                'name': 'y_%s'%broadener,  
                'default': 0, 
            }, 
            'TempRatioPower':{
                'name': 'n_y_%s'%broadener, 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
        
# ENVIRONMENT DEPENDENCES FOR VOIGT PROFILE
    
# ... Gamma0, Delta0, and YRosen are the same as for Lorentz profile

# ENVIRONMENT DEPENDENCES FOR SDVOIGT PROFILE

# Gamma0 =>
    
def environGetArguments_SDVoigt_Gamma0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Gamma0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'SDVoigt 1', 
            'Par_ref':{
                'name': 'gamma_SDV_0_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_SDV_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# Delta0 =>
        
def environGetArguments_SDVoigt_Delta0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Delta0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'SDVoigt 1', 
            'Par_ref':{
                'name': 'delta_SDV_0_%s_%d'%(broadener, T_ref), 
            }, 
            'Coef':{
                'name': 'deltap_SDV_%s_%d'%(broadener, T_ref), 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

# Gamma2 =>
    
def environGetArguments_SDVoigt_Gamma2_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Gamma2.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'SDVoigt 1', 
            'Par_ref':{
                'name': 'gamma_SDV_2_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_gamma_SDV_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

def environGetArguments_SDVoigt_Gamma2_dimensionless(broadener, TRANS): # avoid this;
    """
    Argument selector for the environment dependence function for Gamma2.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['gamma0', 'sd', 'n']
    lookup_cases = [
        {
            '__case__': 'SDVoigt 1', 
            'gamma0':{
                'name': 'gamma_%s'%broadener, 
            }, 
            'sd':{
                'name': 'SD_%s'%broadener, 
            }, 
            'n':{
                'name': 'n_SD_%s'%broadener, 
                'default': 0, 
            }, 
        }, 
        {
            '__case__': 'SDVoigt 2', 
            'gamma0':{
                'name': 'gamma_%s'%broadener, 
            }, 
            'sd':{
                'name': 'SD_%s'%broadener, 
            }, 
            'n':{
                'name': 'n_SD_air', 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
        
# Delta2 =>    
    
def environGetArguments_SDVoigt_Delta2_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Delta0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'SDVoigt 1', 
            'Par_ref':{
                'name': 'delta_SDV_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
            'Coef':{
                'name': 'deltap_SDV_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

# YRosen =>

def environGetArguments_SDVoigt_YRosen_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for YRosen.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'Lorentz 1', 
            'Par_ref':{
                'name': 'Y_SDV_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
            'TempRatioPower':{
                'name': 'n_Y_SDV_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# ENVIRONMENT DEPENDENCES FOR HT PROFILE

# Gamma0 =>

def get_T_ref_for_HT_multitemp(T):
    """
    Get the actual reference temperature for the multitemp HT preset.
    """
    TRanges = [(0, 100), (100, 200), (200, 400), (400, float('inf'))]
    Trefs = [50., 150., 296., 700.]
    for TRange, TrefHT in zip(TRanges, Trefs):
        if T >= TRange[0] and T < TRange[1]:
            break
    return TrefHT
    
def environGetArguments_HT_Gamma0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Gamma0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'gamma_HT_0_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_HT_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
def environGetArguments_HT_Gamma0_multitemp(broadener, TRANS): # CUSTOM MULTITEMP PRESET
    """
    Argument selector for the environment dependence function for Gamma0.
    Search parameters for non-standard "Multi-temperature" environment dependence
    used in HITRAN for H2 molecule, as described in Wcislo et al., JQSRT 2016.
    """
    T_ref = get_T_ref_for_HT_multitemp(TRANS['T']); p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'gamma_HT_0_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_HT_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# Delta0 =>
        
def environGetArguments_HT_Delta0_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Delta0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'delta_HT_0_%s_%d'%(broadener, T_ref), 
            }, 
            'Coef':{
                'name': 'deltap_HT_%s_%d'%(broadener, T_ref), 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

def environGetArguments_HT_Delta0_multitemp(broadener, TRANS): # CUSTOM MULTITEMP PRESET
    """
    Argument selector for the environment dependence function for Delta0.
    Search parameters for non-standard "Multi-temperature" environment dependence
    used in HITRAN for H2 molecule, as described in Wcislo et al., JQSRT 2016.
    """
    T_ref = get_T_ref_for_HT_multitemp(TRANS['T']); p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'delta_HT_0_%s_%d'%(broadener, T_ref), 
            }, 
            'Coef':{
                'name': 'deltap_HT_%s_%d'%(broadener, T_ref), 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# Gamma2 =>
    
def environGetArguments_HT_Gamma2_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Gamma2.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'gamma_HT_2_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_gamma_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

def environGetArguments_HT_Gamma2_multitemp(broadener, TRANS): # CUSTOM MULTITEMP PRESET
    """
    Argument selector for the environment dependence function for Gamma2.
    Search parameters for non-standard "Multi-temperature" environment dependence
    used in HITRAN for H2 molecule, as described in Wcislo et al., JQSRT 2016.
    """
    T_ref = get_T_ref_for_HT_multitemp(TRANS['T']); p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'gamma_HT_2_%s_%d'%(broadener, T_ref), 
            }, 
            'TempRatioPower':{
                'name': 'n_gamma_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# Delta2 =>    
    
def environGetArguments_HT_Delta2_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for Delta0.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'delta_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
            'Coef':{
                'name': 'deltap_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS

def environGetArguments_HT_Delta2_multitemp(broadener, TRANS): # CUSTOM MULTITEMP PRESET
    """
    Argument selector for the environment dependence function for Delta0.
    Search parameters for non-standard "Multi-temperature" environment dependence
    used in HITRAN for H2 molecule, as described in Wcislo et al., JQSRT 2016.
    """
    T_ref = get_T_ref_for_HT_multitemp(TRANS['T']); p_ref = 1.0
    abstract_args = ['Par_ref', 'Coef']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'delta_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
            'Coef':{
                'name': 'deltap_HT_2_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# NuVC =>    
    
def environGetArguments_HT_NuVC_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for NuVC.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'HT 1', 
            'Par_ref':{
                'name': 'nu_HT_%s'%broadener, 
            }, 
            'TempRatioPower':{
                'name': 'kappa_HT_%s'%broadener, 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# YRosen =>

def environGetArguments_HT_YRosen_default(broadener, TRANS):
    """
    Argument selector for the environment dependence function for YRosen.
    """
    T_ref = 296.0; p_ref = 1.0
    abstract_args = ['Par_ref', 'TempRatioPower']
    lookup_cases = [
        {
            '__case__': 'Lorentz 1', 
            'Par_ref':{
                'name': 'Y_HT_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
            'TempRatioPower':{
                'name': 'n_Y_HT_%s_%d'%(broadener, T_ref), 
                'default': 0, 
            }, 
        }, 
    ]
    aux_args = {'T':TRANS['T'], 'T_ref':T_ref, 'p':TRANS['p'], 'p_ref':p_ref}
    CALC_INFO, ARGS = environGetArguments(abstract_args, 
        lookup_cases, aux_args, TRANS)
    return CALC_INFO, ARGS
    
# //////////////////////////////////////////////////////////////////////
# REGISTRY FOR ENVIRONMENT DEPENDENCES FOR PRESSURE-INDUCED PARAMETERS 
PRESSURE_INDUCED_ENVDEP = {

    'Lorentz': {
        'Gamma0': { # name of the abstract parameter
            'default': { # name of the preset
                'getargs': environGetArguments_Lorentz_Gamma0_default, # convert abstract environment dependence arguments to the real database parameters
                'depfunc': environDependenceFn_PowerLaw, # calculate the environment-dependent parameter            
            }, 
        }, 
        'Delta0': {    
            'default': {
                'getargs': environGetArguments_Lorentz_Delta0_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'YRosen': {    
            'default': {
                'getargs': environGetArguments_Lorentz_YRosen_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
    }, 
    
    'Voigt': {
        'Gamma0': {
            'default': {
                'getargs': environGetArguments_Lorentz_Gamma0_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
        'Delta0': {    
            'default': {
                'getargs': environGetArguments_Lorentz_Delta0_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'YRosen': {    
            'default': {
                'getargs': environGetArguments_Lorentz_YRosen_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
    }, 

    'SDVoigt': {
        'Gamma0': {
            'default': {
                'getargs': environGetArguments_SDVoigt_Gamma0_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
        'Delta0': {    
            'default': {
                'getargs': environGetArguments_SDVoigt_Delta0_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'Gamma2': {
            'default': {
                'getargs': environGetArguments_SDVoigt_Gamma2_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
            'dimensionless': {
                'getargs': environGetArguments_SDVoigt_Gamma2_dimensionless, 
                'depfunc': lambda gamma0, sd, n, T, T_ref, p, p_ref: environDependenceFn_PowerLaw(gamma0*sd, n, T, T_ref, p, p_ref), 
            }, 
        }, 
        'Delta2': {    
            'default': {
                'getargs': environGetArguments_SDVoigt_Delta2_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'YRosen': {    
            'default': {
                'getargs': environGetArguments_SDVoigt_YRosen_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
    }, 
    
    'HT': {
        'Gamma0': {
            'default': {
                'getargs': environGetArguments_HT_Gamma0_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
            'multitemp': {
                'getargs': environGetArguments_HT_Gamma0_multitemp, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
        'Delta0': {    
            'default': {
                'getargs': environGetArguments_HT_Delta0_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
            'multitemp': {
                'getargs': environGetArguments_HT_Delta0_multitemp, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'Gamma2': {
            'default': {
                'getargs': environGetArguments_HT_Gamma2_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
            'multitemp': {
                'getargs': environGetArguments_HT_Gamma2_multitemp, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
        'Delta2': {
            'default': {
                'getargs': environGetArguments_HT_Delta2_default, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
            'multitemp': {
                'getargs': environGetArguments_HT_Delta2_multitemp, 
                'depfunc': environDependenceFn_LinearLaw, 
            }, 
        }, 
        'NuVC': {    
            'default': {
                'getargs': environGetArguments_HT_NuVC_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
        'YRosen': {    
            'default': {
                'getargs': environGetArguments_HT_YRosen_default, 
                'depfunc': environDependenceFn_PowerLaw, 
            }, 
        }, 
    }, 
    
}

# ////////////////////////////////////////////

# OLD TEMPERATURE AND PRESSURE DEPENDENCES MOSTLY FOR BACKWARDS COMPATIBILITY

# temperature dependence for intensities (HITRAN)
def EnvironmentDependency_Intensity(LineIntensityRef, T, Tref, SigmaT, SigmaTref, 
                                    LowerStateEnergy, LineCenter):
    const = FloatType64(1.4388028496642257)
    ch = exp(-const*LowerStateEnergy/T)*(1-exp(-const*LineCenter/T))
    zn = exp(-const*LowerStateEnergy/Tref)*(1-exp(-const*LineCenter/Tref))
    LineIntensity = LineIntensityRef*SigmaTref/SigmaT*ch/zn
    return LineIntensity

# environmental dependence for GammaD (HTP, Voigt)
def EnvironmentDependency_GammaD(GammaD_ref, T, Tref):
    # Doppler parameters do not depend on pressure!
    return GammaD_ref*sqrt(T/Tref)

# environmental dependence for Gamma0 (HTP, Voigt)
def EnvironmentDependency_Gamma0(Gamma0_ref, T, Tref, p, pref, TempRatioPower):
    return Gamma0_ref*p/pref*(Tref/T)**TempRatioPower

# environmental dependence for Gamma2 (HTP)
def EnvironmentDependency_Gamma2(Gamma2_ref, T, Tref, p, pref, TempRatioPower):
    return Gamma2_ref*p/pref*(Tref/T)**TempRatioPower
    #return Gamma2_ref*p/pref

# environmental dependence for Delta0 (HTP)
def EnvironmentDependency_Delta0(Delta0_ref, Deltap, T, Tref, p, pref):
    return (Delta0_ref + Deltap*(T-Tref))*p/pref

# environmental dependence for Delta2 (HTP)
def EnvironmentDependency_Delta2(Delta2_ref, T, Tref, p, pref, TempRatioPower):
    return Delta2_ref*p/pref*(Tref/T)**TempRatioPower
    #return Delta2_ref*p/pref

# environmental dependence for nuVC (HTP)
def EnvironmentDependency_nuVC(nuVC_ref, Kappa, T, Tref, p, pref):
    return nuVC_ref*(Tref/T)**Kappa*p/pref

# environmental dependence for nuVC (HTP)
def EnvironmentDependency_Eta(EtaDB, Gamma0, Shift0, Diluent, C):  # C=>CONTEXT
    Eta_Numer = 0
    for species in Diluent:
        abun = Diluent[species]
        Gamma0T = C['Gamma0T_%s'%species]
        Shift0T = C['Shift0T_%s'%species]
        Eta_Numer += EtaDB*abun*(Gamma0T+1j*Shift0T)
    Eta = Eta_Numer / (Gamma0 + 1j*Shift0)    
    return Eta
    
# ------------------------------- /PARAMETER DEPENDENCIES --------------------------------

# ------------------------------- BINGINGS --------------------------------

# default parameter bindings
DefaultParameterBindings = {}

# default temperature dependencies
DefaultEnvironmentDependencyBindings = {}

# ------------------------------- /BINGINGS --------------------------------

# default values for intensity threshold
DefaultIntensityThreshold = 0. # cm*molec

# default value for omega wing in halfwidths (from center)
DefaultOmegaWingHW = 50. # cm-1    HOTW default


# check and argument for being a tuple or list
# this is connected with a "bug" that in Python
# (val) is not a tuple, but (val, ) is a tuple
def listOfTuples(a):
    if type(a) not in set([list, tuple]):
        a = [a]
    return a


# determine default parameters from those which are passed to absorptionCoefficient_...
def getDefaultValuesForXsect(Components, SourceTables, Environment, OmegaRange, 
                             OmegaStep, OmegaWing, IntensityThreshold, Format):
    if SourceTables[0] == None:
        SourceTables = ['__BUFFER__', ]
    if Environment == None:
        Environment = {'T':296., 'p':1.}
    if Components == [None]:
        CompDict = {}
        for TableName in SourceTables:
            # check table existance
            if TableName not in LOCAL_TABLE_CACHE.keys():
                raise Exception('%s: no such table. Check tableList() for more info.' % TableName)
            mol_ids = LOCAL_TABLE_CACHE[TableName]['data']['molec_id']
            iso_ids = LOCAL_TABLE_CACHE[TableName]['data']['local_iso_id']
            if len(mol_ids) != len(iso_ids):
                raise Exception('Lengths if mol_ids and iso_ids differ!')
            MI_zip = zip(mol_ids, iso_ids)
            MI_zip = set(MI_zip)
            for mol_id, iso_id in MI_zip:
                CompDict[(mol_id, iso_id)] = None
        Components = CompDict.keys()
    if OmegaRange == None:
        omega_min = float('inf')
        omega_max = float('-inf')
        for TableName in SourceTables:
            nu = LOCAL_TABLE_CACHE[TableName]['data']['nu']
            numin = min(nu)
            numax = max(nu)
            if omega_min > numin:
                omega_min = numin
            if omega_max < numax:
                omega_max = numax
        OmegaRange = (omega_min, omega_max)
    if OmegaStep == None:
        OmegaStep = 0.01 # cm-1
    if OmegaWing == None:
        OmegaWing = 0.0 # cm-1
    if not Format:
        Format = '%.12f %e'
    return Components, SourceTables, Environment, OmegaRange, \
           OmegaStep, OmegaWing, IntensityThreshold, Format

ABSCOEF_DOCSTRING_TEMPLATE = \
    """
    INPUT PARAMETERS: 
        Components:  list of tuples [(M, I, D)], where
                        M - HITRAN molecule number, 
                        I - HITRAN isotopologue number, 
                        D - relative abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional)
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional)
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres, 
                        'T' - temperature in Kelvin
                        Default={{'p':1., 'T':296.}}
        WavenumberRange:  wavenumber range to consider.
        WavenumberStep:   wavenumber step to consider. 
        WavenumberWing:   absolute wing for calculating a lineshape (in cm-1) 
        WavenumberWingHW:  relative wing for calculating a lineshape (in halfwidths)
        IntensityThreshold:  threshold for intensities
        Diluent:  specifies broadening mixture composition, e.g. {{'air':0.7, 'self':0.3}}
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts for significant digits in WavenumberStep)
        LineMixingRosen: include 1st order line mixing to calculation
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid with respect to parameters WavenumberRange and WavenumberStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using {profile}.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation.
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average) 
        for a reasonable amount of cpu time. To increase calculation accuracy, 
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        {usage_example}
    ---
    """     
           
def absorptionCoefficient_Generic(Components=None, SourceTables=None, partitionFunction=PYTIPS, 
                                  Environment=None, OmegaRange=None, OmegaStep=None, OmegaWing=None, 
                                  IntensityThreshold=DefaultIntensityThreshold, 
                                  OmegaWingHW=DefaultOmegaWingHW, 
                                  GammaL='gamma_air', HITRAN_units=True, LineShift=True, 
                                  File=None, Format=None, OmegaGrid=None, 
                                  WavenumberRange=None, WavenumberStep=None, WavenumberWing=None, 
                                  WavenumberWingHW=None, WavenumberGrid=None, 
                                  Diluent={}, LineMixingRosen=False, 
                                  profile=None, calcpars=None, exclude=set(), 
                                  DEBUG=None):
                                                              
    # Throw exception if profile or calcpars are empty.
    if profile is None: raise Exception('user must provide the line profile function')
    if calcpars is None: raise Exception('user must provide the function for calculating profile parameters')
        
    if DEBUG is not None: 
        VARIABLES['abscoef_debug'] = True
    else:
        VARIABLES['abscoef_debug'] = False
        
    if not LineMixingRosen: exclude.add('YRosen')
    if not LineShift: exclude.update({'Delta0', 'Delta2'})
    
    # Parameters OmegaRange, OmegaStep, OmegaWing, OmegaWingHW, and OmegaGrid
    # are deprecated and given for backward compatibility with the older versions.
    if WavenumberRange is not None:  OmegaRange=WavenumberRange
    if WavenumberStep is not None:   OmegaStep=WavenumberStep
    if WavenumberWing is not None:   OmegaWing=WavenumberWing
    if WavenumberWingHW is not None: OmegaWingHW=WavenumberWingHW
    if WavenumberGrid is not None:   OmegaGrid=WavenumberGrid

    # "bug" with 1-element list
    Components = listOfTuples(Components)
    SourceTables = listOfTuples(SourceTables)
    
    # determine final input values
    Components, SourceTables, Environment, OmegaRange, OmegaStep, OmegaWing, \
    IntensityThreshold, Format = \
       getDefaultValuesForXsect(Components, SourceTables, Environment, OmegaRange, 
                                OmegaStep, OmegaWing, IntensityThreshold, Format)
    
    # warn user about too large omega step
    if OmegaStep>0.005 and profile is PROFILE_DOPPLER: 
        warn('Big wavenumber step: possible accuracy decline')
    elif OmegaStep>0.1: 
        warn('Big wavenumber step: possible accuracy decline')

    # get uniform linespace for cross-section
    #number_of_points = (OmegaRange[1]-OmegaRange[0])/OmegaStep + 1
    #Omegas = linspace(OmegaRange[0], OmegaRange[1], number_of_points)
    if OmegaGrid is not None:
        Omegas = npsort(OmegaGrid)
    else:
        #Omegas = arange(OmegaRange[0], OmegaRange[1], OmegaStep)
        Omegas = arange_(OmegaRange[0], OmegaRange[1], OmegaStep) # fix
    number_of_points = len(Omegas)
    Xsect = zeros(number_of_points)
       
    # reference temperature and pressure
    T_ref_default = FloatType64(296.) # K
    p_ref_default = FloatType64(1.) # atm
    
    # actual temperature and pressure
    T = Environment['T'] # K
    p = Environment['p'] # atm
       
    # create dictionary from Components
    ABUNDANCES = {}
    NATURAL_ABUNDANCES = {}
    for Component in Components:
        M = Component[0]
        I = Component[1]
        if len(Component) >= 3:
            ni = Component[2]
        else:
            try:
                ni = ISO[(M, I)][ISO_INDEX['abundance']]
            except KeyError:
                raise Exception('cannot find component M, I = %d, %d.' % (M, I))
        ABUNDANCES[(M, I)] = ni
        NATURAL_ABUNDANCES[(M, I)] = ISO[(M, I)][ISO_INDEX['abundance']]
        
    # pre-calculation of volume concentration
    if HITRAN_units:
        factor = FloatType64(1.0)
    else:
        factor = volumeConcentration(p, T)
        
    # setup the Diluent variable
    GammaL = GammaL.lower()
    if not Diluent:
        if GammaL == 'gamma_air':
            Diluent = {'air':1.}
        elif GammaL == 'gamma_self':
            Diluent = {'self':1.}
        else:
            raise Exception('Unknown GammaL value: %s' % GammaL)
        
    # Simple check
    print(Diluent)  # Added print statement # CHANGED RJH 23MAR18  # Simple check
    for key in Diluent:
        val = Diluent[key]
        if val < 0 or val > 1: # if val < 0 and val > 1:# CHANGED RJH 23MAR18
            raise Exception('Diluent fraction must be in [0, 1]')
            
    # ================= HERE THE GENERIC PART STARTS =====================

    t = time()
    
    CALC_INFO_TOTAL = []
    
    # SourceTables contain multiple tables
    for TableName in SourceTables:
    
        # exclude parameters not involved in calculation
        DATA_DICT = LOCAL_TABLE_CACHE[TableName]['data']
        parnames_exclude = ['a', 'global_upper_quanta', 'global_lower_quanta', 
            'local_upper_quanta', 'local_lower_quanta', 'ierr', 'iref', 'line_mixing_flag'] 
        parnames = set(DATA_DICT)-set(parnames_exclude)
        
        nlines = len(DATA_DICT['nu'])

        for RowID in range(nlines):
                            
            # create the transition object
            TRANS = CaselessDict({parname:DATA_DICT[parname][RowID] for parname in parnames}) # CORRECTLY HANDLES DIFFERENT SPELLING OF PARNAMES
            TRANS['T'] = T
            TRANS['p'] = p
            TRANS['T_ref'] = T_ref_default
            TRANS['p_ref'] = p_ref_default
            TRANS['Diluent'] = Diluent
            TRANS['Abundances'] = ABUNDANCES
            
            # filter by molecule and isotopologue
            if (TRANS['molec_id'], TRANS['local_iso_id']) not in ABUNDANCES: continue
                
            #   FILTER by LineIntensity: compare it with IntencityThreshold
            TRANS['SigmaT']     = partitionFunction(TRANS['molec_id'], TRANS['local_iso_id'], TRANS['T'])
            TRANS['SigmaT_ref'] = partitionFunction(TRANS['molec_id'], TRANS['local_iso_id'], TRANS['T_ref'])
            LineIntensity = calculate_parameter_Sw(None, TRANS)
            if LineIntensity < IntensityThreshold: continue

            # calculate profile parameters 
            if VARIABLES['abscoef_debug']:
                CALC_INFO = {}
            else:
                CALC_INFO = None                
            PARAMETERS = calcpars(TRANS=TRANS, CALC_INFO=CALC_INFO, exclude=exclude)
            
            # get final wing of the line according to max(Gamma0, GammaD), OmegaWingHW and OmegaWing
            try:
                GammaD = PARAMETERS['GammaD']
            except KeyError:
                GammaD = 0
            try:
                Gamma0 = PARAMETERS['Gamma0']
            except KeyError:
                Gamma0 = 0
            GammaMax = max(Gamma0, GammaD)
            if GammaMax == 0 and OmegaWingHW == 0:
                OmegaWing = 10.0 # 10 cm-1 default in case if Gamma0 and GammaD are missing
                warn('Gamma0 and GammaD are missing; setting OmegaWing to %f cm-1'%OmegaWing)
            OmegaWingF = max(OmegaWing, OmegaWingHW*GammaMax)
            
            # calculate profile on a grid            
            BoundIndexLower = bisect(Omegas, TRANS['nu']-OmegaWingF)
            BoundIndexUpper = bisect(Omegas, TRANS['nu']+OmegaWingF)
            PARAMETERS['WnGrid'] = Omegas[BoundIndexLower:BoundIndexUpper]
            lineshape_vals = profile(**PARAMETERS)
            Xsect[BoundIndexLower:BoundIndexUpper] += factor * lineshape_vals
                   
            # append debug information for the abscoef routine                
            if VARIABLES['abscoef_debug']: DEBUG.append(CALC_INFO)
        
    print('%f seconds elapsed for abscoef; nlines = %d'%(time()-t, nlines))
    
    if File: save_to_file(File, Format, Omegas, Xsect)
    return Omegas, Xsect

def absorptionCoefficient_Priority(*args, **kwargs):
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                         profile=PROFILE_HT, 
                                         calcpars=calculateProfileParametersFullPriority)    
    
def absorptionCoefficient_HT(*args, **kwargs):
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                         profile=PROFILE_HT, 
                                         calcpars=calculateProfileParametersHT)    
                                   
def absorptionCoefficient_SDVoigt(*args, **kwargs):
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                          profile=PROFILE_SDVOIGT, 
                                          calcpars=calculateProfileParametersSDVoigt)
        
def absorptionCoefficient_Voigt(*args, **kwargs):
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                          profile=PROFILE_VOIGT, 
                                          calcpars=calculateProfileParametersVoigt)

def absorptionCoefficient_Lorentz(*args, **kwargs):
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                         profile=PROFILE_LORENTZ, 
                                         calcpars=calculateProfileParametersLorentz)
     
def absorptionCoefficient_Doppler(*args, **kwargs):   
    return absorptionCoefficient_Generic(*args, **kwargs, 
                                         profile=PROFILE_DOPPLER, 
                                         calcpars=calculateProfileParametersDoppler)
    
absorptionCoefficient_Generic.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Generic', 
    usage_example="""
        nu, coef = absorptionCoefficient_Generic(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1}, 
                                              profile=PROFILE_VOIGT, 
                                              calcpars=calcProfileParametersVoigt, 
                                              DEBUG=None, 
                                              )
    """
)

absorptionCoefficient_Priority.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Priority', 
    usage_example="""
        nu, coef = absorptionCoefficient_Priority(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)
                                   
absorptionCoefficient_HT.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='HT', 
    usage_example="""
        nu, coef = absorptionCoefficient_HT(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)

absorptionCoefficient_SDVoigt.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='SDVoigt', 
    usage_example="""
        nu, coef = absorptionCoefficient_SDVoigt(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)

absorptionCoefficient_Voigt.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Voigt', 
    usage_example="""
        nu, coef = absorptionCoefficient_Voigt(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)

absorptionCoefficient_Lorentz.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Lorentz', 
    usage_example="""
        nu, coef = absorptionCoefficient_Lorentz(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)

absorptionCoefficient_Doppler.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Doppler', 
    usage_example="""
        nu, coef = absorptionCoefficient_Doppler(((2, 1), ), 'co2', WavenumberStep=0.01, 
                                              HITRAN_units=False, Diluent={'air':1})
    """
)    
    
# save numpy arrays to file
# arrays must have same dimensions
def save_to_file(fname, fformat, *arg):
    f = open(fname, 'w')
    for i in range(len(arg[0])):
        argline = []
        for j in range(len(arg)):
            argline.append(arg[j][i])
        f.write((fformat+'\n') % tuple(argline))
    f.close()
    
# ---------------------------------------------------------------------------
# SHORTCUTS AND ALIASES FOR ABSORPTION COEFFICIENTS
# ---------------------------------------------------------------------------

absorptionCoefficient_Gauss = absorptionCoefficient_Doppler

def abscoef_HT(table=None, step=None, grid=None, env={'T':296., 'p':1.}, file=None):
    return absorptionCoefficient_HT(SourceTables=table, OmegaStep=step, OmegaGrid=grid, Environment=env, File=file)

def abscoef_Voigt(table=None, step=None, grid=None, env={'T':296., 'p':1.}, file=None):
    return absorptionCoefficient_Voigt(SourceTables=table, OmegaStep=step, OmegaGrid=grid, Environment=env, File=file)
    
def abscoef_Lorentz(table=None, step=None, grid=None, env={'T':296., 'p':1.}, file=None):
    return absorptionCoefficient_Lorentz(SourceTables=table, OmegaStep=step, OmegaGrid=grid, Environment=env, File=file)

def abscoef_Doppler(table=None, step=None, grid=None, env={'T':296., 'p':1.}, file=None):
    return absorptionCoefficient_Doppler(SourceTables=table, OmegaStep=step, OmegaGrid=grid, Environment=env, File=file)

abscoef_Gauss = abscoef_Doppler
    
def abscoef(table=None, step=None, grid=None, env={'T':296., 'p':1.}, file=None): # default
    return absorptionCoefficient_Lorentz(SourceTables=table, OmegaStep=step, OmegaGrid=grid, Environment=env, File=file)
    
# ---------------------------------------------------------------------------
    
def transmittanceSpectrum(Omegas, AbsorptionCoefficient, Environment={'l':100.}, 
                          File=None, Format='%e %e', Wavenumber=None):
    """
    INPUT PARAMETERS: 
        Wavenumber/Omegas:   wavenumber grid                    (required)
        AbsorptionCoefficient:  absorption coefficient on grid  (required)
        Environment:  dictionary containing path length in cm.
                      Default={'l':100.}
        File:         name of the output file                 (optional) 
        Format: c format used in file output, default '%e %e' (optional)
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid
        Xsect:  transmittance spectrum calculated on the grid
    ---
    DESCRIPTION:
        Calculate a transmittance spectrum (dimensionless) based
        on previously calculated absorption coefficient.
        Transmittance spectrum is calculated at an arbitrary
        optical path length 'l' (1 m by default)
    ---
    EXAMPLE OF USAGE:
        nu, trans = transmittanceSpectrum(nu, coef)
    ---
    """
    # compatibility with older versions
    if Wavenumber: Omegas=Wavenumber
    l = Environment['l']
    Xsect = exp(-AbsorptionCoefficient*l)
    if File: save_to_file(File, Format, Omegas, Xsect)
    return Omegas, Xsect

def absorptionSpectrum(Omegas, AbsorptionCoefficient, Environment={'l':100.}, 
                       File=None, Format='%e %e', Wavenumber=None):
    """
    INPUT PARAMETERS: 
        Wavenumber/Omegas:   wavenumber grid                    (required)
        AbsorptionCoefficient:  absorption coefficient on grid  (required)
        Environment:  dictionary containing path length in cm.
                      Default={'l':100.}
        File:         name of the output file                 (optional) 
        Format: c format used in file output, default '%e %e' (optional)
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid
        Xsect:  absorption spectrum calculated on the grid
    ---
    DESCRIPTION:
        Calculate an absorption spectrum (dimensionless) based
        on previously calculated absorption coefficient.
        Absorption spectrum is calculated at an arbitrary
        optical path length 'l' (1 m by default)
    ---
    EXAMPLE OF USAGE:
        nu, absorp = absorptionSpectrum(nu, coef)
    ---
    """
    # compatibility with older versions
    if Wavenumber: Omegas=Wavenumber
    l = Environment['l']
    Xsect = 1-exp(-AbsorptionCoefficient*l)
    if File: save_to_file(File, Format, Omegas, Xsect)
    return Omegas, Xsect

def radianceSpectrum(Omegas, AbsorptionCoefficient, Environment={'l':100., 'T':296.}, 
                     File=None, Format='%e %e', Wavenumber=None):
    """
    INPUT PARAMETERS: 
        Wavenumber/Omegas:   wavenumber grid                   (required)
        AbsorptionCoefficient:  absorption coefficient on grid (required)
        Environment:  dictionary containing path length in cm.
                      and temperature in Kelvin.
                      Default={'l':100., 'T':296.}
        File:         name of the output file                 (optional) 
        Format: c format used in file output, default '%e %e' (optional)
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid
        Xsect:  radiance spectrum calculated on the grid
    ---
    DESCRIPTION:
        Calculate a radiance spectrum (in W/sr/cm^2/cm-1) based
        on previously calculated absorption coefficient.
        Radiance spectrum is calculated at an arbitrary
        optical path length 'l' (1 m by default) and 
        temperature 'T' (296 K by default). For obtaining a
        physically meaningful result 'T' must be the same 
        as a temperature which was used in absorption coefficient.
    ---
    EXAMPLE OF USAGE:
        nu, radi = radianceSpectrum(nu, coef)
    ---
    """
    # compatibility with older versions
    if Wavenumber: Omegas=Wavenumber
    l = Environment['l']
    T = Environment['T']
    Alw = 1-exp(-AbsorptionCoefficient*l)
    LBBTw = 2*hh*cc**2*Omegas**3 / (exp(hh*cc*Omegas/(cBolts*T)) - 1) * 1.0E-7
    Xsect = Alw*LBBTw # W/sr/cm**2/cm**-1
    if File: save_to_file(File, Format, Omegas, Xsect)
    return Omegas, Xsect


# GET X, Y FOR FINE PLOTTING OF A STICK SPECTRUM
def getStickXY(TableName):
    """
    Get X and Y for fine plotting of a stick spectrum.
    Usage: X, Y = getStickXY(TableName).
    """
    cent, intens = getColumns(TableName, ('nu', 'sw'))
    n = len(cent)
    cent_ = zeros(n*3)
    intens_ = zeros(n*3)
    for i in range(n):
        intens_[3*i] = 0
        intens_[3*i+1] = intens[i]
        intens_[3*i+2] = 0
        cent_[(3*i):(3*i+3)] = cent[i]
    return cent_, intens_
# /GET X, Y FOR FINE PLOTTING OF A STICK SPECTRUM


# LOW-RES SPECTRA (CONVOLUTION WITH APPARATUS FUNCTION)

# /LOW-RES SPECTRA (CONVOLUTION WITH APPARATUS FUNCTION)

# /----------------------------------------------------------------------------


# ------------------  HITRAN-ON-THE-WEB COMPATIBILITY -------------------------

def read_hotw(filename):
    """
    Read cross-section file fetched from HITRAN-on-the-Web.
    The format of the file line must be as follows: 
      nu, coef
    Other lines are omitted.
    """
    import sys
    f = open(filename, 'r')
    nu = []
    coef = []
    for line in f:
        pars = line.split()
        try:
            nu.append(float(pars[0]))
            coef.append(float(pars[1]))
        except:
            if False:
                print(sys.exc_info())
            else:
                pass    
    return array(nu), array(coef)

# alias for read_hotw for backwards compatibility
read_xsect = read_hotw
    
# /----------------------------------------------------------------------------

# ------------------  SPECTRAL CONVOLUTION -------------------------

# rectangular slit function
def SLIT_RECTANGULAR(x, g):
    """
    Instrumental (slit) function.
    B(x) = 1/γ , if |x| ≤ γ/2 & B(x) = 0, if |x| > γ/2, 
    where γ is a slit width or the instrumental resolution.
    """
    index_inner = abs(x) <= g/2
    index_outer = ~index_inner
    y = zeros(len(x))
    y[index_inner] = 1/g
    y[index_outer] = 0
    return y

# triangular slit function
def SLIT_TRIANGULAR(x, g):
    """
    Instrumental (slit) function.
    B(x) = 1/γ*(1-|x|/γ), if |x| ≤ γ & B(x) = 0, if |x| > γ, 
    where γ is the line width equal to the half base of the triangle.
    """
    index_inner = abs(x) <= g
    index_outer = ~index_inner
    y = zeros(len(x))
    y[index_inner] = 1/g * (1 - abs(x[index_inner])/g)
    y[index_outer] = 0
    return y

# gaussian slit function
def SLIT_GAUSSIAN(x, g):
    """
    Instrumental (slit) function.
    B(x) = sqrt(ln(2)/pi)/γ*exp(-ln(2)*(x/γ)**2), 
    where γ/2 is a gaussian half-width at half-maximum.
    """
    g /= 2
    return sqrt(log(2))/(sqrt(pi)*g)*exp(-log(2)*(x/g)**2)

# dispersion slit function
def SLIT_DISPERSION(x, g):
    """
    Instrumental (slit) function.
    B(x) = γ/pi/(x**2+γ**2), 
    where γ/2 is a lorentzian half-width at half-maximum.
    """
    g /= 2
    return g/pi/(x**2+g**2)

# cosinus slit function
def SLIT_COSINUS(x, g):
    return (cos(pi/g*x)+1)/(2*g)

# diffraction slit function
def SLIT_DIFFRACTION(x, g):
    """
    Instrumental (slit) function.
    """
    y = zeros(len(x))
    index_zero = x == 0
    index_nonzero = ~index_zero
    dk_ = pi/g
    x_ = dk_*x[index_nonzero]
    w_ = sin(x_)
    r_ = w_**2/x_**2
    y[index_zero] = 1
    y[index_nonzero] = r_/g
    return y

# apparatus function of the ideal Michelson interferometer
def SLIT_MICHELSON(x, g):
    """
    Instrumental (slit) function.
    B(x) = 2/γ*sin(2pi*x/γ)/(2pi*x/γ) if x!=0 else 1, 
    where 1/γ is the maximum optical path difference.
    """
    y = zeros(len(x))
    index_zero = x == 0
    index_nonzero = ~index_zero
    dk_ = 2*pi/g
    x_ = dk_*x[index_nonzero]
    y[index_zero] = 1
    y[index_nonzero] = 2/g*sin(x_)/x_
    return y

# spectral convolution with an apparatus (slit) function
def convolveSpectrum(Omega, CrossSection, Resolution=0.1, AF_wing=10., 
                     SlitFunction=SLIT_RECTANGULAR, Wavenumber=None):
    """
    INPUT PARAMETERS: 
        Wavenumber/Omega:    wavenumber grid                     (required)
        CrossSection:  high-res cross section calculated on grid (required)
        Resolution:    instrumental resolution γ                 (optional)
        AF_wing:       instrumental function wing                (optional)
        SlitFunction:  instrumental function for low-res spectra calculation (optional)
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid
        CrossSection: low-res cross section calculated on grid
        i1: lower index in Omega input
        i2: higher index in Omega input
        slit: slit function calculated over grid [-AF_wing; AF_wing]
                with the step equal to instrumental resolution. 
    ---
    DESCRIPTION:
        Produce a simulation of experimental spectrum via the convolution 
        of a “dry” spectrum with an instrumental function.
        Instrumental function is provided as a parameter and
        is calculated in a grid with the width=AF_wing and step=Resolution.
    ---
    EXAMPLE OF USAGE:
        nu_, radi_, i, j, slit = convolveSpectrum(nu, radi, Resolution=2.0, AF_wing=10.0, 
                                                SlitFunction=SLIT_MICHELSON)
    ---
    """    
    # compatibility with older versions
    if Wavenumber: Omega=Wavenumber
    step = Omega[1]-Omega[0]
    if step>=Resolution: raise Exception('step must be less than resolution')
    #x = arange(-AF_wing, AF_wing+step, step)
    x = arange_(-AF_wing, AF_wing+step, step) # fix
    slit = SlitFunction(x, Resolution)
    slit /= sum(slit)*step # simple normalization
    left_bnd = int(len(slit)/2) # new versions of Numpy don't support float indexing
    right_bnd = len(Omega) - int(len(slit)/2) # new versions of Numpy don't support float indexing
    CrossSectionLowRes = convolve(CrossSection, slit, mode='same')*step
    return Omega[left_bnd:right_bnd], CrossSectionLowRes[left_bnd:right_bnd], left_bnd, right_bnd, slit

# spectral convolution with an apparatus (slit) function
def convolveSpectrumSame(Omega, CrossSection, Resolution=0.1, AF_wing=10., 
                         SlitFunction=SLIT_RECTANGULAR, Wavenumber=None):
    """
    Convolves cross section with a slit function with given parameters.
    """
    # compatibility with older versions
    if Wavenumber: Omega=Wavenumber
    step = Omega[1]-Omega[0]
    if step>=Resolution: raise Exception('step must be less than resolution')
    #x = arange(-AF_wing, AF_wing+step, step)
    x = arange_(-AF_wing, AF_wing+step, step) # fix
    slit = SlitFunction(x, Resolution)
    slit /= sum(slit)*step # simple normalization
    left_bnd = 0
    right_bnd = len(Omega)
    CrossSectionLowRes = convolve(CrossSection, slit, mode='same')*step
    return Omega[left_bnd:right_bnd], CrossSectionLowRes[left_bnd:right_bnd], left_bnd, right_bnd, slit

def convolveSpectrumFull(Omega, CrossSection, Resolution=0.1, AF_wing=10., SlitFunction=SLIT_RECTANGULAR):
    """
    Convolves cross section with a slit function with given parameters.
    """
    step = Omega[1]-Omega[0]
    x = arange(-AF_wing, AF_wing+step, step)
    slit = SlitFunction(x, Resolution)
    print('step=')
    print(step)
    print('x=')
    print(x)
    print('slitfunc=')
    print(SlitFunction)
    CrossSectionLowRes = convolve(CrossSection, slit, mode='full')*step
    return Omega, CrossSectionLowRes, None, None

# ------------------------------------------------------------------

# ------------------  SAVE CALC INFO IN CSV FORMAT -------------------------

def save_abscoef_calc_info(filename, parname, CALC_INFO_LIST, delim=';'):
    """
    This is an attempt to save the CALC_INFO from the 
    new versions of the abscoef function.
    Currently it saves one parameters per call.
    To build your own function that better suits your needs
    please consult the structure of the CALC_INFO elements.
    """
    col = [] # make empty collection
    order = []    
    for CALC_INFO in CALC_INFO_LIST:
        INFO = CALC_INFO[parname]
        item = {}
        item['val'] = INFO['value']
        if 'val' not in order: 
            order.append('val')
        for broadener in INFO['mixture']:
            for argname in INFO['mixture'][broadener]['args']:
                src_name = '%s_%s_src'%(argname, broadener)
                val_name = '%s_%s_val'%(argname, broadener)
                item[src_name] = INFO['mixture'][broadener]['args'][argname]['source']
                item[val_name] = INFO['mixture'][broadener]['args'][argname]['value']
                if src_name not in order:
                    order.append(src_name)
                if val_name not in order:
                    order.append(val_name)
        col.append(item)    
    # Export the result to the CSV file.
    #col.export_csv('test2.py_%s_%s.csv'%(TABLE_NAMETABLE_NAME, parname), order=order)
    with open(filename, 'w') as fout:
        header = ('%s'%delim).join(order)
        fout.write(header+'\n')
        for CALC_INFO in col:
            line = ('%s'%delim).join([str(CALC_INFO.get(pname, '')) for pname in order])
            fout.write(line+'\n')
