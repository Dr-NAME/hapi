HAPI_VERSION = '1.2.2.1'; __version__ = HAPI_VERSION

# version header
print('HAPI version: %s' % HAPI_VERSION)
print('To get the most up-to-date version please check http://hitran.org/hapi')
print('ATTENTION: Python versions of partition sums from TIPS-2021 are now available in HAPI code')
print('')
print('           MIT license: Copyright 2021 HITRAN team, see more at http://hitran.org. ')
print('')
print('           If you use HAPI in your research or software development,')
print('           please cite it using the following reference:')
print('           R.V. Kochanov, I.E. Gordon, L.S. Rothman, P. Wcislo, C. Hill, J.S. Wilzewski,')
print('           HITRAN Application Programming Interface (HAPI): A comprehensive approach')
print('           to working with spectroscopic data, J. Quant. Spectrosc. Radiat. Transfer 177, 15-30 (2016)')
print('           DOI: 10.1016/j.jqsrt.2016.03.005')
print('')
print('           ATTENTION: This is the core version of the HITRAN Application Programming Interface.')
print('                      For more efficient implementation of the absorption coefficient routine, ')
print('                      as well as for new profiles, parameters and other functional,')
print('                      please consider using HAPI2 extension library.')
print('                      HAPI2 package is available at http://github.com/hitranonline/hapi2')
print('')

from .hapi import *
