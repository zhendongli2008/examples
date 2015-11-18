#!/usr/bin/env python
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import lo
from pyscf.mrpt.nevpt2 import sc_nevpt
from pyscf.dmrgscf.dmrgci import DMRGCI
import pickle
import os
from pyscf.dmrgscf import settings
#settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR']

b = 1.5
mol = gto.Mole()
mol.verbose = 5
mol.output = 'cr2-%3.2f.out' % b
mol.max_memory = 70000
mol.atom = [
    ['Cr',(  0.000000,  0.000000, -b/2)],
    ['Cr',(  0.000000,  0.000000,  b/2)],
]
mol.basis = {'Cr':'cc-pvdz-dk'}
mol.symmetry = True
mol.build()

m = scf.sfx2c1e(scf.RHF(mol))
m.conv_tol = 1e-9
m.chkfile = 'cr2.chk'
m.level_shift_factor = 0.5
m.scf()

dm = m.make_rdm1()
m.level_shift_factor = 0.0
m.scf(dm)
m.analyze()

mc = mcscf.CASSCF(m, 12, 12)
mc.chkfile = 'mc.chk'
cas_occ = {'A1g':2, 'A1u':2,'E1ux':1, 'E1uy':1,'E1gx':1, 'E1gy':1, 'E2ux': 1, 'E2uy': 1, 'E2gx': 1, 'E2gy': 1}
caslst =  mcscf.addons.select_mo_by_irrep(mc,cas_occ)
list.sort(caslst)
mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
mc.mc1step(mo)

sc_nevpt(mc)
