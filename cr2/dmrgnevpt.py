#!/usr/bin/env python
import numpy
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
settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR']

b = 2.0
mol = gto.Mole()
mol.verbose = 7
mol.output = 'cr2-%s.out' % b
mol.atom = [
    ['Cr',(  0.000000,  0.000000, -b/2)],
    ['Cr',(  0.000000,  0.000000,  b/2)],
]
file = open('ccpvdz-dk','r')
mol.basis = {'Cr': gto.basis.parse(file.read())}
file.close()

mol.symmetry = True
mol.build()
m = scf.sfx2c1e(scf.RHF(mol))
file = open('dmrgscf18o_mo%s'%b,'r')
mo = pickle.load(file)
file.close()
mc = mcscf.CASCI(m, 20, 28)
mc.mo_coeff = mo
mc.fcisolver = DMRGCI(mol,maxM=500, tol =1e-8 )
mc.fcisolver.outputlevel = 3
mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20, 24, 28, 30, 34]
mc.fcisolver.scheduleMaxMs  = [200, 400, 800, 1200, 2000, 4000, 3000, 2000, 1000, 500]
mc.fcisolver.scheduleTols   = [0.0001, 0.0001, 0.0001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7 ]
mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0]
mc.fcisolver.twodot_to_onedot = 38
mc.fcisolver.maxIter = 50
mc.mo_coeff = mo
mc.casci()
sc_nevpt(mc)
os.system("rm %s/*/Spin*tmp"%settings.BLOCKSCRATCHDIR)
os.system("rm %s/*/C*tmp"%settings.BLOCKSCRATCHDIR)
os.system("rm %s/*/D*tmp"%settings.BLOCKSCRATCHDIR)

