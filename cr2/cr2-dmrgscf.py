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

b = 1.5
mol = gto.Mole()
mol.verbose = 5
mol.output = 'cr2-%3.2f.out' % b
mol.max_memory = 70000
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
m.conv_tol = 1e-9
m.chkfile = 'hf_chk%s'%b
m.level_shift_factor = 0.5
m.scf()
dm = m.make_rdm1()
m.level_shift_factor = 0.0
m.scf(dm)
m.analyze()

mc = mcscf.CASSCF(m, 12, 12)
mc.chkfile = 'mc_chk_12o%s'%b
cas_occ = {'A1g':2, 'A1u':2,'E1ux':1, 'E1uy':1,'E1gx':1, 'E1gy':1, 'E2ux': 1, 'E2uy': 1, 'E2gx': 1, 'E2gy': 1}
caslst =  mcscf.addons.select_mo_by_irrep(mc,cas_occ)
print 'type', type(caslst)
list.sort(caslst)
mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
mc.mc1step(mo)
mo = mc.mo_coeff

mc = mcscf.CASSCF(m, 20, 28)

#######
mc.fcisolver = DMRGCI(mol,maxM=1000, tol =1e-6 )
#mc.ci_update_dep = 4
mc.max_cycle_micro = 6
mc.callback = mc.fcisolver.restart_scheduler_()
#######

mc.chkfile = 'mc_chk_18o%s'%b
cas_occ = {'A1g':4, 'A1u':4,'E1ux':2, 'E1uy':2,'E1gx':2, 'E1gy':2, 'E2ux': 1, 'E2uy': 1, 'E2gx': 1, 'E2gy': 1}
caslst =  mcscf.addons.select_mo_by_irrep(mc,cas_occ,mo)
list.sort(caslst)
mo = mcscf.addons.sort_mo(mc, mo, caslst, 1)
mc.mc1step(mo)

os.system("rm %s/*/Spin*tmp"%settings.BLOCKSCRATCHDIR)
os.system("cp -r %s %s_%s" %(settings.BLOCKSCRATCHDIR,settings.BLOCKSCRATCHDIR,b))
sc_nevpt(mc)
