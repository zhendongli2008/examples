import numpy
import scipy.linalg
from pyscf import tools,gto,scf,dft
import h5py

def sqrtm(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())

#=============================
# DUMP from chkfile to molden
#=============================
fname = 'hs.chk' #femo' #c2h4'
chkfile = fname
outfile = fname+'0.molden'
tools.molden.from_chkfile(outfile, chkfile)

#=============================
# Natural orbitals
# Lowdin basis X=S{-1/2}
# psi = chi * C 
#     = chi' * C'
#     = chi*X*(X{-1}C')
#=============================
mol,mf = scf.chkfile.load_scf(chkfile)
mo_coeff = mf["mo_coeff"]
ova=mol.intor_symmetric("cint1e_ovlp_sph")
nb = mo_coeff.shape[1]
# Check overlap
diff = reduce(numpy.dot,(mo_coeff[0].T,ova,mo_coeff[0])) - numpy.identity(nb)
print numpy.linalg.norm(diff)
diff = reduce(numpy.dot,(mo_coeff[1].T,ova,mo_coeff[1])) - numpy.identity(nb)
print numpy.linalg.norm(diff)
# UHF-alpha/beta
ma = mo_coeff[0]
mb = mo_coeff[1]
nalpha = (mol.nelectron+mol.spin)/2
nbeta  = (mol.nelectron-mol.spin)/2
print nalpha,nbeta,mol.spin,nb
pTa = numpy.dot(ma[:,:nalpha],ma[:,:nalpha].T)
pTb = numpy.dot(mb[:,:nbeta],mb[:,:nbeta].T)
pT = pTa+pTb
pT = 0.5*pT
# Lowdin basis
s12 = sqrtm(ova)
s12inv = lowdin(ova)
#eig,coeff = scipy.linalg.eigh(pT,ova,type=2)
pT = reduce(numpy.dot,(s12,pT,s12))
print 'idemponency of DM:',numpy.linalg.norm(pT.dot(pT)-pT)
#
# 'natural' occupations and orbitals
#
enorb = mf["mo_energy"]
print '\nCMO_enorb:'
print enorb
#import matplotlib.pyplot as plt
#plt.plot(range(nb),enorb[0],'ro')
#plt.plot(range(nb),enorb[1],'bo')
#plt.show()
fa = reduce(numpy.dot,(ma,numpy.diag(enorb[0]),ma.T))
fb = reduce(numpy.dot,(mb,numpy.diag(enorb[1]),mb.T))
#
# Non-orthogonal cases: FC=SCE
# Fao = SC*e*C{-1} = S*C*e*Ct*S
# OAO basis:
# F = Xt*Fao*X = S1/2*C*e*Ct*S1/2
#
fav = (fa+fb)/2
fOAO = reduce(numpy.dot,(s12,fav,s12))
shift = 0.0
pTshift = pT + shift*fOAO
eig,coeff = scipy.linalg.eigh(pTshift)
#eig = 2.0*numpy.diag( reduce(numpy.dot,(coeff.T,pT,coeff)) )
eig = 2.0*eig
eig[eig<0.0]=0.0
eig[abs(eig)<1.e-14]=0.0
ifplot = False #True
if ifplot:
   import matplotlib.pyplot as plt
   plt.plot(range(nb),eig,'ro')
   plt.show()
coeff = numpy.dot(s12inv,coeff)
diff = reduce(numpy.dot,(coeff.T,ova,coeff)) - numpy.identity(nb)
print 'CtSC-I',numpy.linalg.norm(diff)
#
# Expectation value of natural orbitals <i|F|i>
#
fexpt = reduce(numpy.dot,(coeff.T,ova,fav,ova,coeff))
enorb = numpy.diag(fexpt)
# Sort by occupancy
index = numpy.argsort(-eig)
enorb = enorb[index]
nocc  = eig[index]
#
# Reordering and define active space according to thresh
#
coeff = coeff[:,index]
idx = 0
thresh = 0.01
active=[]
for i in range(nb):
   if nocc[i]<=2.0-thresh and nocc[i]>=thresh:
      active.append(True)
   else:
      active.append(False)
print '\nNatural orbitals:'
print 'Offdiag(F) =',numpy.linalg.norm(fexpt - numpy.diag(enorb))
for i in range(nb):
   print 'orb:',i,active[i],nocc[i],enorb[i]
print active.count(True)
active = numpy.array(active)
actIndices = list(numpy.argwhere(active==True).flatten())
print actIndices
cOrbs = coeff[:,:actIndices[0]]
aOrbs = coeff[:,actIndices]
vOrbs = coeff[:,actIndices[-1]+1:]
nb = cOrbs.shape[0]
nc = cOrbs.shape[1]
na = aOrbs.shape[1]
nv = vOrbs.shape[1]
print 'core orbs:',cOrbs.shape
print 'act  orbs:',aOrbs.shape
print 'vir  orbs:',vOrbs.shape
assert nc+na+nv == nb
#
# dump UNO
#
from pyscf.tools import molden
with open(fname+'_uno.molden','w') as thefile:
    molden.header(mol,thefile)
    molden.orbital_coeff(mol,thefile,coeff)

#=============================
# local orbitals
#=============================
from pyscf.tools import molden,localizer
iflocal  = False #True
if iflocal:
   loc = localizer.localizer(mol,ma[:,:mol.nelectron/2],'boys')
   loc.verbose = 10
   new_coeff = loc.optimize()
   loc = localizer.localizer(mol,ma[:,mol.nelectron/2:],'boys')
   new_coeff2 = loc.optimize()
   lmo = numpy.hstack([new_coeff,new_coeff2])
   with open(fname+'lmo.molden','w') as thefile:
      molden.header(mol,thefile)
      molden.orbital_coeff(mol,thefile,lmo,symm=['A']*lmo.shape[1])

#=====================
# Population analysis
#=====================
# (K,n)
cOrbsOAO = numpy.dot(s12,cOrbs)
aOrbsOAO = numpy.dot(s12,aOrbs)
vOrbsOAO = numpy.dot(s12,vOrbs)
print 'Ortho-cOAO',numpy.linalg.norm(numpy.dot(cOrbsOAO.T,cOrbsOAO)-numpy.identity(nc))
print 'Ortho-aOAO',numpy.linalg.norm(numpy.dot(aOrbsOAO.T,aOrbsOAO)-numpy.identity(na))
print 'Ortho-vOAO',numpy.linalg.norm(numpy.dot(vOrbsOAO.T,vOrbsOAO)-numpy.identity(nv))

ifACT = False
if not ifACT:
   nc2 = nc
   na2 = na
   nv2 = nv 

#==========================================
# Now try to get localized orbitals (SCDM)
#==========================================
def scdm(coeff,ova,aux):
   no = coeff.shape[1]	
   ova = reduce(numpy.dot,(coeff.T,ova,aux))
   # ova = no*nb
   q,r,piv = scipy.linalg.qr(ova, pivoting=True)
   bc = ova[:,piv[:no]]
   ova = numpy.dot(bc.T,bc)
   s12inv = lowdin(ova)
   cnew = reduce(numpy.dot,(coeff,bc,s12inv))
   return cnew

# Get <i|F|i> 
def psort(ova,fav,coeff):
   # OCC-SORT
   pTnew = 2.0*reduce(numpy.dot,(coeff.T,s12,pT,s12,coeff))
   nocc  = numpy.diag(pTnew)
   index = numpy.argsort(-nocc)
   ncoeff = coeff[:,index]
   nocc   = nocc[index]
   enorb = numpy.diag(reduce(numpy.dot,(ncoeff.T,ova,fav,ova,ncoeff)))
   return ncoeff,nocc,enorb

# AO ref: 
# aux=numpy.identity(nb)
# OAO ref
aux = s12inv
#from pyscf import lo
#aux = lo.orth.orth_ao(mol,method='meta_lowdin',pre_orth_ao=lo.orth.pre_orth_ao(mol))

cOrbs = s12inv.dot(cOrbsOAO)
aOrbs = s12inv.dot(aOrbsOAO)
vOrbs = s12inv.dot(vOrbsOAO)
clmo = scdm(cOrbs,ova,aux)
almo = scdm(aOrbs,ova,aux)
vlmo = scdm(vOrbs,ova,aux)

# E-SORT
mo_c,n_c,e_c = psort(ova,fav,clmo)
mo_o,n_o,e_o = psort(ova,fav,almo)
mo_v,n_v,e_v = psort(ova,fav,vlmo)
coeff = numpy.hstack((mo_c,mo_o,mo_v))

# CHECK
diff = reduce(numpy.dot,(coeff.T,ova,coeff)) - numpy.identity(nb)
print 'diff=',numpy.linalg.norm(diff)
from pyscf.tools import molden
with open(fname+'_scdm.molden','w') as thefile:
    molden.header(mol,thefile)
    molden.orbital_coeff(mol,thefile,coeff)

#==========================================
# lowdin-pop of the obtained LMOs in OAOs 
#==========================================
print '\nLowdin population for LMOs:'
lcoeff = s12.dot(coeff)
diff = reduce(numpy.dot,(lcoeff.T,lcoeff)) - numpy.identity(nb)
print 'diff=',numpy.linalg.norm(diff)

pthresh = 0.02
labels = mol.spheric_labels()
ifACTONLY = False #True
nelec = 0.0
nact = 0.0
for iorb in range(nb):
   vec = lcoeff[:,iorb]**2
   idx = list(numpy.argwhere(vec>pthresh))
   if ifACTONLY == False:
      if iorb < nc2:
         print ' iorb_C=',iorb,' occ=',n_c[iorb],' fii=',e_c[iorb]
         nelec += n_c[iorb]
      elif iorb >= nc2 and iorb < nc2+na2:
         print ' iorb_A=',iorb,' occ=',n_o[iorb-nc2],' faa=',e_o[iorb-nc2]
         nelec += n_o[iorb-nc2]
      else:
         print ' iorb_V=',iorb,' occ=',n_v[iorb-nc2-na2],' fvv=',e_v[iorb-nc2-na2]
         nelec += n_v[iorb-nc2-na2]
      for iao in idx:
         print '    iao=',labels[iao],' pop=',vec[iao]
   else:
      if iorb >= nc2 and iorb < nc2+na2:
         print ' iorb_A=',iorb,' faa=',e_o[iorb-nc2]
         for iao in idx:
            print '    iao=',labels[iao],' pop=',vec[iao]
print 'nelec=',nelec

#==========================
# select 'active' orbitals
#==========================
# HS case
a1 = [80,82,83,84,85,86] # S-3p = 7
o1 = [ 2]*6
a2 = [87,88,89,90,91,92,93,94,95,96] # Fe-Mo (3d,4d) = 10
o2 = [ 1]*8+[0]*2 
a3 = [97,98,99,101,103,105] # Mo-s + Fe4d = 6
o3 = [0]*6
#==========================
# select 'active' orbitals
#==========================
caslst = a1+a2
norb = len(caslst)
ne_act = 20
s = 1 # 0,1,2,3,4, High-spin case ms = s
ne_alpha = ne_act/2 + s
ne_beta  = ne_act/2 - s
nalpha = ne_alpha 
nbeta = ne_beta
norb = len(caslst)
print 'norb/nacte=',norb,[nalpha,nbeta]

from pyscf.dmrgscf import settings
settings.MPIPREFIX = 'srun'

from pyscf.dmrgscf.dmrgci import * 
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.dmrgscf.dmrgci import DMRGCI
from pyscf.mrpt.nevpt2 import sc_nevpt
from pyscf.dmrgscf import settings
settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR1']

mol.build(
    verbose = 7,
    output = 'hs.out'
)

mf = scf.sfx2c1e(scf.RHF(mol))

mc = mcscf.CASSCF(mf, norb, [nalpha,nbeta])
mc.chkfile = 'hs_mc.chk'
mc.max_memory=30000
mc.fcisolver = DMRGCI(mol,maxM=1000, tol =1e-6)
mc.callback = mc.fcisolver.restart_scheduler_()
orbs = mc.sort_mo(caslst,coeff,base=0)
mc.mc1step(orbs)

#CASCI-NEVPT2
#mc = mcscf.CASCI(mf, norb, [nalpha,nbeta])
#mc.chkfile = 'hs_mc.chk'
#f = h5py.File('hs_mc.chk','r')
#mo = f["mcscf/mo_coeff"]
#f.close()
#mc.fcisolver = DMRGCI(mol,maxM=500, tol =1e-8)
##mc.fcisolver.outputlevel = 3
#mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20, 24, 28, 30, 34]
#mc.fcisolver.scheduleMaxMs  = [200, 400, 800, 1200, 2000, 4000, 3000, 2000, 1000, 500]
#mc.fcisolver.scheduleTols   = [0.0001, 0.0001, 0.0001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7 ]
#mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0]
#mc.fcisolver.twodot_to_onedot = 38
#mc.fcisolver.maxIter = 50
#mc.mo_coeff = mo
#mc.casci()
#sc_nevpt(mc)
#os.system("rm %s/*/Spin*tmp"%settings.BLOCKSCRATCHDIR)
#os.system("rm %s/*/C*tmp"%settings.BLOCKSCRATCHDIR)
#os.system("rm %s/*/D*tmp"%settings.BLOCKSCRATCHDIR)

