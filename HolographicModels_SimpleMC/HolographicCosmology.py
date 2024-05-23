from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np



class HolographicCosmology(LCDMCosmology):
    """
        This is Holographic cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter

    """


    def __init__(self,varyc = True):
        # Holographic parameter
        self.c_par = Parameter("c", 0.7, 0.1, (0.5, 1.1), "c")
        self.varyc  = varyc

        self.c  = self.c_par.value

              # This value is quite related to the initial z
        self.zini = 3
        #self.avals = np.linspace(1./(1+self.zini), 1, 300)
        self.zvals    = np.linspace(0, self.zini, 300)
        #self.z_int = np.linspace(0, 2, 500)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "c":
                self.c = p.value  
            
            

        self.initialize()
        return True


    def RHS_hde(self, Omega, z):
        #c = 0.75
        fact = (1 + 2*np.sqrt(Omega)/self.c) 
        dOmega = -Omega*(1 - Omega)*fact/(1 + z) + 0
        return dOmega




    def initialize(self):
        
        Ode0 =  (1 - self.Om)
        result_E = odeint(self.RHS_hde, Ode0, self.zvals)
        self.Ode = interp1d(self.zvals, result_E[:,0])
        #f = self.Ode(self.z_int)
        #print('Omega_de(z=0) para cada valor usando interp1d',f[0])
        #self.Omega_hde = interp1d(self.avals, Ode[:, 0])
        return True

    

    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        z = 1./a-1
        hubble = (self.Om/a**3)/(1-self.Ode(z))
        #f1 = (self.Om/a[0]**3)/(1-self.Ode(z)[0])
        #print(100*self.h*hubble)      
        return hubble


    

    

