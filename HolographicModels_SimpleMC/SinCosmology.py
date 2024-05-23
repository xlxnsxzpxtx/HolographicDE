from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np



class SinCosmology(LCDMCosmology):
    """
        This is Holographic cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter

    """


    def __init__(self,varyc = True,varybeta = True,varyalpha = True ):
        # Holographic parameter
        self.c_par = Parameter("c",1.0, 0.1, (0.5, 2.0), "c")
        self.beta_par  = Parameter("beta", 0.2, 0.1, (0.0,1.0),  "\beta")
        self.alpha_par = Parameter("alpha",0.8, 0.1, (0.0,1.0), "\alpha")
        self.varyc  = varyc
        self.varybeta  = varybeta
        self.varyalpha  = varyalpha 

        self.c  = self.c_par.value
        self.beta  = self.beta_par.value 
        self.alpha  = self.alpha_par.value 

              # This value is quite related to the initial z
        self.zini = 3
        #self.avals = np.linspace(1./(1+self.zini), 1, 300)
        self.zvals    = np.linspace(0, self.zini, 50)
        #self.z_int = np.linspace(0, 2, 500)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varybeta):  l.append(self.beta_par)
        if (self.varyalpha):  l.append(self.alpha_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "c":
                self.c = p.value  
            elif p.name == "beta":
               self.beta = p.value
            elif p.name == "alpha":
               self.alpha = p.value 
            

        self.initialize()
        return True


    def RHS_hde(self, Omega, z):
        f = self.alpha + self.beta*np.sin(z)
        f_prim =  self.beta*np.cos(z)
        Q = self.c**2/(self.Om*(1+z)**3)*(self.h**2)
        dOmega = - (Omega*(1-Omega)/(1+z))*((2*np.sqrt(Omega)/self.c)*((Q*(1-Omega)/Omega)**((f-1)/(2*(f-2))))*(2 - f)  + 2*f -  2 + (np.log(Q*(1-Omega)/Omega)**(-(1+z)*f_prim/(f-2))) )
        return dOmega




    def initialize(self):
        
        Ode0 = (1 - self.Om)
        result_E = odeint(self.RHS_hde, Ode0, self.zvals)
        self.Ode = interp1d(self.zvals, result_E[:,0])
        print(result_E[:,0])
        #print(self.alpha)
        #print(self.beta)
        #print(result_E[:,0])
        #f = self.Ode(self.z_int)
        #print('Omega_de(z=0) para cada valor usando interp1d',f[0])
        #self.Omega_hde = interp1d(self.avals, Ode[:, 0])
        return True

    

    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        z = 1./a-1
        hubble = (self.Om/a**3)/(1-self.Ode(z))
        #print(100*self.h*hubble)
        #print(self.b)
        #f1 = (self.Om/a[0]**3)/(1-self.Ode(z)[0])
        #print(100*self.h*hubble)      
        return hubble


    

    

