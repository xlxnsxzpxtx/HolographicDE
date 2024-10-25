import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter

class TsallisCosmology(LCDMCosmology):
    """
    This is Holographic Dark Energy with Tsallis entropy.
    This class inherits LCDMCosmology class as the rest of the cosmological
    models already included in SimpleMC.
    :param varyc: variable w0 parameter
    """

    def __init__(self, varyc=True,varys=True):
        # Holographic and Tsallis parameter
        self.c_par = Parameter("c", 1.0, 0.01, (0.0,2.0), "c")
        self.s_par = Parameter("s", 1.2, 0.01, (1.0,1.9), "s") 
     
        self.varyc  = varyc
        self.varys  = varys  
        
        #s
        self.c  = self.c_par.value # holographic parameter 
        self.s  = self.s_par.value # Tsallis parameter

        # This value is quite related to the initial z
        self.zini = 3
        #self.xfin = np.log(1./(1+self.zini))
        #self.xval =(self.xfin,0)
        #self.scale = 10**(-2)
        self.zval = (0, 3)
        self.xini = np.log(1./(1+self.zini))
        self.xval = (0,self.xini)
        self.t_eval = np.linspace(0,self.xini, 50)



        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varys):  l.append(self.s_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name   == "c":
                self.c = p.value
            elif p.name == "s":
                self.s = p.value


        self.initialize()
        return True

# x = ln(a)
    






    def RHS_hde(self,x,Omega):

        Omega_DE = Omega
        delta = self.s 
        H0 = 100*self.h 

        Q =  2 * (2 - delta) * (self.c**2)**(1 / (2 * (delta - 2))) * (H0 * np.sqrt(self.Om))**((1 - delta) / (delta - 2))

        dOmega   = (Omega_DE * (1 - Omega_DE))*(2 * delta- 1 + Q * ((1 - Omega_DE) ** ((1 - delta) / (2 * (2 - delta))))*(Omega_DE ** (1 / (2 * (2 - delta))))*np.exp((3 * (1 - delta) / (2 * (2 - delta)) * x))) 

        return dOmega

        



    def initialize(self):
        
        Ode0 = [1 - self.Om]
        result_E = solve_ivp(self.RHS_hde, self.xval, Ode0, t_eval=self.t_eval, method='RK45', atol=1e-12, rtol=1e-12)
        
        # Interpolate the result
        self.Ode = interp1d(result_E.t, result_E.y[0], kind='cubic')

        #plt.plot(self.zval,result_E)
        
        return True




    # External methods useful for plotting or testing.

    def EoS(self,z):
         
        x = np.log(1./(1+z)) 
        
        Omega_DE = self.Ode(x)
        delta = self.s 
        H0 = 100*self.h 

        Q =  2 * (2 - delta) * (self.c**2)**(1 / (2 * (delta - 2))) * (H0 * np.sqrt(self.Om))**((1 - delta) / (delta - 2))
        
        term1 = (1 - 2 * delta) / 3

     # Second term
        exponent1 = 1 / (2 * (2 - delta))
        exponent2 = (delta - 1) / (2 * (delta - 2))
        exponent3 = 3 * (1 - delta) / (2 * (delta - 2))

        term2 = (Q / 3) * (Omega_DE ** exponent1) * ((1 - Omega_DE) ** exponent2) * np.exp(exponent3 * x)

    # Combine terms
        w = term1 - term2

        return w


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        x = np.log(a)
        hubble = (self.Om/(a**3))/(1-self.Ode(x))

        return hubble

 