# Barrow cosmology
from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np



class BarrowHDECosmology(LCDMCosmology):
    """
        This is Holographic Dark Energy in modfied Barrow Cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter

    """


    def __init__(self, varyc=True,varyb=True):
        # Holographic and Barrow parameter
        self.c_par = Parameter("c", 1.5, 0.0001, (1.0,2.0), "c")
        self.b_par = Parameter("b", 0.2, 0.000001, (0.1,0.4), "b") 
        #self.varyOk = varyOkq
        self.varyc  = varyc
        self.varyb  = varyb  
        
        #self.Ok = Ok_par.value
        self.c  = self.c_par.value # holographic parameter 
        self.b  = self.b_par.value # barrow parameter

        # This value is quite related to the initial z
        self.zini = 3
        #self.xfin = np.log(1./(1+self.zini))
        #self.scale = 10**(-2)
        self.zvals = (0, 3)
        self.t_eval = np.linspace(0, self.zini, 50)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varyb):  l.append(self.b_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name   == "c":
                self.c = p.value
            elif p.name == "b":
                self.b = p.value


        self.initialize()
        return True

# x = ln(a)
    






    def RHS_hde(self,z,Omega):
        x = np.log(1./(1+z))
        H0 = 100*self.h

        exp1 = 1/(self.b-2)
        exp2 = self.b/(2-self.b)
        exp3 = self.b/(2*(self.b-2))     
        exp4 = 1/(2-self.b)
        exp5 = 3*self.b/(2*(self.b-2))

        Q = (2-self.b)*((self.c**2)**exp1 )*((H0*np.sqrt(self.Om))**exp2)

        
        factor1 = - (Omega*(1 - Omega))/(1+z) 
        term1 = self.b + 1 
        term2 = Q*((1-Omega)**exp3)*(Omega**exp4)*np.exp(exp5*x)

        dOmega = factor1*(term1 + term2)
        


    
        return dOmega

    def EoS(self,z,Omega):
        
        
        exponent1= 1/(2-self.b)
        exponent2 = self.b/(2*(self.b-2))
        exponent3 = 3*self.b/(2*(2-self.b))
        exponent4 = (self.b)/(2-self.b) 
        exponent5 = 1/(self.b-2)
        
        x = np.log(1./(1+z))
        H0 = (100*self.h)
        Q =(2-self.b)*((self.c*2)**exponent5)*((H0*np.sqrt(self.Om))**exponent4)


        w = -(1+self.b)/3 - (Q/3)*(Omega**exponent1)*((1-Omega)**exponent2)*np.exp(exponent3*x)
        
        return w 



    def initialize(self):
        
        Ode0 = [1 - self.Om]
        result_E = solve_ivp(self.RHS_hde, self.zvals, Ode0, t_eval=self.t_eval, method='RK45', atol=1e-12, rtol=1e-12)
        
        # Interpolate the result
        self.Ode = interp1d(result_E.t, result_E.y[0], kind='cubic')
        z_plot = np.linspace(0, self.zini, 50)
        Omega_values = self.Ode(z_plot)
        Eos_values = self.EoS(z_plot, Omega_values)
        
        #plt.plot(z_plot, Eos_values)
      
        return True




    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        z = 1/a-1
        hubble = (self.Om/a**3)/(1-self.Ode(z))

        #print(100*self.h*hubble)
        #print(self.b)
        #f1 = (self.Om/a[0]**3)/(1-self.Ode(z)[0])
        #print(100*self.h*hubble)      
        return hubble
