import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter

class TsallisCosmology(LCDMCosmology):
    """
    This is Holographic Dark Energy in modified Tsallis Cosmology.
    This class inherits LCDMCosmology class as the rest of the cosmological
    models already included in SimpleMC.
    :param varyc: variable w0 parameter
    """

    def __init__(self, varyc=True,varys=True):
        # Holographic and Barrow parameter
        self.c_par = Parameter("c", 0.5, 0.01, (0.0,2.0), "c")
        self.s_par = Parameter("s", 1.5, 0.01, (1.0,1.5), "s") 
        #self.varyOk = varyOk
        self.varyc  = varyc
        self.varys  = varys  
        
        #self.Ok = Ok_par.value
        self.c  = self.c_par.value # holographic parameter 
        self.s  = self.s_par.value # barrow parameter

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
    






    def RHS_hde(self,z,Omega):
        exponent1 = (1-self.s)/(2*(2-self.s))
        exponent2 = 1/(2*(2-self.s))
        exponent3 = 3*(1-self.s)/(2*(2-self.s))
        exponent4 = (1-self.s)/(self.s-2) 
        exponent5 = 1/(2*(self.s-2))

        x = np.log(1./(1+z))
        H0 = (100*self.h)
        Q =2*(2-self.s)*((self.c*2)**exponent5)*((H0*np.sqrt(self.Om))**exponent4)
        # Compute common terms
            
        
        
        factor1 = - (Omega*(1 - Omega))/(1+z)
        term1 = 2*self.s-1 
        term2 = Q*((1-Omega)**exponent1)*(Omega**exponent2)*np.exp(exponent3*x)

        dOmega = factor1*(term1 + term2)

        return dOmega

    def EoS(self,z,Omega):
        
        
        exponent1= 1/(2*(2-self.s))
        exponent2 = (self.s-1)/(2*(self.s-2))
        exponent3 = 3*(1-self.s)/(2*(self.s-2))
        exponent4 = (1-self.s)/(self.s-2) 
        exponent5 = 1/(2*(self.s-2))
        
        x = np.log(1./(1+z))
        H0 = (100*self.h)
        Q =2*(2-self.s)*((self.c*2)**exponent5)*((H0*np.sqrt(self.Om))**exponent4)


        w = (1-2*self.s)/3 - (Q/3)*(Omega**exponent1)*((1-Omega)**exponent2)*np.exp(exponent3*x)
        
        return w 

    



    def initialize(self):
        
        Ode0 = [1 - self.Om]
        result_E = solve_ivp(self.RHS_hde, self.zvals, Ode0, t_eval=self.t_eval, method='RK45', atol=1e-12, rtol=1e-12)
        
        # Interpolate the result
        self.Ode = interp1d(result_E.t, result_E.y[0], kind='cubic')
        #z_plot = np.linspace(0, self.zini, 50)
        #Omega_values = self.Ode(z_plot)
        #Eos_values = self.EoS(z_plot, Omega_values)
        
      
        #plt.plot(z_plot, Eos_values)
        #result_E_re = result_E.reshape(-1)
        #Eos_linear = self.EoS(result_E_re,self.zvals)
        #print(result_E)
        #plt.plot(self.zvals,result_E_re)
        #plt.grid(True)
        #plt.xlabel('z')
        #plt.ylabel('$\omega_{de}$')
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
