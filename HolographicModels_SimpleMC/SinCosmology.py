from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
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


    def __init__(self,varyc = True,varyk = True,varyr = True ):
        # Holographic parameter
        self.c_par = Parameter("c",  1.38,  0.0001,   (0.5,1.5), "c")
        self.k_par  = Parameter("k", 0.001, 0.0000001, (0.0,0.2),  "k")
        self.r_par = Parameter("r",  0.02,  0.000001, (0.0 , 0.2), "r")
        self.varyc  = varyc
        self.varyk  = varyk 
        self.varyr  = varyr

        self.c  = self.c_par.value
        self.k  = self.k_par.value 
        self.r  = self.r_par.value 

              # This value is quite related to the initial z
        self.zini = 3
        #self.avals = np.linspace(1./(1+self.zini), 1, 300)
        self.zvals = (0, 3)
        self.t_eval = np.linspace(0, self.zini, 50)
        #self.z_int = np.linspace(0, 2, 500)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varyk):  l.append(self.k_par)
        if (self.varyr):  l.append(self.r_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "c":
                self.c = p.value  
            elif p.name == "k":
               self.k = p.value
            elif p.name == "r":
               self.r = p.value 
            

        self.initialize()
        return True


    def RHS_hde(self,z,Omega):
        x = np.log(1./(1+z))
        C = 3*(self.c**2)
        H0 = (100*self.h)
        Q = (C*np.exp(3*x))/(3*(H0**2)*self.Om)
        term1 = - (Omega* (1 - Omega))/(1 + z) 
        sqrt_Omega_DE = np.sqrt(Omega)
        term2_base = (C /3)**(-0.5)
        fraction_Q_Omega = Q * (1 - Omega)/ Omega
    
    # Exponents
        exponent1 = (self.k + self.r * np.sin(z)) / (2 * (self.k + self.r*np.sin(z) - 2))
        exponent2 = (-(1 + z) *self.r * np.cos(z)) / (self.k + self.r *np.sin(z) - 2)
    
    # Calculate individual components of the right-hand side
        rhs1 = sqrt_Omega_DE * term2_base * (fraction_Q_Omega)**exponent1 * (2 - self.k - self.r*np.sin(z))
        rhs2 = np.log(fraction_Q_Omega**exponent2)
        rhs3 = self.k+ self.r*np.sin(z) + 1
    
    # Combine all terms to calculate Omega_DE'
        rhs = rhs1 + rhs2 + rhs3
        Omega_DE_prime = term1 * rhs
    
        return Omega_DE_prime


    def EoS(self,z,Omega):
        x = np.log(1./(1+z))
        C = 3*(self.c**2)
        H0 = 100*self.h
        Q = (C*np.exp(3*x))/(3*(H0**2)*self.Om)
        sin_z = np.sin(z)
        cos_z = np.cos(z)
        sqrt_Omega_DE = np.sqrt(Omega)
        term_base = (C / 3)**(-0.5)
        fraction_Q_Omega = Q*(1 - Omega)/Omega

    # Components of the equation
        term1 = - (self.k + self.r*sin_z + 1) / 3
        exponent1 = (self.k + self.r*sin_z) / (2 * (self.k + self.r*sin_z - 2))
        term2 = ((self.k + self.r*sin_z- 2) * sqrt_Omega_DE / 3) * term_base * (fraction_Q_Omega**exponent1)
        exponent2 = 1 / (2 - self.k -self.r*sin_z)
        term3 = ((1 + z) * self.r* cos_z / 3) * np.log(fraction_Q_Omega**exponent2)

    # Calculate w_DE
        w_DE = term1 + term2 + term3
    
        return w_DE


    def initialize(self):
        
        Ode0 = [1 - self.Om]
        result_E = solve_ivp(self.RHS_hde, self.zvals, Ode0, t_eval=self.t_eval, method='RK45', atol=1e-12, rtol=1e-12)
        
        # Interpolate the result

        self.Ode = interp1d(result_E.t, result_E.y[0], kind='cubic')
        z_plot = np.linspace(0, self.zini, 50)
        Omega_values = self.Ode(z_plot)
        Eos_values = self.EoS(z_plot, Omega_values)
        plt.plot(z_plot, Eos_values)
        #plt.grid(True)
        #plt.xlabel('z')
        #plt.ylabel('$\omega_{de}$')
        #print(result_E[:,0])
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


    

    

