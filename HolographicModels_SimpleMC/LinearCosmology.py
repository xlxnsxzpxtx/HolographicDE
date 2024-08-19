from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

class LinearCosmology(LCDMCosmology):
    """
        This is Holographic cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter
    """

    def __init__(self, varyc=True, varyb=True, varya=True):
        # Holographic parameter
        self.c_par = Parameter("c",   1.0,    0.001,  (0.0, 2), "c")
        self.b_par = Parameter("b",   0.25,  0.001,   (-0.5,0.5), "b")
        self.a_par = Parameter("a",   0.25,  0.001,   (-0.5,0.5), "a")
        self.varyc = varyc
        self.varyb = varyb
        self.varya = varya

        self.c = self.c_par.value
        self.b = self.b_par.value
        self.a = self.a_par.value

        # This value is quite related to the initial z
        self.zini = 3
        self.zvals = (0, 3)
        self.t_eval = np.linspace(0, self.zini, 50)

        LCDMCosmology.__init__(self)
        self.updateParams([])

    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if self.varyc: l.append(self.c_par)
        if self.varyb: l.append(self.b_par)
        if self.varya: l.append(self.a_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "c":
                self.c = p.value
            elif p.name == "b":
                self.b = p.value
            elif p.name == "a":
                self.a = p.value
        self.initialize()
        return True

    def RHS_hde(self, z, Omega):
        x = np.log(1. / (1 + z))
        H0 = (100*self.h)
        exponent1 = (self.a + self.b*z) / (2 * (self.a + self.b*z - 2))
        exponent2 = -(1+z)*self.b/(self.a +self.b*z-2)
        Q = ((self.c**2)* np.exp(3*x)) / ((H0**2)*self.Om)
        dOmega_DE_dz =  -((Omega*(1 - Omega))/ (1 + z))*(np.sqrt(Omega)*((self.c**2)**(-0.5))*((Q*(1-Omega)/Omega)**exponent1)*(2-self.a-self.b*z) + np.log((Q*(1-Omega)/Omega)**exponent2) + self.a + self.b*z +  1)
        return dOmega_DE_dz

    def EoS(self,z,Omega):
        x = np.log(1. / (1 + z))
        H0 = (100*self.h)
        Q = ((self.c ** 2) * np.exp(3 * x)) / ((H0 ** 2) * self.Om)
        exponent1 = (self.a + self.b * z) / (2 * (self.a + self.b * z - 2))
        exponent2 = 1. / (2 - self.a - self.b * z)
        w_DE = -(self.a + self.b * z + 1) / 3 + ((self.a + self.b * z - 2) * np.sqrt(Omega) / 3)*((self.c ** 2) ** (-0.5))*((Q*(1 - Omega)/Omega)**exponent1) + ((1 + z) * self.b / 3) * np.log((Q * (1 - Omega) / Omega) ** exponent2)
        return w_DE

    def initialize(self):
        Ode0 = [1 - self.Om]
        result_E = solve_ivp(self.RHS_hde, self.zvals, Ode0, t_eval=self.t_eval, method='RK45', atol=1e-12, rtol=1e-12)
        
        # Interpolate the result
        self.Ode = interp1d(result_E.t, result_E.y[0], kind='cubic')
        #print("Solution values (y[0]):", result_E.y[0])
        # Plot the EoS for demonstration purposes
        
        #z_plot = np.linspace(0, self.zini, 50)
        #Omega_values = self.Ode(z_plot)
        #Eos_values = self.EoS(z_plot, Omega_values)
        
       
        #plt.plot(z_plot, Eos_values)
        #plt.grid(True)
        #plt.xlabel('z')
        #plt.ylabel('$\omega_{de}$')
        #plt.legend()
        #plt.show()

        return True

    def RHSquared_a(self, a):
        z = 1. / a - 1
        hubble = (self.Om / a ** 3) / (1 - self.Ode(z))
        return hubble
