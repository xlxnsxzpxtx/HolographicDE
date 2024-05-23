from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import optimize
import numpy as np



class TsallisCosmology(LCDMCosmology):
    """
        This is Holographic Dark Energy in modfied Tsallis Cosmology.
        This class inherits LCDMCosmology class as the rest of the cosmological
        models already included in SimpleMC.

        :param varyc: variable w0 parameter

    """

 
    def __init__(self, varyc=True,varys=True):
        # Holographic and Tsallis parameter
        self.c_par = Parameter("c", 1.0, 0.05, (0.2, 2.0), "c")
        self.s_par = Parameter("s", 0.5, 0.05, (0.0, 3.0), "s") 
        #self.B_par = Parameter("B", 3.0, 0.05, (2.5,3.5), "B") # https://arxiv.org/abs/1806.01301
        #self.varyOk = varyOk
        self.varyc  = varyc
        self.varys  = varys  
        #self.varyB  = varyB
        
        self.Ok = Ok_par.value
        self.c  = self.c_par.value # holographic parameter 
        self.s  = self.s_par.value # Tsallis parameter
        #self.B =  self.B_par.value 
        # This value is quite related to the initial z
        self.zini = 3
        self.xini = np.log(1./(1+self.zini))
        #self.scale = 10**(-2)
        self.xvals = np.linspace(0,self.xini,50)

        LCDMCosmology.__init__(self)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #if (self.varyOk): l.append(Ok_par)
        if (self.varyc):  l.append(self.c_par)
        if (self.varys):  l.append(self.s_par)
        #if (self.varyB):  l.append(self.B_par)
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
            #elif p.name == "B": 
            #    self.B == p.value
            #elif p.name == "Ok":
             #   self.Ok = p.value
            #self.setCurvature(self.Ok)
            #if (abs(self.Ok) > 1.0):
             #    return False

        self.initialize()
        return True

# x = ln(a)
    def RHS_x_hde_tsallis(self, Omega, x):    
        b = 2*(self.s -1)
        Q = (2-Omega)*((self.c**2)**(1./(b - 2)))*((self.h*np.sqrt(self.Om))**(b/(2 - b)))
        dOmega = Omega*(1-Omega)*(b + 1 + Q*((1-Omega)**(b/2*(b -2)))*(Omega**(1./(2-b)))*np.exp((3*b/(2-b))*x))
        return dOmega

   # def compute_Ode(self, Ode_ini):
    #    Ode = Ode_ini*self.scale
    #    solution = odeint(self.RHS_a_hde_tsallis, Ode, self.avals, h0=1E-5)
     #   return solution


    #def ini_sol(self, Ode_ini):
     #   diference = self.compute_Ode(Ode_ini)[-1] - (1-self.Om-self.Ok)
      #  return diference


    def initialize(self):
        """
        Main method that searches the initial conditions for a given model.

        """
    
        if 1.9999<self.s<2.001: 
           Ode0 = (1 - self.Om)   
        else: 
            Ode0 = 1 - self.Om
            result_E = odeint(self.RHS_x_hde_tsallis, Ode0, self.xvals)

            self.Ode = interp1d(self.xvals, result_E[:, 0])
           #print(self.s)
        return True

        #ini_val = optimize.newton(self.ini_sol, 1)
        
        #Ode = self.compute_Ode(ini_val)

        #ini_vals = Ode0
    

         #found, i.e. c<0.4
        




    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        x = np.log(a)
        hubble = (self.Om/a**3)/(1-self.Ode(x))
        return hubble


