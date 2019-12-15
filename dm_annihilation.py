import numpy as np
import data_write
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

class annihilation:

    def __init__(self, ann_cs = 3e-26, dec_rate = 4e-26, w_m = 0.308, w_b = 0.0495, w_x = 0.259, w_l = 0.692, h = 0.678):
        self.halo_profile = None    #halo density profile
        self.d_l = None             #luminosity distance to halo (pc)
        self.ann_cs = ann_cs        #annihilation cross-section
        self.dec_rate = dec_rate    #decay rate
        self.z = 10                 #default redshift (at which accretion ends and virial radius is set)
        self.w_m = w_m              #PLANCK (2015) matter density parameter
        self.w_l = w_l              #PLANCK (2015) lambda parameter
        self.w_x = w_x              #dark matter density parameter  #omega_m - omega_b (baryons)
        self.h = h                  #normalised Hubble constant (= Ho/100)
        self.ann_or_dec = None      #sets functions for annihilation ("ann") or decay ("dec")
        self.m_x = None             #WIMP mass (GeV)
        self.m_vir = None           #halo mass (M_sun)
        self.c_vir = None           #halo concentration
        self.r_vir = None           #halo radius
        self.r_97 = None            #halo radius for 97% flux containment
        self.r_vir_arr = None       #halo radius integration sample
        self.rho_arr = None         #halo density profile over r_vir_arr
        self.channel = None         #annihilation channel
        self.delta_c = None         #overdensity parameter
        self.rho_crit = None        #critical density for flat universe
        self.rho_bar = None         #average background density (as a function of z)
        self.jfac = None            #numerical j factor in flux calculation
        self.n = 1000               #default number of integration steps


    #initialise basic halo and WIMP properties
    #input p is a list of [WIMP mass (m_x), DM halo mass (m_vir), ann/dec channel (ch), halo profile (h_p), halo distance (d_l), annihilation or decay mode (ann_or_dec)]
    #units should at this point be ([m_x] = GeV/c^2, [m_vir] = M_sun, [d_l] = AU)
    def setup(self,p):

        m_x         = p[0]
        m_vir       = p[1]
        ch          = p[2]
        h_p         = p[3]
        d_l         = p[4]
        ann_or_dec  = p[5]

        AU_to_pc = 1/206265
        self.m_x = m_x
        self.m_vir = m_vir
        self.channel = ch
        self.halo_profile = h_p
        self.d_l = d_l*AU_to_pc
        self.ann_or_dec = ann_or_dec

        self.rho_crit = self.set_rho_crit()
        self.rho_bar = self.set_rho_bar(self.z)
        self.delta_c = self.set_deltac()

        self.c_vir = self.set_c_vir(m_vir)
        self.r_vir = self.set_rvir(m_vir)
        self.r_vir_arr = np.logspace(np.log10(self.r_vir*1e-7),np.log10(self.r_vir),self.n)
        self.rho_arr = self.halo_density(m_vir,self.r_vir_arr)


    #read the correct spectrum file and return the data arrays
    def get_spectrum(self):

        ch = self.channel
        m = self.m_x

        filename = "dnde/"+str(ch)+"_"+str(m)+"GeV.txt"
        dnde = []
        E = []

        with open(filename,"r") as file:
            lines = file.readlines()
            for i in range(np.size(lines)):
                if i == 0:
                    continue #skip header line
                col = lines[i].split()
                E.append(float(col[0]))
                dnde.append(float(col[1]))

        spec = [np.array(E),np.array(dnde)]
        return spec


    #calculate a numerical value for the halo concentration based on the paper by F. Prada et al 2012 10.1111
    def set_c_vir(self,m):

        m_sun = 1.0

        #assuming z = 0
        x = (self.w_l/self.w_m)**(1/3)
        y = (1e12*m_sun)/(self.h * m)
        [A,b,c,d] = [2.881,1.257,1.022,0.060]

        def xintegrand(x1):
            return (x1**(3/2))/((1+x1**3)**(3/2))
        i = integrate.quad(xintegrand, 0, x)

        D = (5/2)*x**(-5/2)*(1+x**3)**(1/2)*i[0]
        sigma = D * (16.9*y**0.41)/(1 + 1.102*y**0.20 + 6.22*y**0.33)
        c_vir = A*((sigma/b)**c + 1)*np.exp(d/(sigma**2))

        return c_vir


    #return critical density in units of M_sun/pc^3
    def set_rho_crit(self):

        pc_to_m = 3.0857e16     #pc -> m
        m_to_pc = 1/pc_to_m     #m -> pc

        G = 4.302e-3              #units of pc/M_sun * (km/s)^2
        G = G*(m_to_pc*1e-3)**2   #units of pc^3/M_sun.s^2

        H_0 = self.h*100*(m_to_pc*1e-3*1e-6) #units of s^-1
        H_02 = H_0**2

        rho = (3*H_02)/(8*np.pi*G)

        return rho

    #return average background density
    def set_rho_bar(self,z):

        rho_bar = self.rho_crit*self.w_m*(1+z)**3
        return rho_bar

    #return overdensity parameter
    def set_deltac(self):

        x = self.w_l
        return 18*np.pi**2 - 82*x - 39*x**2


    #return r_vir, measured in pc.
    #input parameters should be in units of ([m_vir] = M_sun)
    def set_rvir(self,m_vir):

        if self.halo_profile == "burkert":
            r = ((3/4)*(m_vir/(np.pi*self.delta_c*self.rho_crit)))**(1/3)

        elif self.halo_profile == "nfw":
            r = ((3/4)*(m_vir/(np.pi*self.delta_c*self.rho_crit)))**(1/3)

        elif self.halo_profile == ("ucmh") or ("moore"):
            z = 10  #estimated redshift when accretion stops
            r = (0.019)*(1e3/(1+z))*(self.m_vir)**(1/3)

        return r


    #returns array of the halo density, depending on chosen halo profile, in the units of r, m_vir (pc, M_sun by default)
    #accepts r as an array, ie. r_vir_arr
    def halo_density(self, m_vir,r):

        r_vir = self.r_vir
        r_s = r_vir/self.c_vir  #scale radius

        """maximum central density calculation (for ucmh/moore profiles)"""
        t = 13.76e9                 #current age of universe
        ti = 59e6                   #time of matter/radiation equality (when clump starts accreting)
        y_to_s = 365.25*24*60*60
        GeVcm_to_Msunpc = 1/37.96     #conversion from GeV/cm^3 to M_sun/pc^3

        age = (t - ti)*(y_to_s)
        rho_max = self.m_x/(self.ann_cs*age)    #[rho_max] = GeV/cm^3
        rho_max = rho_max*GeVcm_to_Msunpc       #[rho_max] = Msun/pc^3

        """define density profiles"""
        if self.halo_profile == "burkert":
            r_s = r_s/1.52
            profile = 1/((1 + r/r_s)*(1 + (r/r_s)**2))        #un-normalised density profile

            #normalisation
            integrand = (4*np.pi*r**2)*profile
            rho_s = m_vir/integrate.simps(integrand,r)

            rho = rho_s*profile

        elif self.halo_profile == "nfw":
            profile = 1/((r/r_s)*(1 + (r/r_s))**2)            #un-normalised density profile

            #normalisation
            integrand = (4*np.pi*r**2)*profile
            rho_s = m_vir/integrate.simps(integrand,r)

            rho = profile*rho_s

        elif self.halo_profile == "ucmh":
            f_x = self.w_x/self.w_m
            rho = (3*f_x*m_vir)/(16*np.pi*(r_vir)**(3.0/4)*(r)**(9.0/4))

            r_central = np.where(rho >= rho_max)[0]
            rho[r_central] = rho_max

        elif self.halo_profile == "moore":
            #see 1806.07389 (delos2018) for details
            alpha = 3/2
            profile = 1/((r/r_s)**alpha*(1+r/r_s)**alpha)       #unnormalised density profile
            integrand = (4*np.pi*r**2)*profile                  #for normalisation

            #normalisation
            rho_s = m_vir/integrate.simps(integrand,r)
            rho = profile*rho_s
            r_central = np.where(rho >= rho_max)[0]
            rho[r_central] = rho_max

        return rho


    #plotting test for density profile
    def halo_graph(self):

        r = self.r_vir_arr
        rho = self.rho_arr

        plt.plot(r,rho,'k')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$\rho$ (M$_{\odot}$/pc$^3$)')
        plt.xlabel('r (pc)')


    #override the halo distance for a given time t, impact paramter b and halo velocity v_c
    #input parameters should be in units of ([b] = AU, [v_c] = km/s, [t] = s)
    #output units should be in [d_l] = pc
    def set_halo_dist(self,b,v_c,t):
        AU_to_km = 149e6
        km_to_pc = 1/3.0857e13

        d = np.sqrt((b*AU_to_km)**2 + (t*v_c)**2)   #[d] = km
        self.d_l = d*km_to_pc                       #[self.d_l] = pc

        return self.d_l


    #extra factor to account for halo not being a point source
    #contains ~97% of the flux from the virial radius
    #input parameters should be in units of ([d] = pc)
    def extension(self, d):

        if self.halo_profile == "ucmh":
            r_97 = 1e1*self.r_vir_arr[np.where(self.rho_arr == self.rho_arr[0])[0][-1]]
        elif self.halo_profile == "moore":
            r_97 = 1e4*self.r_vir_arr[np.where(self.rho_arr == self.rho_arr[0])[0][-1]]

        self.r_97 = r_97
        r = np.logspace(np.log10(self.r_vir*1e-7),np.log10(r_97),self.n)
        rho = self.halo_density(self.m_vir,r)

        if self.ann_or_dec == "a" or self.ann_or_dec == "ann":
            ext = integrate.simps(rho**2*r**2/(d**2 + r**2),r)/integrate.simps(rho**2*r**2/d**2,r)
        else: #decay
            ext = integrate.simps(rho*r**2/(d**2 + r**2),r)/integrate.simps(rho*r**2/d**2,r)

        return ext


    #compute j-factor for the current default halo, returned in units of ([j] = GeV^2/cm^5 for annihilation and [j] = GeV/cm^2 for decay)
    def jfact_int(self):

        Msun_to_GeV = (1.99e30)*((2.99e8)**2)*(1/1.602e-19)*(1e-9)
        pc_to_cm = (3.0857e16)*(1e2)

        if self.ann_or_dec == "a" or self.ann_or_dec == "ann":
            #annihilation
            integrand = (self.rho_arr)**2*(self.r_vir_arr)**2
            j = integrate.simps(integrand,self.r_vir_arr)

            j = j/(self.d_l**2)  #j should now be in units of M_sun^2/pc^5
            j = j*(Msun_to_GeV**2)*(pc_to_cm**-5)
        else:
            #decay
            integrand = (self.rho_arr)*(self.r_vir_arr)**2
            j = integrate.simps(integrand,self.r_vir_arr)

            j = j/(self.d_l**2)  #j should now be in units of M_sun/pc^2
            j = j*(Msun_to_GeV)*(pc_to_cm**-2)

        return j


    #returns fluence in units of ([S] = kJ/m^2)
    #input parameters should be in units of ([b] = AU, [v_c] = km/s, [del_t] = s)
    def fluence(self, b, v_c, del_t):

        pc_to_cm = (3.0857e16)*(1e2)
        spec = self.get_spectrum()
        E = spec[0]                  #energy range (pppdmcid cookbook results)
        dnde = spec[1]               #energy spectrum (pppdmid cookbook results)
        j = self.jfact_int()         #[j] = GeV^2/cm^5 or GeV/cm^2

        if self.ann_or_dec == "a" or self.ann_or_dec == "ann":
            prefac = (self.ann_cs)/(2*self.m_x**2)  #[prefac] = GeV^-2.cm^3.s^-1  (annihilation)
        else:
            prefac = (self.dec_rate)/(self.m_x)     #[prefac] = GeV^-1.s^-1  (decay)

        """calculation of time integral (includes extension factor)"""
        t_range = np.linspace(0,del_t,self.n)
        t_integrand = []

        #the distance to the clump is changing with time, so self.d_l needs to be updated and then included in the time integral (not the j-factor)
        j = j*((self.d_l*pc_to_cm)**2)  #[j] = GeV^2/cm^3

        for t in t_range:
            d = self.set_halo_dist(b,v_c,t)
            ext = self.extension(d)
            t_integrand.append(ext/((d*pc_to_cm)**2)) #update for distance as clump is moving

        E_integrand = j*dnde*E*prefac*(2*integrate.simps(np.array(t_integrand), t_range))   #*2 for del_t (defined as half time interval)

        #energy integral
        S = integrate.simps(E_integrand,E)    #[S] = GeV/cm^2

        #unit conversion
        cm_to_m = (1e-2)
        GeV_to_kJ = (1e9)*(1.602e-19)*(1e-3)

        S = S*(GeV_to_kJ)/(cm_to_m**2)
        return S


    #return flux values in an array in the form [energy, flux] in units of MeV.cm^2.s^-1
    def flux(self):

        E = self.get_spectrum()[0]      #energy range (pppdmcid cookbook results)
        dnde = self.get_spectrum()[1]   #energy spectrum (pppdmcid cookbook results)
        j = self.jfact_int()            #[j] = GeV^2/cm^5 or GeV/cm^2

        if self.ann_or_dec == "a" or self.ann_or_dec == "ann":
            prefac = (self.ann_cs)/(2*self.m_x**2)      #[prefac] = cm^3.s^-1.GeV^-2
        else:
            prefac = (self.dec_rate)/(self.m_x)         #[prefac] = GeV^-1.s^-1

        F = j*prefac*dnde*self.extension(self.d_l)      #[F] = GeV^-1.cm^-2.s^-1
        F = F*E**2                                      #[F] = GeV.cm^-2.s^-1

        #E = E*1e3                                       #[E] = MeV
        #F = F*1e3                                       #[F] = MeV.cm^-2.s^-1

        #GeV_to_erg = (1e9)*(1.602e-19)*(1e7)
        """fluence test"""
        d10 = 3600*24*10
        fluence_test = integrate.simps(F/E,E)*d10

        #unit conversion
        cm_to_m = (1e-2)
        MeV_to_KJ = (1e6)*(1.602e-19)*(1e-3)
        fluence_test = fluence_test*MeV_to_KJ*cm_to_m**(-2)

        print(r"fluence test for d_l = {:2g} pc: S = {:2g} kJ/m$^2$".format(self.d_l,fluence_test))
        """end fluence test"""

        return [E,F]

    #create simple plot of the computed flux
    def flux_plot(self,E,F):

        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)

        ax.set_xscale('log')
        ax.set_yscale('log')
#        ax.set_xlim([1e1,2e5])
#        ax.set_ylim([1e3,1e7])

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel(r'Flux (MeV cm$^{-2}$ s$^{-1}$)')
        fig.tight_layout()

        ax.plot(E,F,'k')
        #fig.savefig('fucmh_1000au_mx100_t10y.png',dpi=600)


    #saves the flux in the energy range 1e-7 -> 1e-1 GeV, as needed in the atmospheric simulations and plots flux
    def thomas_flux(self,E,F):

        write = []
        i = 0
        while (i < np.size(E)):
            write.append(str(E[i]) + ", " + str(F[i]) + "\n")
            i = i+1

        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel(r'Flux (MeV.cm$^{-2}$.s$^{-1}$)')
        ax.text(1.3e-4,0.15*np.max(F),r'm$_{\chi}$ = ' + str(self.m_x) + ' GeV'+'\n'+r'M$_h$ = ' + str(self.m_vir) + 'M$_{\odot}$')

        ax.plot(E,F,'k')
        fig.tight_layout()

        #fig.savefig(self.halo_profile+'_mx'+str(self.m_x)+'_mh'+str(self.m_vir)+'.png',dpi=600)
        data_write.write_thomas_flux(write,self.halo_profile,self.m_x)


    #returns angular resolution of a halo at distance dist in degrees
    #input parameters should be in units ([dist] = pc)
    def ang_res(self,dist):

        delta = 2*np.arctan(self.r_97/(dist))
        print(delta,delta*(180/np.pi))
        return delta


    #returns the differential rate dGamma/dt that halos encounter the Earth with a minimum fluence of 100kJ/m2
    #computation is similar to green2016 (PhysRevD), but without lensing considerations
    #d_table is table with two columns, should have units of ([mass of halo] = M_sun, [distance] = AU)
    #mu (solar masses) and sigma are parameters for halo mass distribution function
    def enc_rate(self,d_table,mu,sigma):

        """constants"""
        AU_to_pc = 1/206265
        km_to_pc = 1/3.0857e13
        f = 1                           #fraction of galactic DM contained in halos
        v_earth = 220*km_to_pc          #relative circular velocity of Earth through the galaxy, in pc/s
        rho_0 = 0.0079                  #background 'smooth' DM density at the Earth's position in galaxy, in Msun/pc^3
        numels = 1000                   #integration steps

        """interpolation for values in d_table"""
        f1 = interp1d(d_table[0],d_table[1])
        dM = np.logspace(np.log10(np.min(d_table[0])),np.log10(np.max(d_table[0])),numels)
        dnew = f1(dM)

        """integration and calculation of dgamma/dt"""
        integrand = np.zeros(numels)
        psi = np.zeros(numels)
        psioverM = np.zeros(numels)
        for i in range(numels):
            d = dnew[i]*AU_to_pc
            M = dM[i]
            if math.isnan(d): #values in table might be saved as nan which will mess with integral
                dnew[i] = 0
                d = 0
            psi[i] = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((np.log(M/mu))**2)/(2*sigma**2))
            psioverM[i] = psi[i]/M
            integrand[i] = (d**2/M**2)*psi[i]

        print("integral of psi/M = {:2g}".format(integrate.simps(psioverM,dM)))

        mass_int = integrate.simps(integrand,dM)
        dGammadt = np.pi**2*f*v_earth*rho_0*mass_int

        """curve-fitting"""
#        x = dM
#        y = dnew
#        def power_law(x,alpha,c):
#            return c*(x**alpha)
#
#        popt, pcov = curve_fit(power_law,x,y)
#        print (popt, pcov)
#        fig, ax2 = plt.subplots()
#        ax2.set_xscale('log')
#        ax2.set_yscale('log')
#        ax2.set_ylabel(r'Distance (AU)')
#        ax2.set_xlabel(r'Halo Mass (M$_\odot$)')
##        ax2.text('fitting constants')
#        ax2.plot(dM, dnew, 'r', dM, power_law(dM,*popt), 'k--')

        """plotting mass function"""
#        fig, ax = plt.subplots()
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        ax.plot(dM,psi)
#        ax.set_ylabel(r'$\Psi$ (M)')
#        ax.set_xlabel(r'M (M$_\odot$)')

        """plotting mass v distance with interpolation"""
#        fig, ax = plt.subplots()
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        ax.plot(d_table[0],d_table[1],'kx',dM,dnew,'r-')
#        ax.set_ylabel(r'Distance (AU)')
#        ax.set_xlabel(r'Halo Mass (M$_\odot$)')

        """plotting mass v distance graph with extra elements and high dpi"""
        #restrict domain to < 10^4 solar masses
#        ind = np.where(dM<=1e4)
#
#        fig, ax = plt.subplots()
#        ax.set_xscale('log')
#        ax.set_yscale('log')
#        ax.plot(dM[ind],dnew[ind],'k')
#        ax.set_ylabel(r'Distance (AU)')
#        ax.set_xlabel(r'Halo Mass (M$_\odot$)')
#        fig.tight_layout()
#        fig.savefig("distance.pdf",dpi=600)


        return dGammadt
