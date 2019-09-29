"""
File to create peripheral results like flux and density plots
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#Planck 2015 parameters
w_m = 0.308     #matter density parameter
w_b = 0.0495    #baryon density parameter
w_x = 0.259     #dark matter density parameter
w_l = 0.692     #lambda density parameter
h = 0.678       #normalised hubble constant

m_x = 100       #wimp mass (GeV)
ann_cs = 3e-26  #wimp annihilation cross-section

"""maximum central density calculation (for ucmh/moore profiles)"""
t = 13.76e9                             #current age of universe
ti = 59e6                               #time of matter/radiation equality (when clump starts accreting)
y_to_s = 365.25*24*60*60
GeVcm_to_Msunpc = 1/37.96               #conversion from GeV/cm^3 to M_sun/pc^3

age = (t - ti)*(y_to_s)
rho_max = m_x/(ann_cs*age)              #[rho_max] = GeV/cm^3
rho_max = rho_max*GeVcm_to_Msunpc       #[rho_max] = Msun/pc^3
print(rho_max)


def set_c_vir(m):

    m_sun = 1.0
    #assuming z = 0
    x = (w_l/w_m)**(1/3)
    y = (1e12*m_sun)/(h * m)
    [A,b,c,d] = [2.881,1.257,1.022,0.060]

    def xintegrand(x1):
        return (x1**(3/2))/((1+x1**3)**(3/2))
    i = integrate.quad(xintegrand, 0, x)

    D = (5/2)*x**(-5/2)*(1+x**3)**(1/2)*i[0]
    sigma = D * (16.9*y**0.41)/(1 + 1.102*y**0.20 + 6.22*y**0.33)
    c_vir = A*((sigma/b)**c + 1)*np.exp(d/(sigma**2))

    return c_vir

def halo_density(hp,m_vir,r_vir,c_vir,r):

    r_s = r_vir/c_vir  #scale radius

    """define density profiles"""
    if hp == "Burkert":
        r_s = r_s/1.52
        profile = 1/((1 + r/r_s)*(1 + (r/r_s)**2))        #un-normalised density profile

        #normalisation
        integrand = (4*np.pi*r**2)*profile
        rho_s = m_vir/integrate.simps(integrand,r)

        rho = rho_s*profile
        
    elif hp == "Einasto": 
        
        pass

    elif hp == "NFW":
        profile = 1/((r/r_s)*(1 + (r/r_s))**2)            #un-normalised density profile

        #normalisation
        integrand = (4*np.pi*r**2)*profile
        rho_s = m_vir/integrate.simps(integrand,r)

        rho = profile*rho_s

    elif hp == "UCMH":
        f_x = w_x/w_m
        rho = (3*f_x*m_vir)/(16*np.pi*(r_vir)**(3.0/4)*(r)**(9.0/4))

        r_central = np.where(rho >= rho_max)[0]
        rho[r_central] = rho_max

    elif hp == "Moore-like":
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

halo_profiles = ["UCMH", "Moore-like", "NFW"]
#halo_profiles = ["ucmh", "nfw"]
m_vir = 100
z = 10                                              #estimated redshift when accretion stops
n = 500
c_vir = set_c_vir(m_vir)
r_vir = (0.019)*(1e3/(1+z))*(m_vir)**(1/3)              #virial radius of ucmh/moore-like profiles
r_arr = np.logspace(np.log10(r_vir*1e-4),np.log10(r_vir),n)
rho = np.zeros((np.size(halo_profiles),n))

for i in range(np.size(halo_profiles)):
    rho[i] = halo_density(halo_profiles[i], m_vir, r_vir, c_vir, r_arr)
    
rho_1 = np.max(rho)
r_1 = np.max(r_arr)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(r_arr/r_1,rho[0]/rho_1, 'red')
ax.plot(r_arr/r_1,rho[1]/rho_1, 'darkmagenta')
ax.plot(r_arr/r_1,rho[2]/rho_1, 'blue')
#ax.plot(r_arr/r_1,rho[3]/rho_1, 'blue')
#ax.axhline(y = rho_max/rho_1, color = 'gray', ls = ':')
ax.legend(halo_profiles)
ax.set_ylabel(r'$\rho$')
ax.set_xlabel(r'radius')
ax.ticklabel_format(axis='both', style='plain')
fig.tight_layout()
#fig.savefig("profilecomparison.pdf",dpi=600)
