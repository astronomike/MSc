"""
File to create peripheral results like flux and density plots
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import dm_annihilation

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
#print(rho_max)


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

def plot_halocomparison():
    
    halo_profiles = ["UCMH", "Moore-like"]
    #halo_profiles = ["ucmh", "nfw"]
    m_vir = 100
    z = 10                                              #estimated redshift when accretion stops
    n = 500
    c_vir = set_c_vir(m_vir)
    r_vir = (0.019)*(1e3/(1+z))*(m_vir)**(1/3)              #virial radius of ucmh/moore-like profiles
    r_arr = np.logspace(np.log10(r_vir*1e-5),np.log10(r_vir),n)
    rho = np.zeros((np.size(halo_profiles),n))
    
    for i in range(np.size(halo_profiles)):
        rho[i] = halo_density(halo_profiles[i], m_vir, r_vir, c_vir, r_arr)
        
    rho_1 = np.max(rho)
    r_1 = np.max(r_arr)
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.axhline(y = rho_max/rho_1, color = 'black', linewidth = 0.9, alpha = 1, ls = ':', label='_nolegend_')
    
    ax.plot(r_arr/r_1,rho[0]/rho_1, 'k', linestyle = "-")  #ucmh
    ax.plot(r_arr/r_1,rho[1]/rho_1, 'k', linestyle = "--")    #moore-like
    #ax.plot(r_arr/r_1,rho[2]/rho_1, 'black', linestyle = "-.")     #nfw
    #ax.plot(r_arr/r_1,rho[3]/rho_1, 'blue')
    #ax.set_ylim([1e-11,2e0])

    ax.text(0.4,4e-1,r"$\rho_{max}$")
    ax.legend(halo_profiles, loc = 0, fontsize=12)    
    ax.set_ylabel(r'$\rho\, /\, \rho_{\mathrm{max}}$',fontsize=14)
    ax.set_xlabel(r'$r\, /\, R_{\mathrm{UCMH}}$',fontsize=14)
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig.tight_layout()
    plt.show()
    fig.savefig("profilecomparison_full.pdf",dpi=300)

plot_halocomparison()

"""
###############################################################################
Plots of annihilation spectra from dnde folder
"""

#inputs = (ch = channel string, m = WIMP mass)
def get_spectrum(ch,m):

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

def plot_spectra():
    
    ch_list = ['q','b','e']    
    m_list = [10,100,1000]
    
    #create main subplot
    fig, ax = plt.subplots(1,3, figsize=(15,6) ,sharey='row', subplot_kw=dict(xscale='log', yscale='log', xlim=(1e-4,1e2), ylim=(1e-1,1e2)))
    
    #removing labels and tick labels for inner axes 
    fig.add_subplot(111, frameon=False)
    plt.grid(False)   
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    
    plt.xlabel(r'Energy (GeV)', fontsize=18, labelpad=10)    #labelpad for some extra spacing between axes label and tick labels
    plt.ylabel(r'dN$_{\gamma}$/dE (GeV$^{-1}$)', fontsize=18, labelpad=10)
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    """
    ax[0].set_xlim(1e-4,1e2)
    ax[1].set_xlim(1e-4,1e2)
    ax[2].set_xlim(1e-4,1e2)
    """
    colours = ["blue", "deeppink", "forestgreen"]
    linestyles = ["--", "-", "-."]
    channels = [r"$q\bar{q}$",r"$b\bar{b}$",r"$e^{\!+}\!e^{\!-}\!$"]
    titles = [r"$m_{\chi} = 10$ GeV",r"$m_{\chi} = 100$ GeV",r"$m_{\chi} = 1000$ GeV"]
    
    #get the data and plot each one
    for j in range(np.size(m_list)):
        for i in range(np.size(ch_list)):
            spec = get_spectrum(ch_list[i],m_list[j])
            E = spec[0]
            dnde = spec[1]
            ax[j].plot(E,dnde,color=colours[i],linewidth=2,linestyle=linestyles[i])
            ax[j].legend(channels,fontsize=13)
            ax[j].set_title(titles[j],fontsize=18)
    
    plt.tight_layout()
    plt.show()
    #fig.savefig("spectra.pdf",dpi=300, bbox_inches='tight')

#plot_spectra()
    

def plot_flux():
    
    ch_list = ['q','b','e']    
    m_list = [10,100,1000]
    xx = dm_annihilation.annihilation()
  
    #create main subplot
    fig, ax = plt.subplots(1,3, figsize=(15,6) ,sharey='row', subplot_kw=dict(xscale='log', yscale='log'))
    
    #removing labels and tick labels for inner axes 
    fig.add_subplot(111, frameon=False)
    plt.grid(False)   
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    
    plt.xlabel(r'Energy (GeV)', fontsize=18, labelpad=10)    #labelpad for some extra spacing between axes label and tick labels
    plt.ylabel(r'd$\Phi$/dE (GeV cm$^{-2}$ s$^{-1}$)', fontsize=18, labelpad=10)
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    """
    ax[0].set_xlim(1e-4,1e2)
    ax[1].set_xlim(1e-4,1e2)
    ax[2].set_xlim(1e-4,1e2)
    """
    colours = ["blue", "deeppink", "forestgreen"]
    linestyles = ["--", "-", "-."]
    channels = [r"$q\bar{q}$",r"$b\bar{b}$",r"$e^{\!+}\!e^{\!-}\!$"]
    titles = [r"$m_{\chi} = 10$ GeV",r"$m_{\chi} = 100$ GeV",r"$m_{\chi} = 1000$ GeV"]
    
    #get the data and plot each one
    for j in range(np.size(m_list)):
        for i in range(np.size(ch_list)):
            #spec = get_spectrum(ch_list[i],m_list[j])
            #E = spec[0]
            #dnde = spec[1]
            par_arr = [m_list[j],100,ch_list[i],'moore',1000,'ann']
            xx.setup(par_arr)          
            [E,F] = xx.flux()
            
            ax[j].plot(E,F,color=colours[i],linewidth=2,linestyle=linestyles[i])
            ax[j].legend(channels,fontsize=13,loc=2)
            ax[j].set_title(titles[j],fontsize=18)
            #ax[j].set_ylim([np.max(F)*1e-4,np.max(F)*1e2])
            ax[j].set_ylim([1e-2,3e2])
            ax[j].set_xlim([np.max(E)*1e-5,np.max(E)])
            
    plt.tight_layout()
    plt.show()
    fig.savefig("fluxes_moore.pdf",dpi=300, bbox_inches='tight')

#plot_flux()


"""
#########################################################################################################
UV absorption cross section
"""

def plot_ozoneapsorption():
    lambda_ozone = np.array([238.27, 245.8, 253.7, 263.6, 272.0902, 281.3286, 289.3598, 296.7284, 302.1500, 314.1332, 322.0719, 334.24, 344.35])
    sigma = np.array([7.51e-18, 1.006e-17, 1.154e-17, 9.66e-18, 6.70e-18, 3.31e-18, 1.402e-18, 5.60e-19, 2.65e-19, 4.86e-20, 2.15e-20, 3.11e-21, 0.974e-21])
    
    fig, ax = plt.subplots(figsize=(5,4))
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(lambda_ozone,sigma, 'k')    
    ax.set_ylabel(r'$\sigma_{\mathrm{abs}} \, (\mathrm{cm^2})$',fontsize=14)
    ax.set_xlabel("Wavelength (nm)",fontsize=14)
    ax.axvspan(280, 315, alpha=0.3, color='blue')
    ax.text(292,1e-21,"UV-B",fontsize=12)
    ax.text(322,1e-21,"UV-A",fontsize=12)
    ax.text(260,1e-21,"UV-C",fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig("ozone_absorption.pdf",dpi=300)
    
#plot_ozoneapsorption()


"""
#########################################################################################################
extension factor comparison
"""

def extension(): 
    
    
    AU_to_km = 149e6
    km_to_pc = 1/3.0857e13
    xx = dm_annihilation.annihilation()
    
    mx = 100    #WIMP mass in GeV
    M = 100     #halo mass in solar masses
    ch = 'q'    #channel (kinda irrelevant here)
    mode = 'ann'
    halo_profiles = ["ucmh", "moore"]
    d_arr = np.logspace(np.log10(1),np.log(100),1000)*AU_to_km*km_to_pc
    ext = np.zeros(np.size(d_arr))
   
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xscale('log')
    ax.set_yscale('log')   
    ax.axhline(y = 1, color = 'black', linewidth = 0.9, alpha = 1, ls = ':', label='_nolegend_')
    
    plt.rcParams["mathtext.fontset"] = "cm" #to get the right looking epsilon
    ax.set_ylabel(r'Extension $\epsilon$',fontsize=14)
    ax.set_xlabel('Distance (AU)',fontsize=14)   
    colours = ['k','k'] 
    linestyles = ['-', '--']
    j = 0
    for hp in halo_profiles:
         p = [mx,M,ch,hp,1,mode] 
         xx.setup(p)   
         for i in range(np.size(d_arr)):
             #print(d_arr[i])
             ext[i] = xx.extension(d_arr[i])
         ax.plot(d_arr*1/(AU_to_km*km_to_pc),ext, color=colours[j],linestyle=linestyles[j])
         j = j+1

    ax.legend(["UCMH", "Moore-like"], loc=(0.65,0.6))
    fig.tight_layout()
    plt.show()
    fig.savefig("extension.pdf",dpi=300)

#extension()
