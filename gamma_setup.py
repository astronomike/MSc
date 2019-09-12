import numpy as np
import matplotlib.pyplot as plt
import sys

import data_read, data_write
import dm_annihilation

"""
###############################################################################################################################
read given file, and set values in parameter list to be used later

*NB - the values found in input file should be in the units:
([m_x] = GeV/c^2, [m_vir] = M_sun, [d_l] = AU)

Notes for unit clarity and any future mods:
All input values of distance (d_l) should be in units of AU (for convenience - this is the length scale for the effects I'm
currently looking at). The annihilation class itself uses units of pc (because of most values being defined using pc), and so
any input distances are converted inside the class to pc.
The function dm_annihilation.fluence(b,v_c,del_t) overrides the input value of d_l, as the distance changes over time when
calculating the total fluence over the amount of time given by del_t
###############################################################################################################################
"""
data_file = sys.argv[1]
parameters = data_read.get_data(data_file)

for i in range(np.size(parameters,0)):
    name = parameters[i][0].lower()
    value = parameters[i][1]
    if name == 'm_x':
        m_x = int(value)
    elif name == 'm_vir' or name == 'm_h':
        m_vir = float(value)
    elif name == 'channel' or name == 'ch':
        ch = value
    elif name == 'profile' or name == 'h_profile':
        h_p = value
    elif name == 'd_l' or name == 'd':
        d_l = float(value)
    elif name == 'mode' or name == 'ann_or_dec':
        ann_or_dec = value

print("density profile = {}\nm_vir = {} Msun\ndist = {} AU\nm_x = {} GeV\nchannel = {}\nmode = {}\n ".format(h_p,m_vir,d_l,m_x,ch,ann_or_dec))



"""
################################################################################################################################
initialise annihilation class (has basic parameters that will remain constant for all calculations)
"""
xx = dm_annihilation.annihilation()

#create instance of annihilation class with specific parameters given by input parameters file
par_arr = [m_x,
           m_vir,
           ch,
           h_p,
           d_l,
           ann_or_dec]
xx.setup(par_arr)

#variables needed for halo position across sky
hour = 3600
days = 24*hour
del_t = 5*days               #half time interval for clump movement ([del_t] = s)
v_c = 300                    #clump velocity ([v_c] = km/s)


"""
################################################################################################################################
calculation and plotting functions
"""

#return the distance to a halo that generates a minimum fluence of s_thresh ([s_thresh ]= kJ/m^2)
def fluence_threshold(s_thresh):
    d_index = 0
    d_before = 0
    d_arr_min = 1e-4
    d_arr_max = 1e8
    numels = 100
    d_arr = np.logspace(np.log10(d_arr_min),np.log10(d_arr_max),numels)   #[d_arr] = AU

    #increases distance to clump (using d_arr)
    #Fluence should be decreasing as distance increases
    for d in d_arr:
        S = xx.fluence(d,v_c,del_t)
        print("distance = {:2g} AU, fluence = {:2g} kJ/m^2".format(d,S))
        if (S < s_thresh):
            d_lim = d_before
            d_index += 1
            if d_before != 0.0:
                #reinitiate the array to start at the previous distance, since function is monotonic, save some computation time
                d_arr = np.logspace(np.log10(d_before),np.log10(d_arr_max),numels)
            break
        else:
            d_before = d
            continue

    return d_lim



#returns an array of the minimum impact parameter (distance) for a chosen fluence threshold, for selected WIMP masses and over a range of halo masses
#distance is returned in units of AU
def d_table(m_vir_arr,s_thresh):
    d_index = 0
    d_before = 0
    d_lim = np.zeros(np.size(m_vir_arr))

    numels = 1000
    d_arr_min = 1e-1   #AU
    d_arr_max = 1e8    #AU
    d_arr = np.logspace(np.log10(d_arr_min),np.log10(d_arr_max),numels)   #[d_arr] = AU

    for M in m_vir_arr:
        #loop for each clump virial mass
        par_arr = [m_x, M, ch, h_p, d_l, ann_or_dec]
        xx.setup(par_arr)
        for d in d_arr:
            #increases distance to clump (using d_arr)
            #Fluence should be decreasing as distance increases
            S = xx.fluence(d,v_c,del_t)
            #print("distance = {:2g} AU, fluence = {:2g} kJ/m^2".format(d,S))
            if (S < s_thresh):
                #condition if calculated fluence is less than threshold - set d_lim to the previous distance (that gave a fluence higher than the threshold)
                d_lim[d_index] = d_before
                d_index += 1
                if d_before != 0.0:
                    #reinitiate the array to start at the previous distance, since function is monotonic, save some computation time
                    d_arr = np.logspace(np.log10(d_before),np.log10(d_before*1e2),numels)
                break
            else:
                #condition if calculated fluence is greater than threshold - increase distance (decrease fluence) and try again
                d_before = d
                continue
        print("M = {:2g}, d = {:2g}".format(M,d_lim[d_index-1]))

    d_lim = np.where(d_lim == 0, np.nan,d_lim) 	#stops plotting 0 values

    return [m_vir_arr,d_lim]


#plots the minimum impact parameter for a chosen fluence threshold, for a selected WIMP mass and over a range of halo masses
def threshold_plot(m_x_arr,m_vir_arr,d_thresh,s_thresh_legend):
	plt.figure(1,figsize=(6,6))

	plt.xlabel('M$_{halo}$ (M$_{sol}$)')
	plt.ylabel('Impact Parameter (AU)')
	m_x_index = 1
	plt.text(m_vir_arr[0],(d_thresh[-1]),"channel: {}, del_t: {}s".format(ch,del_t*2),verticalalignment = "top",horizontalalignment = "left")
	for m_x in m_x_arr:
		plt.subplot(1,1,m_x_index)
		for s_thresh in s_thresh_arr:
			plt.plot(m_vir_arr,d_thresh)

		plt.legend(s_thresh_legend)
		m_x_index += 1

	plt.savefig("fluence_thresh_s10_t1000.png",dpi = 400)
	plt.show()


#find the angular size of the clump (using the r_95 radius)
def angular_plot():
    AU_to_m = 149e9
    dist = np.logspace(np.log10(1*AU_to_m),np.log10(100000*AU_to_m),N)
    delta = []

    for d in dist:
        delta.append(xx.ang_res(d) * 180/np.pi)

    plt.plot(dist/AU_to_m,delta,'b')
    plt.xlabel('distance to ucmh (AU)')
    plt.xscale('log')
    plt.ylabel('angular size (deg)')
    plt.show()



"""
################################################################################################################
function calls
"""
#useful inputs for plotting
N = 500
m_vir_first = 1
m_vir_last = 100
n = (m_vir_last-m_vir_first)/N
m_vir_arr = np.arange(m_vir_first,m_vir_last,n)
m_x_arr = [10,100,1000]
d_arr = np.logspace(np.log10(1),np.log10(1e5),N*10)      #[b_arr] = AU

#d_table(m_vir_arr,100)
#d_lim = fluence_threshold(100)
#print(d_lim)

#%%
"""flux and fluence"""
#print(d_l, xx.d_l)
#[E,F] = xx.flux()
##print(E,F)
##xx.thomas_flux(E,F)
#S = xx.fluence(d_l,v_c,del_t)
#print(r"Fluence = {:2g} kJ/m$^2$".format(S))
#xx.flux_plot(E,F)

#%%
"""encounter rate"""
Earth_age = 4.5e9*365.25*24*60*60       #age of the Earth in s
threshold = 100                      #fluence threshold in kJ/m^2
M_min = 1e-8                           
M_max = 1e12

read_or_write = "test"      #"read" or "write" or "pass"

if read_or_write == "write":
    print("write option for encounter rate")
    M = np.logspace(np.log10(M_min),np.log10(M_max),N)
    d = d_table(M,threshold)
    df = open("distance_table_s" + str(threshold) + "m" + str(m_x) + ".txt","w")
    for i in range(N):
        line = str(d[0][i]) + " " + str(d[1][i]) + "\n"
        df.write(line)
    df.close()
elif read_or_write == "read":
    print("read option for encounter rate")
    df = open("distance_table_s" + str(threshold) + "m" + str(m_x) + ".txt","r")
    lines = df.readlines()
    d = np.zeros(np.size(lines))
    M = np.zeros(np.size(lines))
    for i in range(np.size(lines)):
        values = lines[i].split(" ")
        M[i] = float(values[0])
        d[i] = float(values[1])
    df.close()

    #calculating gamma for different values of mu, sigma
#    mu_arr = [1e-6,1e-4,1e-2,1e0,1e2,1e4]
#    sigma_arr = [0.25,0.5]
#    for mu in mu_arr:
#        for sigma in sigma_arr:
#            dGammadt = xx.enc_rate([M,d],mu,sigma)
#            print("sigma = {}, mu = {:2g}: Gamma = {:2g} enc \n".format(sigma,mu,dGammadt*Earth_age))

    dGammadt = xx.enc_rate([M,d],1e-2,0.25)
    print("sigma = {}, mu = {:2g}: Gamma = {:2g} enc \n".format(0.25,1e-2,dGammadt*Earth_age))
#%%
"""plotting graphs of fluence thresholds for different WIMP masses"""
#for plotting selected values of the fluence and multiple WIMP masses
#s_thresh_legend = [r"$m_\chi$ = 10 GeV",r"$m_\chi$ = 100 GeV", r"$m_\chi$ = 1 TeV",r"$m_\chi$ = 10 TeV"]
#s_thresh_colours = ["k-","k--","k-.","k:"]

#for plotting selected values of m and multiple fluences
s_thresh_arr = [10,100,1000]
s_thresh_legend = ["10 kJ/m$^2$","100 kJ/m$^2$","1 MJ/m$^2$"]
s_thresh_colours = ["r","brown","k"]
s_thresh_linestyle = ["-","--",":"]

writing_or_plotting = "test"

if writing_or_plotting == "writing":
    #for generating and writing the data to an external file:
    d_lim_arr = fluence_threshold(m_x_arr[0],m_vir_arr,d_arr,s_thresh_arr[1])
    data_write.write_d_lim_data(m_vir_arr,d_lim_arr,s_thresh_arr[1],del_t,m_x_arr[0],xx.ann_or_dec)
elif writing_or_plotting == "plotting":
    #for reading the saved data and plotting:
    plt.figure(1,figsize=(6,6))
    plt.xlabel('$M_{halo}$ ($M_{\odot}$)')
    plt.ylabel('Impact Parameter (AU)')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.title('Minimum distance of UCMH required to produce \n gamma-ray fluence\n')
    i = 0
    m_x_arr = [1000]
    s_thresh_arr = [100]
    for m_x in m_x_arr:
        for s in s_thresh_arr:
            val = data_read.read_d_lim_data("dlimit_s" + str(s) + "_t" + str(del_t*2) + "_m" + str(m_x) + xx.ann_or_dec +".txt")
            x = np.array(val[0])
            y = np.array(val[1])
            plt.plot(x,y,color=s_thresh_colours[i],linestyle=s_thresh_linestyle[i])
            i += 1
    plt.legend(s_thresh_legend)
    plt.text(50, 49, r"$m_\chi = {}$ TeV, t = 10 days".format(10), verticalalignment = "top", horizontalalignment = "center")
    plt.savefig("fluence_thresh_sall_m10000_t10d.png",dpi = 600)
    plt.show()

#%%
"""Space for other random or temporary functions"""


#plotting mass-distance relationship for different fluence thresholds, wimp masses and halo profiles
#variables to plot over
hp = ["ucmh"]
S_arr = [100]
mx_arr = [10,100,1000]

AU_to_pc = 1/206265
M = np.logspace(np.log10(M_min),np.log10(M_max),N)  #halo mass range
distances = np.zeros((3,N))  #distance array 
r_97_arr = np.zeros((3,N))   #array for r_97 radius


#this plotting function assumes data files for distance-vs-mass are already created with the name format
#"'hp'distance_table_s'threshold'm'wimpmass'.txt"
for i in range(np.size(hp)):                #halo profile loop
    for j in range(np.size(S_arr)):         #threshold loop  
        for k in range(np.size(mx_arr)):    #wimp mass loop
            
            df = open(hp[i] + "distance_table_s" + str(S_arr[j]) + "m" + str(mx_arr[k]) + ".txt","r")
            lines = df.readlines()
    
            #loop for each halo mass value in mass-distance table.
            #change i,j,k in index for distance, r_97 array to the corresponding variable for the plot
            for m in range(np.size(lines)):
                values = lines[m].split(" ")
                distances[k,m] = float(values[1])
        
                #create array of r_97 values for plot 
                p = [mx_arr[k], M[m], ch, hp[i], d_l, ann_or_dec]
                xx.setup(p)
                xx.extension(d_l)
                r_97_arr[k,m] = xx.r_97/AU_to_pc
            df.close()

#only plot up to 1e4 solar mass halos            
ind = np.where(M<=1e4)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(M[ind],distances[0][ind],'k',M[ind],distances[1][ind],'k--',M[ind],distances[2][ind],'k:')
ax.plot(M[ind],r_97_arr[0][ind],'r',M[ind],r_97_arr[1][ind],'r--',M[ind],r_97_arr[2][ind],'r:',alpha=0.5)
#ax.fill_between(M[ind], r_97_arr[0][ind], 2e-2, facecolor='red', alpha=0.3)
ax.legend([r"m$_{\chi}$ = 10 GeV",r"m$_{\chi}$ = 100 GeV",r"m$_{\chi}$ = 1 TeV",r"< r$_{97}$"])
ax.set_ylabel(r'Distance (AU)')
ax.set_xlabel(r'Halo Mass (M$_\odot$)')
fig.tight_layout()
fig.savefig("r_97ucmhmxalls100.pdf",dpi=600)







