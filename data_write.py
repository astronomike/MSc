import numpy as np

def write_file(m_vir, del_t, out_arr):
    outfile_name = str(int(m_vir)) + "m_" + str(del_t*2) + "s_fluence.txt"
    outfile = open(outfile_name,"w")
    outfile.write("impact par (AU), fluence (kJ/m2)\n")
    outfile.writelines(out_arr)
    outfile.close()

def write_matrix(b_arr,m_vir_arr,x):
	outfile_name = "fluence_matrix_1000s.txt"
	outfile = open(outfile_name,"w")
	outfile.write("impact par (AU), halo mass (M_sun), fluence (kJ/m2)\n")

	for i in range(np.size(b_arr)):
		for j in range(np.size(m_vir_arr)):
			line = str(b_arr[j]) + ", " + str(m_vir_arr[i]) + ", " + str(x[j,i]) 
			outfile.writelines(line)

	outfile.close()
	
def change_b_to_AU(infile):
	file = open(infile,"r")	
	lines = file.read().splitlines()
	print(lines)
	new = []
	for i in range(np.size(lines)-1):
		vals = lines[i+1].split(",")
		print(vals)
		b = float(vals[0])
		print(b)
		AU = 149e6 #AU in km 
		b_AU = b/AU
		newline = str(b_AU) + ", " + vals[1] + ", " + vals[2] + "\n"
		new.append(newline)
		print(newline)

	newfile = open(infile,"w")
	newfile.writelines(new)
	file.close()
	newfile.close()

def write_d_lim_data(m_vir_arr,d_lim_arr,s_thresh,del_t,m_x,ann_or_dec):
    outfile_name = "dlimit_s" + str(s_thresh) + "_t" + str(del_t*2) + "_m" + str(m_x) + ann_or_dec +".txt"
    outfile = open(outfile_name,"w")
    outfile.write("halo mass (M_sun),impact parameter (AU)\n")
    AU = 149e6 #AU in km
    	
    for i in range(np.size(m_vir_arr)):
        line = str(m_vir_arr[i]) + "," + str(d_lim_arr[i]/AU) + "\n" 
        outfile.writelines(line)
    
    outfile.close()
	
#change_b_to_AU("fluence_matrix_1000s.txt")

def write_thomas_flux(write,profile,m_x):
    outfile_name = profile+"_m" + str(m_x) + ".txt"
    outfile = open(outfile_name,"w")
    #outfile.write("E (MeV), Flux (MeV.cm-2.s-1)\n")
    outfile.writelines(write)
    outfile.close()
