import numpy as np

def get_data(datafile):
	
        with open(datafile,"r") as df:
            parameters = []            
            lines = df.read().splitlines()
            
            for i in range(np.size(lines)):
                if (lines[i] == "" or lines[i][0] == "#" or lines[i][0] == "\n"):
                    continue
                else:
                    parameters.append(lines[i].split(" = "))
            
        return parameters
	
def read_d_lim_data(datafile):
	
	with open(datafile,"r") as df:
		lines = df.read().splitlines()
		m_vir_arr = []
		d_lim = []
		
		for i in range(np.size(lines)-1):
			values = lines[i+1].split(",")
			m_vir_arr.append(float(values[0]))
			d_lim.append(float(values[1]))
			
		return [m_vir_arr,d_lim]