# MSc


Main source code for masters project 'Did Dark Matter Kill the Dinosaurs?'

___

Included files and their functions - 

  * _gamma\_setup.py_ contains function calls and plotting functionality for most results and applications of the code. It needs to be run with an input text file that has relevant parameter values (_parameters.txt_). *i.e.* to run use 

    `$ python gamma_setup.py parameters.txt` 

  * _parameters.txt_ contains the parameter input values for various dark matter and halo properties. Instructions for dimensions/units are included in the file. 

  * _dm\_annihilation.py_ contains all cosmological parameters, the class and function definitions for all physical quantities, calculations involving gamma ray fluxes from dark matter annihilation/decay and calculations describing the mass-distribution of various DM sub-halos. 

  * _data\_read.py_ & _data\_write.py_ include data manipulation tools for reading and writing spectra/results files. 

  * _environments.yml_ is the python virtual environment file.

The folder _/dnde_ contains the particle energy spectra from annihilation products (for quark channel), obtained from [PPP4DMID](http://www.marcocirelli.net/PPPC4DMID.html). 
