"""
pacmof.py
PACMOF computes partial atomic charges in Metal-Organic Framework using a Random Forest machine learning model trained on a set physically meaningful set of properties that describes the individual atom and its local environment.

Handles the primary functions
"""

#%%
def get_features_from_cif_parallel(path_to_cif):

	""" Description

	Computes the features for any given CIF file. The resultant features are updated in the output ASE atoms object under atoms.info['features'].
	The calculation is parallelized using Dask, hence, this function is recommended over the serial version, especially if the CIF file is large (>2000 atoms).

	:type path_to_cif: string
	:param path_to_cif: path to the cif file as input`
	:raises: None

	:rtype: ASE atoms object with feature array of shape (number_of_atoms, 6) updated under atoms.info['features']
	"""


	def find_nearest2(i, atoms):

		import numpy as np 
		distances = atoms.get_distances(i, slice(None), mic =True)
		distances = np.sort(distances)
		indices = np.where( distances< distances[2])[0]
		indices= indices[indices!=i] # * Remove self
		return indices.tolist(), np.mean(distances[indices])
	
	def use_correct_func(i,flags, atoms):
		# This is a temporary function, I will map this over Dask to speed up the calc. of features.
		func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}
		return func_dict[flags[i]](i, atoms)

	# def find_neighors_dask(flag):
	# 	import dask.array as da 
	# 	func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}
	# 	return func_dict[flag](i,data)
		
	# * Small Z 
	def find_neighbors_smallZ(i, atoms):

		import numpy as np 

		distances = atoms.get_distances(i, slice(None), mic =True)
		sum_radii = cov_radii[i]+cov_radii
		indices = np.where( distances< (sum_radii+0.3) )[0]
		indices= indices[indices!=i] # * Remove self
		
		return indices.tolist(), np.mean(distances[indices])

	# * Large Z 
	def find_neighbors_largeZ(i, atoms):
		
		import numpy as np
		from pymatgen.analysis.local_env import CrystalNN
		from pymatgen.io import ase 
		distances = atoms.get_distances(i, slice(None), mic =True)
		mof       = ase. AseAtomsAdaptor.get_structure(atoms=atoms)	
		nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
		local_env = nn_object.get_nn_info(mof, i)
		indices   = [local_env[index]['site_index'] for index in range(len(local_env))]
		return indices, np.mean(distances[indices])

	# * Oxygens and nitrogen	
	def find_neighbors_oxynitro(i, atoms):

		import numpy as np 

		distances = atoms.get_distances(i, slice(None), mic =True)
		sum_radii = cov_radii[i]+cov_radii
		indices = np.where( distances< (sum_radii+0.5) )[0]
		indices= indices[indices!=i] # * Remove self
		return indices.tolist(),np.mean(distances[indices])

	import numpy as np
	from pymatgen.analysis.local_env import CrystalNN
	import pandas as pd

	radius =  {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
			'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
			'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
			'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
			'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
			'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
			'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
			'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
			'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
			'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
			'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
			'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
			'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
			'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
			'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
			'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
			'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
			'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
			'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
			'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
			'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
			'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
			'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
			'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69}
	#electronegativity in pauling scale from CRC Handbook of Chemistry and Physics (For elements not having pauling electronegativity, Allred Rochow electronegativity is taken)
	electronegativity =  {'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57,
			'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,
			'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
			'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
			'Cl': 3.16, 'Ar': 0.00, 'K': 0.82, 'Ca': 1.00,
			'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
			'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
			'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
			'As': 2.01, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00,
			'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
			'Nb': 1.60, 'Mo': 2.16, 'Tc': 2.10, 'Ru': 2.20,
			'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
			'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10,
			'I': 2.66, 'Xe': 2.60, 'Cs': 0.79, 'Ba': 0.89,
			'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
			'Pm': 1.07, 'Sm': 1.17, 'Eu': 1.01, 'Gd': 1.20,
			'Tb': 1.10, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
			'Tm': 1.25, 'Yb': 1.06, 'Lu': 1.00, 'Hf': 1.30,
			'Ta': 1.50, 'W': 1.70, 'Re': 1.90, 'Os': 2.20,
			'Ir': 2.20, 'Pt': 2.20, 'Au': 2.40, 'Hg': 1.90,
			'Tl': 1.80, 'Pb': 1.80, 'Bi': 1.90, 'Po': 2.00,
			'At': 2.20, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90,
			'Ac': 1.10, 'Th': 1.30, 'Pa': 1.50, 'U': 1.70,
			'Np': 1.30, 'Pu': 1.30, 'Am': 1.30, 'Cm': 1.30}
	#First ionization energy (from CRC Handbook of Chemistry and Physics)
	first_ip =  {'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323,
			'B': 8.298, 'C': 11.260, 'N': 14.534, 'O': 13.618,
			'F': 17.423, 'Ne': 21.565, 'Na': 5.139, 'Mg': 7.646,
			'Al': 5.986, 'Si': 8.152, 'P': 10.487, 'S': 10.360,
			'Cl': 12.968, 'Ar': 15.760, 'K': 4.341, 'Ca': 6.113,
			'Sc': 6.561, 'Ti': 6.828, 'V': 6.746, 'Cr': 6.767,
			'Mn': 7.434, 'Fe': 7.902, 'Co': 7.881, 'Ni': 7.640,
			'Cu': 7.726, 'Zn': 9.394, 'Ga': 5.999, 'Ge': 7.899,
			'As': 9.789, 'Se': 9.752, 'Br': 11.814, 'Kr': 14.000,
			'Rb': 4.177, 'Sr': 5.695, 'Y': 6.217, 'Zr': 6.634,
			'Nb': 6.759, 'Mo': 7.092, 'Tc': 7.280, 'Ru': 7.360,
			'Rh': 7.459, 'Pd': 8.337, 'Ag': 7.576, 'Cd': 8.994,
			'In': 5.786, 'Sn': 7.344, 'Sb': 8.608, 'Te': 9.010,
			'I': 10.451, 'Xe': 12.130, 'Cs': 3.894, 'Ba': 5.212,
			'La': 5.577, 'Ce': 5.539, 'Pr': 5.473, 'Nd': 5.525,
			'Pm': 5.582, 'Sm': 5.644, 'Eu': 5.670, 'Gd': 6.150,
			'Tb': 5.864, 'Dy': 5.939, 'Ho': 6.021, 'Er': 6.108,
			'Tm': 6.184, 'Yb': 6.254, 'Lu': 5.426, 'Hf': 6.825,
			'Ta': 7.550, 'W': 7.864, 'Re': 7.833, 'Os': 8.438,
			'Ir': 8.967, 'Pt': 8.959, 'Au': 9.226, 'Hg': 10.437,
			'Tl': 6.108, 'Pb': 7.417, 'Bi': 7.286, 'Po': 8.414,
			'At': 9.318, 'Rn': 10.748, 'Fr': 4.073, 'Ra': 5.278,
			'Ac': 5.170, 'Th': 6.307, 'Pa': 5.890, 'U': 6.194,
			'Np': 6.266, 'Pu': 6.026, 'Am': 5.974, 'Cm': 5.991}
	#pymatgent nearest neighbor to get local enveronment
	import pymatgen as pm
	from pymatgen.io.ase import AseAtomsAdaptor 
	from ase.io import read, write
	from dask.diagnostics import ProgressBar
	# data = read(path_to_cif)
	print("Reading the CIF file...")
	data_pm = pm.Structure.from_file(path_to_cif, primitive=False)
	data = AseAtomsAdaptor.get_atoms(data_pm)
	number_of_atoms = data.get_number_of_atoms()

	cov_radii	      =np.array([radius[s] for s in data.get_chemical_symbols()])
	en_pauling        =np.array([electronegativity[s] for s in data.get_chemical_symbols()])
	ionization_energy =np.array([first_ip[s] for s in data.get_chemical_symbols()])
	
	# * Divide the atoms into different groups based on atomic number (Z) for finding the coordination shell.
	atomic_numbers = data.get_atomic_numbers()	

	# * Create a dictionary of functions for the different atomic number ranges
	bins = [0,7,9,120]
	flags = list(map(np.str, np.digitize(atomic_numbers, bins)))

	print('Computing the features...')
	import dask.bag as db
	atom_ids =list(range(number_of_atoms))
	atom_ids_db = db.from_sequence(atom_ids).map(use_correct_func, flags, data)
	with ProgressBar():
		output_from_dask = np.array(atom_ids_db.compute())
	[neighbor_list, avg_neighbor_dist] = output_from_dask[:, 0], output_from_dask[:,1]

	# func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}

	# neighbor_list, avg_neighbor_dist = zip(*[func_dict[flags[i]](i,data) for i in range(number_of_atoms)])

	# neighbor_list, avg_neighbor_dist = list(neighbor_list), list(avg_neighbor_dist)

	#* Find all the atoms with no neighbors, hopefully there aren't any such atoms.
	# * We have to use a for loop since Python's fancy indexing doesn't work so well on lists.
	nl_length = [len(nl) for nl in neighbor_list]
	no_neighbors = np.where(np.array(nl_length)==0)[0]
	# print(len(no_neighbors))
	for nn in no_neighbors:
		# print(nn)
		# temp1, temp2 = find_nearest2(nn,data)
		neighbor_list[nn], avg_neighbor_dist[nn] = find_nearest2(nn,data)

	# * We can use pandas to get values from the dictionary
	enSeries = pd.Series(electronegativity)
	ipSeries = pd.Series(first_ip)

	# * Symbols for the neighbors
	neighbor_symbols = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list] 


	average_en_shell = [np.mean(enSeries[ns].values) for ns in neighbor_symbols]
	average_ip_shell = [np.mean(ipSeries[ns].values) for ns in neighbor_symbols]

	features = np.vstack((en_pauling, ionization_energy, nl_length, avg_neighbor_dist, average_en_shell, average_ip_shell)).T
	
	data.info['features']=features
	return data # * Returns the ASE atoms object and the features array.
# %%
def get_features_from_cif_serial(path_to_cif):

	""" Description

	Computes the features for any given CIF file. The resultant features are updated in the output ASE atoms object
	under atoms.info['features']. Feature calculation is done in serial, hence, use this function only for small CIF files (<2000 atoms) if at all 
	one chooses to use it.

	:type path_to_cif: string
	:param path_to_cif: path to the cif file as input`
	:raises: None

	:rtype: ASE atoms object with feature array of shape (number_of_atoms, 6) updated under atoms.info['features']
	"""


	def find_nearest2(i, atoms):

		import numpy as np 
		distances = atoms.get_distances(i, slice(None), mic =True)
		distances = np.sort(distances)
		indices = np.where( distances< distances[2])[0]
		indices= indices[indices!=i] # * Remove self
		return indices.tolist(), np.mean(distances[indices])
	
	# def use_correct_func(i,flags, atoms):
	# 	# This is a temporary function, I will map this over Dask to speed up the calc. of features.
	# 	func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}
	# 	return func_dict[flags[i]](i, atoms)

	# def find_neighors_dask(flag):
	# 	import dask.array as da 
	# 	func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}
	# 	return func_dict[flag](i,data)
		
	# * Small Z 
	def find_neighbors_smallZ(i, atoms):

		import numpy as np 

		distances = atoms.get_distances(i, slice(None), mic =True)
		sum_radii = cov_radii[i]+cov_radii
		indices = np.where( distances< (sum_radii+0.3) )[0]
		indices= indices[indices!=i] # * Remove self
		
		return indices.tolist(), np.mean(distances[indices])

	# * Large Z 
	def find_neighbors_largeZ(i, atoms):
		
		import numpy as np
		from pymatgen.analysis.local_env import CrystalNN
		from pymatgen.io import ase 
		distances = atoms.get_distances(i, slice(None), mic =True)
		mof       = ase. AseAtomsAdaptor.get_structure(atoms=atoms)	
		nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
		local_env = nn_object.get_nn_info(mof, i)
		indices   = [local_env[index]['site_index'] for index in range(len(local_env))]
		return indices, np.mean(distances[indices])

	# * Oxygens and nitrogen	
	def find_neighbors_oxynitro(i, atoms):

		import numpy as np 

		distances = atoms.get_distances(i, slice(None), mic =True)
		sum_radii = cov_radii[i]+cov_radii
		indices = np.where( distances< (sum_radii+0.5) )[0]
		indices= indices[indices!=i] # * Remove self
		return indices.tolist(),np.mean(distances[indices])

	import numpy as np
	from pymatgen.analysis.local_env import CrystalNN
	import pandas as pd

	radius =  {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
			'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
			'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
			'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
			'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
			'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
			'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
			'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
			'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
			'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
			'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
			'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
			'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
			'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
			'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
			'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
			'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
			'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
			'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
			'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
			'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
			'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
			'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
			'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69}
	#electronegativity in pauling scale from CRC Handbook of Chemistry and Physics (For elements not having pauling electronegativity, Allred Rochow electronegativity is taken)
	electronegativity =  {'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57,
			'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,
			'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
			'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
			'Cl': 3.16, 'Ar': 0.00, 'K': 0.82, 'Ca': 1.00,
			'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
			'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
			'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
			'As': 2.01, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00,
			'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
			'Nb': 1.60, 'Mo': 2.16, 'Tc': 2.10, 'Ru': 2.20,
			'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
			'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10,
			'I': 2.66, 'Xe': 2.60, 'Cs': 0.79, 'Ba': 0.89,
			'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
			'Pm': 1.07, 'Sm': 1.17, 'Eu': 1.01, 'Gd': 1.20,
			'Tb': 1.10, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
			'Tm': 1.25, 'Yb': 1.06, 'Lu': 1.00, 'Hf': 1.30,
			'Ta': 1.50, 'W': 1.70, 'Re': 1.90, 'Os': 2.20,
			'Ir': 2.20, 'Pt': 2.20, 'Au': 2.40, 'Hg': 1.90,
			'Tl': 1.80, 'Pb': 1.80, 'Bi': 1.90, 'Po': 2.00,
			'At': 2.20, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90,
			'Ac': 1.10, 'Th': 1.30, 'Pa': 1.50, 'U': 1.70,
			'Np': 1.30, 'Pu': 1.30, 'Am': 1.30, 'Cm': 1.30}
	#First ionization energy (from CRC Handbook of Chemistry and Physics)
	first_ip =  {'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323,
			'B': 8.298, 'C': 11.260, 'N': 14.534, 'O': 13.618,
			'F': 17.423, 'Ne': 21.565, 'Na': 5.139, 'Mg': 7.646,
			'Al': 5.986, 'Si': 8.152, 'P': 10.487, 'S': 10.360,
			'Cl': 12.968, 'Ar': 15.760, 'K': 4.341, 'Ca': 6.113,
			'Sc': 6.561, 'Ti': 6.828, 'V': 6.746, 'Cr': 6.767,
			'Mn': 7.434, 'Fe': 7.902, 'Co': 7.881, 'Ni': 7.640,
			'Cu': 7.726, 'Zn': 9.394, 'Ga': 5.999, 'Ge': 7.899,
			'As': 9.789, 'Se': 9.752, 'Br': 11.814, 'Kr': 14.000,
			'Rb': 4.177, 'Sr': 5.695, 'Y': 6.217, 'Zr': 6.634,
			'Nb': 6.759, 'Mo': 7.092, 'Tc': 7.280, 'Ru': 7.360,
			'Rh': 7.459, 'Pd': 8.337, 'Ag': 7.576, 'Cd': 8.994,
			'In': 5.786, 'Sn': 7.344, 'Sb': 8.608, 'Te': 9.010,
			'I': 10.451, 'Xe': 12.130, 'Cs': 3.894, 'Ba': 5.212,
			'La': 5.577, 'Ce': 5.539, 'Pr': 5.473, 'Nd': 5.525,
			'Pm': 5.582, 'Sm': 5.644, 'Eu': 5.670, 'Gd': 6.150,
			'Tb': 5.864, 'Dy': 5.939, 'Ho': 6.021, 'Er': 6.108,
			'Tm': 6.184, 'Yb': 6.254, 'Lu': 5.426, 'Hf': 6.825,
			'Ta': 7.550, 'W': 7.864, 'Re': 7.833, 'Os': 8.438,
			'Ir': 8.967, 'Pt': 8.959, 'Au': 9.226, 'Hg': 10.437,
			'Tl': 6.108, 'Pb': 7.417, 'Bi': 7.286, 'Po': 8.414,
			'At': 9.318, 'Rn': 10.748, 'Fr': 4.073, 'Ra': 5.278,
			'Ac': 5.170, 'Th': 6.307, 'Pa': 5.890, 'U': 6.194,
			'Np': 6.266, 'Pu': 6.026, 'Am': 5.974, 'Cm': 5.991}
	#pymatgent nearest neighbor to get local enveronment
	import pymatgen as pm
	from pymatgen.io.ase import AseAtomsAdaptor 
	from ase.io import read, write
	# from dask.diagnostics import ProgressBar
	# data = read(path_to_cif)
	print("Reading the CIF file...")
	data_pm = pm.Structure.from_file(path_to_cif, primitive=False)
	data = AseAtomsAdaptor.get_atoms(data_pm)
	number_of_atoms = data.get_number_of_atoms()

	cov_radii	      =np.array([radius[s] for s in data.get_chemical_symbols()])
	en_pauling        =np.array([electronegativity[s] for s in data.get_chemical_symbols()])
	ionization_energy =np.array([first_ip[s] for s in data.get_chemical_symbols()])
	
	# * Divide the atoms into different groups based on atomic number (Z) for finding the coordination shell.
	atomic_numbers = data.get_atomic_numbers()	

	# * Create a dictionary of functions for the different atomic number ranges
	bins = [0,7,9,120]
	flags = list(map(np.str, np.digitize(atomic_numbers, bins)))

	print('Computing the features...')
	# import dask.bag as db
	# atom_ids =list(range(number_of_atoms))
	# atom_ids_db = db.from_sequence(atom_ids).map(use_correct_func, flags, data)
	# with ProgressBar():
	# 	output_from_dask = np.array(atom_ids_db.compute())
	# [neighbor_list, avg_neighbor_dist] = output_from_dask[:, 0], output_from_dask[:,1]

	func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}

	neighbor_list, avg_neighbor_dist = zip(*[func_dict[flags[i]](i,data) for i in range(number_of_atoms)])

	neighbor_list, avg_neighbor_dist = list(neighbor_list), list(avg_neighbor_dist)

	#* Find all the atoms with no neighbors, hopefully there aren't any such atoms.
	# * We have to use a for loop since Python's fancy indexing doesn't work so well on lists.
	nl_length = [len(nl) for nl in neighbor_list]
	no_neighbors = np.where(np.array(nl_length)==0)[0]
	# print(len(no_neighbors))
	for nn in no_neighbors:
		# print(nn)
		# temp1, temp2 = find_nearest2(nn,data)
		neighbor_list[nn], avg_neighbor_dist[nn] = find_nearest2(nn,data)

	# * We can use pandas to get values from the dictionary
	enSeries = pd.Series(electronegativity)
	ipSeries = pd.Series(first_ip)

	# * Symbols for the neighbors
	neighbor_symbols = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list] 


	average_en_shell = [np.mean(enSeries[ns].values) for ns in neighbor_symbols]
	average_ip_shell = [np.mean(ipSeries[ns].values) for ns in neighbor_symbols]

	features = np.vstack((en_pauling, ionization_energy, nl_length, avg_neighbor_dist, average_en_shell, average_ip_shell)).T
	
	data.info['features']=features
	return data # * Returns the ASE atoms object and the features array.
# %%
def get_charges_single_serial(path_to_cif,  create_cif=False, path_to_output_dir='.', add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):

	""" Description
	Computes the partial charges for a single CIF file and returns an ASE atoms object updated with the estimated charges 
	included as atoms.info['_atom_site_charges']. Features for each CIF is calculated in serial using Numpy. 
	Options are included for using a different pickled sklearn model and for write an output CIF with the new charges.
	
	:type path_to_cif: string
	:param path_to_cif: path to the cif file as input`

	:type create_cif: bool
	:param create_cif: whether to output a new CIF file while '_atom_site_charges' added 

	:type path_to_output_dir: string
	:param path_to_output_dir: path to the output directory for creating the new CIF file.

	:type add_string: string
	:param add_string: A string added to the filename to distinguish the output cif file from the original one.

	:type use_default_model: bool
	:param use_default_model: whether  to use the pre-trained model or not. If set to False you can set path to a different pickle file using 'path_to_pickle_obj'.

	:type path_to_pickle_obj: string
	:param path_to_pickle_obj: path to a pickle file containing the scikit-learn model one wants to use. Is used only if use_default_model is set to False.

	:raises:

	:rtype: an ase atoms object with the partial charges added as atoms.info['_atom_site_charges']
	"""

	import numpy as np 
	import joblib
	import os 
	# * Get the path of the pickle and load the model
	print("Loading the model...")
	if use_default_model==True:
		this_dir, this_filename = os.path.split(__file__)
		path_to_pickle_obj = os.path.join(this_dir, "data", "ML_Model_RF_HP_tuned.pkl")
		# print(path_to_pickle_obj)
		model = joblib.load(path_to_pickle_obj)
	else: 
		model = joblib.load(path_to_pickle_obj)
	print("Computing features...")
	data= get_features_from_cif_serial(path_to_cif)
	features = data.info['features']
	print('Estimating charges...')
	charges = model.predict(features)
	# * Adjust the charges for neutrality
	charges_adj = charges - np.sum(charges)*np.abs(charges)/np.sum(np.abs(charges))

	data.info['_atom_site_charge']=charges_adj.tolist()

	if np.any(np.abs(charges-charges_adj) > 0.2):
		print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality!")
	# if write_cif==True:

	if create_cif==True:
		print('Writing new cif file...')
		path_to_cif = os.path.abspath(list_of_cifs[i])
		old_name = os.path.basename(path_to_cif)
		new_name = old_name.split('.')[-2]+add_string+'.cif'
		# data1 = data_all[i]
		# new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
		path_to_output_dir = os.path.abspath(path_to_output_dir)
		path_to_output_cif = os.path.join(path_to_output_dir,new_name)
		write_cif(path_to_output_cif, data_all[i])
	
	return data

#%%
def get_charges_single_large(path_to_cif,  create_cif=False, path_to_output_dir='.', add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):

	""" Description
	Computes the partial charges for a single CIF file and returns an ASE atoms object updated with the estimated charges 
	included as atoms.info['_atom_site_charges']. Features for each CIF is calculated in parallel using Dask. Options are included for using a different pickled sklearn model and for write an output CIF with the new charges.
	
	:type path_to_cif: string
	:param path_to_cif: path to the cif file as input`

	:type create_cif: bool
	:param create_cif: whether to output a new CIF file while '_atom_site_charges' added 

	:type path_to_output_dir: string
	:param path_to_output_dir: path to the output directory for creating the new CIF file.

	:type add_string: string
	:param add_string: A string added to the filename to distinguish the output cif file from the original one.

	:type use_default_model: bool
	:param use_default_model: whether  to use the pre-trained model or not. If set to False you can set path to a different pickle file using 'path_to_pickle_obj'.

	:type path_to_pickle_obj: string
	:param path_to_pickle_obj: path to a pickle file containing the scikit-learn model one wants to use. Is used only if use_default_model is set to False.

	:raises:

	:rtype: an ase atoms object with the partial charges added as atoms.info['_atom_site_charges']
	"""

	import numpy as np 
	import joblib
	import os 
	# * Get the path of the pickle and load the model
	print("Loading the model...")
	if use_default_model==True:
		this_dir, this_filename = os.path.split(__file__)
		path_to_pickle_obj = os.path.join(this_dir, "data", "ML_Model_RF_HP_tuned.pkl")
		# print(path_to_pickle_obj)
		model = joblib.load(path_to_pickle_obj)
	else: 
		model = joblib.load(path_to_pickle_obj)
	print("Computing features...")
	data= get_features_from_cif_parallel(path_to_cif)
	features = data.info['features']
	print('Estimating charges...')
	charges = model.predict(features)
	# * Adjust the charges for neutrality
	charges_adj = charges - np.sum(charges)*np.abs(charges)/np.sum(np.abs(charges))

	data.info['_atom_site_charge']=charges_adj.tolist()

	if np.any(np.abs(charges-charges_adj) > 0.2):
		print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality!")
	# if write_cif==True:

	if create_cif==True:
		print('Writing new cif file...')
		path_to_cif = os.path.abspath(list_of_cifs[i])
		old_name = os.path.basename(path_to_cif)
		new_name = old_name.split('.')[-2]+add_string+'.cif'
		# data1 = data_all[i]
		# new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
		path_to_output_dir = os.path.abspath(path_to_output_dir)
		path_to_output_cif = os.path.join(path_to_output_dir,new_name)
		write_cif(path_to_output_cif, data_all[i])
	
	return data
#%%
def write_cif(fileobj, images, format='default'):

	""" Description
	This is a clone of the ASE's write_cif funtion from the ase.io.cif module. It is modified so as to write the '_atom_site_charge' also
	while writing the CIF file.

	:type fileobj: string/file handle
	:param fileobj:path string or file handle to the output CIF

	:type images: ASE atoms object
	:param images: the atoms object you want to write to CIF format.

	:type format: string
	:param format: Some option found within the original function. Refer to ASE's documentation for more info.

	:raises:

	:rtype: None. Just writs the file.
	"""
	

	def write_enc(fileobj, s):
		"""Write string in latin-1 encoding."""
		fileobj.write(s.encode("latin-1"))

	from ase.utils import basestring
	from ase.parallel import paropen
	# from ase.io import cif

	"""Write *images* to CIF file."""
	if isinstance(fileobj, basestring):
		fileobj = paropen(fileobj, 'wb')

	if hasattr(images, 'get_positions'):
		images = [images]

	for i, atoms in enumerate(images):
		write_enc(fileobj, 'data_image%d\n' % i)

		a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

		if format == 'mp':

			comp_name = atoms.get_chemical_formula(mode='reduce')
			sf = split_chem_form(comp_name)
			formula_sum = ''
			ii = 0
			while ii < len(sf):
				formula_sum = formula_sum + ' ' + sf[ii] + sf[ii + 1]
				ii = ii + 2

			formula_sum = str(formula_sum)
			write_enc(fileobj, '_chemical_formula_structural       %s\n' %
					  atoms.get_chemical_formula(mode='reduce'))
			write_enc(fileobj, '_chemical_formula_sum      "%s"\n' %
					  formula_sum)

		# Do this only if there's three non-zero lattice vectors
		if atoms.number_of_lattice_vectors == 3:
			write_enc(fileobj, '_cell_length_a       %g\n' % a)
			write_enc(fileobj, '_cell_length_b       %g\n' % b)
			write_enc(fileobj, '_cell_length_c       %g\n' % c)
			write_enc(fileobj, '_cell_angle_alpha    %g\n' % alpha)
			write_enc(fileobj, '_cell_angle_beta     %g\n' % beta)
			write_enc(fileobj, '_cell_angle_gamma    %g\n' % gamma)
			write_enc(fileobj, '\n')

			write_enc(fileobj, '_symmetry_space_group_name_H-M    %s\n' %
					  '"P 1"')
			write_enc(fileobj, '_symmetry_int_tables_number       %d\n' % 1)
			write_enc(fileobj, '\n')

			write_enc(fileobj, 'loop_\n')
			write_enc(fileobj, '  _symmetry_equiv_pos_as_xyz\n')
			write_enc(fileobj, "  'x, y, z'\n")
			write_enc(fileobj, '\n')

		write_enc(fileobj, 'loop_\n')

		# Is it a periodic system?
		coord_type = 'fract' if atoms.pbc.all() else 'Cartn'

		if format == 'mp':
			write_enc(fileobj, '  _atom_site_type_symbol\n')
			write_enc(fileobj, '  _atom_site_label\n')
			write_enc(fileobj, '  _atom_site_symmetry_multiplicity\n')
			write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_occupancy\n')
		else:
			write_enc(fileobj, '  _atom_site_label\n')
			write_enc(fileobj, '  _atom_site_occupancy\n')
			write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
			write_enc(fileobj, '  _atom_site_thermal_displace_type\n')
			write_enc(fileobj, '  _atom_site_B_iso_or_equiv\n')
			write_enc(fileobj, '  _atom_site_type_symbol\n')
			write_enc(fileobj, '  _atom_site_charge\n')
			
		if coord_type == 'fract':
			coords = atoms.get_scaled_positions().tolist()
			charges =  atoms.info['_atom_site_charge']
		else:
			coords = atoms.get_positions().tolist()
		symbols = atoms.get_chemical_symbols()
		occupancies = [1 for i in range(len(symbols))]
	
		charges =  atoms.info['_atom_site_charge']
		# try to fetch occupancies // rely on the tag - occupancy mapping
		try:
			occ_info = atoms.info['occupancy']
			
			for i, tag in enumerate(atoms.get_tags()):
				occupancies[i] = occ_info[tag][symbols[i]]
				# extend the positions array in case of mixed occupancy
				for sym, occ in occ_info[tag].items():
					if sym != symbols[i]:
						symbols.append(sym)
						coords.append(coords[i])
						occupancies.append(occ)
		except KeyError:
			pass

		no = {}

		for symbol, pos, occ, charge in zip(symbols, coords, occupancies, charges):
			if symbol in no:
				no[symbol] += 1
			else:
				no[symbol] = 1
			if format == 'mp':
				write_enc(fileobj,
						  '  %-2s  %4s  %4s  %7.5f  %7.5f  %7.5f  %6.1f %6.1f\n' %
						  (symbol, symbol + str(no[symbol]), 1,
						   pos[0], pos[1], pos[2], occ, charge))
			else:
				write_enc(fileobj,
						  '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %s  %6.4f\n'
						  % ('%s%d' % (symbol, no[symbol]),
							 occ,
							 pos[0],
							 pos[1],
							 pos[2],
							 'Biso',
							 1.0,
							 symbol, charge))
	return None		
# %%
def get_charges_multiple_onebyone(list_of_cifs,  create_cif=False, path_to_output_dir='.', add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):

	""" Description
	Compute the partial charges in a list of CIFs. This function saves time by loading the Random forest model only once. Each of the individual calculatiions
	are parallelized and this function is made for when the CIF files involved are very large (>2000 atoms). The files will be processed one by one but each calculation will be parallelized for speed.

	:type list_of_cifs: list
	:param list_of_cifs: a list of paths to all the cif files to compute the charges for.

	:type create_cif: bool
	:param create_cif: Tells whether to create a new CIF with the 'add_string' added to the filename, that includes the new estimated _atom_site_charges

	:type path_to_output_dir:string (path)
	:param path_to_output_dir: Where to create the output cif file.

	:type add_string: string
	:param add_string: A string added to the filename to distinguish the output cif file from the original one.

	:type use_default_model: bool
	:param use_default_model: whether  to use the pre-trained model or not. If set to False you can set path to a different pickle file using 'path_to_pickle_obj'.

	:type path_to_pickle_obj: string
	:param path_to_pickle_obj: path to a pickle file containing the scikit-learn model one wants to use. Is used only if use_default_model is set to False.

	:raises:

	:rtype:  a list of ase atoms objects with charges added as atoms.info['_atom_site_charges'] to each atoms object
	"""
	
	from tqdm import tqdm
	import numpy as np 
	import joblib
	import os 

	def adjust_charge(charges):
		import numpy as np
		return charges - np.sum(charges)*np.abs(charges)/np.sum(np.abs(charges))

	# * Get the path of the pickle and load the model
	print("Loading the model...")
	if use_default_model==True:
		this_dir, this_filename = os.path.split(__file__)
		path_to_pickle_obj = os.path.join(this_dir, "data", "ML_Model_RF_HP_tuned.pkl")
		# print(path_to_pickle_obj)
		model = joblib.load(path_to_pickle_obj)
	else: 
		model = joblib.load(path_to_pickle_obj)

	print('Calculating the features for all the files...')
	data_all= [get_features_from_cif_parallel(l) for l in tqdm(list_of_cifs)]
	# data_all = list(data_all)
	charges_all = [model.predict(d.info['features']) for d in data_all]
	# print(len(data_all), len(features_all), len(charges_all))
	print('Estimating charges for all the files...')
	for i in range(len(list_of_cifs)):
			charges_adj = adjust_charge(charges_all[i])
			data_all[i].info['_atom_site_charge'] = charges_adj
			if np.any(np.abs(charges_all[i]-charges_adj) > 0.2):
				print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality in file\n"+list_of_cifs[i])
			
	# * Write the output cif files
	if create_cif==True:
		print('Writing new cif file...') 
		for i in range(len(list_of_cifs)):
			# print(i)
			path_to_cif = os.path.abspath(list_of_cifs[i])
			old_name = os.path.basename(path_to_cif)
			new_name = old_name.split('.')[-2]+add_string+'.cif'
			# data1 = data_all[i]
			# new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
			path_to_output_dir = os.path.abspath(path_to_output_dir)
			path_to_output_cif = os.path.join(path_to_output_dir,new_name)
			write_cif(path_to_output_cif, data_all[i])
	print("Done!")
	return data_all

# %%
def get_charges_multiple_parallel(list_of_cifs,  create_cif=False, path_to_output_dir='.', add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):

	""" Description
	Compute the partial charges in a list of CIFs. This function saves time by loading the Random forest model only once. This function is embarassingly parallel 
	over different files using Dask (both multiprocessing or on an HPC). Recommended for use in high-throuput applications involving CIF files of less than 2000 atoms. All files 
	will be processed in parallel but the individual calculations will be done on a sinlge CPU using numpy, which is fast for most CIFs.

	:type list_of_cifs: list
	:param list_of_cifs: a list of paths to all the cif files to compute the charges for.

	:type create_cif: bool
	:param create_cif: Tells whether to create a new CIF with the 'add_string' added to the filename, that includes the new estimated _atom_site_charges

	:type path_to_output_dir:string (path)
	:param path_to_output_dir: Where to create the output cif file.

	:type add_string: string
	:param add_string: A string added to the filename to distinguish the output cif file from the original one.

	:type use_default_model: bool
	:param use_default_model: whether  to use the pre-trained model or not. If set to False you can set path to a different pickle file using 'path_to_pickle_obj'.

	:type path_to_pickle_obj: string
	:param path_to_pickle_obj: path to a pickle file containing the scikit-learn model one wants to use. Is used only if use_default_model is set to False.

	:raises:

	:rtype:  a list of ase atoms objects with charges added as atoms.info['_atom_site_charges'] to each atoms object
	"""

	
	from tqdm import tqdm
	import numpy as np 
	import joblib
	import os 

	def adjust_charge(charges):
		import numpy as np
		return charges - np.sum(charges)*np.abs(charges)/np.sum(np.abs(charges))

	import dask.bag as db 
	from dask.diagnostics import ProgressBar
	# * Get the path of the pickle and load the model
	print("Loading the model...")
	if use_default_model==True:
		this_dir, this_filename = os.path.split(__file__)
		path_to_pickle_obj = os.path.join(this_dir, "data", "ML_Model_RF_HP_tuned.pkl")
		# print(path_to_pickle_obj)
		model = joblib.load(path_to_pickle_obj)
	else: 
		model = joblib.load(path_to_pickle_obj)

	print('Calculating the features for all the files...')
	
	data_db = db.from_sequence(list_of_cifs).map(get_features_from_cif_serial)
	with ProgressBar():
		data_all = data_db.compute()
	# data_all= zip(*[get_features_from_cif(l) for l in tqdm(list_of_cifs)])
	
	
	data_all = list(data_all)
	charges_all = [model.predict(d.info['features']) for d in data_all]
	# print(len(data_all), len(features_all), len(charges_all))
	print('Estimating charges for all the files...')
	for i in range(len(list_of_cifs)):
		charges_adj = adjust_charge(charges_all[i])
		data_all[i].info['_atom_site_charge'] = charges_adj
		if np.any(np.abs(charges_all[i]-charges_adj) > 0.2):
			print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality in file\n"+list_of_cifs[i])

	# * Write the output cif files
	if create_cif==True:
		print('Writing new cif file...') 
		for i in range(len(list_of_cifs)):
			# print(i)
			path_to_cif = os.path.abspath(list_of_cifs[i])
			old_name = os.path.basename(path_to_cif)
			new_name = old_name.split('.')[-2]+add_string+'.cif'
			# data1 = data_all[i]
			# new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
			path_to_output_dir = os.path.abspath(path_to_output_dir)
			path_to_output_cif = os.path.join(path_to_output_dir,new_name)
			write_cif(path_to_output_cif, data_all[i])
	print("Done!")
	return data_all

	
	# if compute==True:
	# 	return charges_db.compute()
	# else:
	# 	return charges_db

if __name__ == "__main__":
	
	# Do something if this file is invoked on its own
	"""
	This is for calling the module by its name, perhaps from the terminal.
	I have to decide if we want to keep this and how to use it if we do -- Arun.
	"""
	print(canvas())
