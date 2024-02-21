"""pacmof.py PACMOF computes partial atomic charges in Metal-Organic Framework using a Random Forest machine learning
model trained on a set physically meaningful set of properties that describes the individual atom and its local
environment.

Handles the primary functions
"""


# %%
def get_features_from_cif_parallel(path_to_cif, client_name='dummy'):
    """ Description

	Computes the features for any given CIF file. The resultant features are updated in the output ASE atoms object under atoms.info['features'].
	The calculation is parallelized using Dask, hence, this function is recommended over the serial version, especially if the CIF file is large (>2000 atoms).

	:param path_to_cif: path to the cif file as input`
	:type path_to_cif: string

	:param client_name: Used to run these calculations on a predefined dask cluster.
	:type client_name: Client object from dask distributed

	:raises: None

	:rtype: ASE atoms object with feature array of shape (number_of_atoms, 6) updated under atoms.info['features']
	"""

    # %%
    def find_nearest2(i, atoms):

        import numpy as np
        distances = atoms.get_distances(i, slice(None), mic=True)
        distances = np.sort(distances)
        indices = np.where(distances < distances[2])[0]
        indices = indices[indices != i]  # * Remove self
        return indices.tolist(), np.mean(distances[indices])

    # %%
    # def use_correct_func(i,flags1, atoms):
    # 	# This is a temporary function, I will map this over Dask to speed up the calc. of features.
    # %%
    # * Small Z
    def find_neighbors_smallZ(i, atoms_dict):
        import numpy as np
        from ase.data import covalent_radii
        from ase import Atoms
        # struct = Structure.from_dict(struct_dict.result())  # pymatgen structure from dict
        # atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        atoms = Atoms.fromdict(atoms_dict)
        cov_radii = covalent_radii[atoms.get_atomic_numbers()]
        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii[i] + cov_radii
        indices = np.where(distances < (sum_radii + 0.3))[0]
        indices = indices[indices != i]  # * Remove self

        return indices.tolist(), np.mean(distances[indices])

    # %%
    # * Small Z
    def find_neighbors_smallZ2(i, struct):
        import numpy as np
        from ase.data import covalent_radii
        from pymatgen.io import ase
        # from pymatgen import Structure
        # struct = Structure.from_dict(struct_dict.get())  # pymatgen structure from dict
        atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        cov_radii = covalent_radii[atoms.get_atomic_numbers()]
        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii[i] + cov_radii
        indices = np.where(distances < (sum_radii + 0.3))[0]
        indices = indices[indices != i]  # * Remove self
        return indices.tolist(), np.mean(distances[indices])

    # %%
    # def smallZ_many(chunk, atoms):
    # 	# np.apply_along_axis(find_neighbors_smallZ, axis=1, arr=chunk, atoms=atoms)
    # 	# return np.array([find_neighbors_smallZ(x,atoms) for x in chunk], dtype='object')
    # 	return np.array(np.apply_along_axis(func1d=find_neighbors_smallZ, axis=1, atoms=data, arr=chunk), dtype='object')
    # def oxynitro_many(chunk, atoms):
    # 	# return np.array([find_neighbors_oxynitro(x, atoms) for x in chunk], dtype='object')
    # 	return np.array(np.apply_along_axis(func1d=find_neighbors_oxynitro,  axis=1, atoms=data, arr=chunk), dtype='object')
    # def largeZ_many(chunk, mof_pm, atoms):
    # 	return np.array(np.apply_along_axis(func1d=find_neighbors_largeZ,  axis=1, mof_pm=mof_pm, atoms=data, arr=chunk), dtype='object')
    # return np.array([find_neighbors_largeZ(x, mof_pm, atoms) for x in chunk], dtype='object')
    # %%
    # * Large Z
    def find_neighbors_largeZ(i,struct_dict, atoms_dict):
        import numpy as np
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen import Structure
        struct = Structure.from_dict(struct_dict)  # pymatgen structure object
        # atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        # Ignore pymatgen warnings
        from ase import Atoms
        # struct = Structure.from_dict(struct_dict.result())  # pymatgen structure from dict
        # atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        atoms = Atoms.fromdict(atoms_dict)
        distances = atoms.get_distances(i, slice(None), mic=True)
        import warnings
        warnings.filterwarnings("ignore")
        nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
        local_env = nn_object.get_nn_info(mof_pm, n=np.int(i))
        indices = []
        for index in range(len(local_env)):
            indices.append(local_env[index]['site_index'])
        return indices, np.mean(distances[indices])

    # %%
    def find_neighbors_largeZ2(i, struct):
        import numpy as np
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.io import ase
        # from pymatgen import Structure
        # struct = Structure.from_dict(struct_dict.get())  # pymatgen structure object
        atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        distances = atoms.get_distances(i, slice(None), mic=True)
        # Ignore pymatgen warnings
        import warnings
        warnings.filterwarnings("ignore")
        nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
        local_env = nn_object.get_nn_info(mof_pm, n=np.int(i))
        indices = []
        for index in range(len(local_env)):
            indices.append(local_env[index]['site_index'])
        return indices, np.mean(distances[indices])

    # %%
    # * Oxygens and nitrogen
    def find_neighbors_oxynitro(i,atoms_dict):
        import numpy as np
        from ase.data import covalent_radii
        from ase import Atoms
        # struct = Structure.from_dict(struct_dict.result())  # pymatgen structure from dict
        # atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        atoms = Atoms.fromdict(atoms_dict)
        # struct = Structure.from_dict(struct_dict)  # pymatgen structure object
        # atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        cov_radii_ = covalent_radii[atoms.get_atomic_numbers()]
        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii_[i] + cov_radii_
        indices = np.where(distances < (sum_radii + 0.5))[0]
        indices = indices[indices != i]  # * Remove self
        return indices.tolist(), np.mean(distances[indices])

    # %%
    # * Oxygens and nitrogen
    def find_neighbors_oxynitro2(i, struct):
        import numpy as np
        from ase.data import covalent_radii
        from pymatgen.io import ase
        # from pymatgen import Structure
        # struct = Structure.from_dict(struct_dict.get())  # pymatgen structure object
        atoms = ase.AseAtomsAdaptor.get_atoms(structure=struct)  # ase atoms object
        cov_radii_ = covalent_radii[atoms.get_atomic_numbers()]
        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii_[i] + cov_radii_
        indices = np.where(distances < (sum_radii + 0.5))[0]
        indices = indices[indices != i]  # * Remove self
        return indices.tolist(), np.mean(distances[indices])

    # %%
    # import numpy as np
    # # func_dict = {'1': find_neighbors_smallZ, '2': find_neighbors_oxynitro, '3': find_neighbors_largeZ}
    # if np.array(flags1)[i] == 1:
    # 	return find_neighbors_smallZ(i, atoms)
    # elif np.array(flags1)[i] == 2:
    # 	return find_neighbors_oxynitro(i, atoms)
    # elif np.array(flags1)[i] == 3:
    # 	return find_neighbors_largeZ(i, atoms)
    # else:
    # 	return np.NaN

    # def find_neighors_dask(flag):
    # 	import dask.array as da
    # 	func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}
    # 	return func_dict[flag](i,data)
    # %%
    # # * Small Z
    # def find_neighbors_smallZ(i, atoms):
    #
    # 	import numpy as np
    # 	from ase.data import  covalent_radii
    # 	cov_radii = covalent_radii[atoms.get_atomic_numbers()]
    # 	distances = atoms.get_distances(i, slice(None), mic =True)
    # 	sum_radii = cov_radii[i]+cov_radii
    # 	indices = np.where( distances< (sum_radii+0.3) )[0]
    # 	indices= indices[indices!=i] # * Remove self
    #
    # 	return indices.tolist(), np.mean(distances[indices])
    #
    # # * Large Z
    # def find_neighbors_largeZ(i, atoms):
    #
    # 	import numpy as np
    # 	from pymatgen.analysis.local_env import CrystalNN
    # 	from pymatgen.io import ase
    # 	distances = atoms.get_distances(i, slice(None), mic =True)
    # 	mof       = ase. AseAtomsAdaptor.get_structure(atoms=atoms)
    # 	nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
    # 	local_env = nn_object.get_nn_info(mof, i)
    # 	indices   = [local_env[index]['site_index'] for index in range(len(local_env))]
    # 	return indices, np.mean(distances[indices])
    #
    # # * Oxygens and nitrogen
    # def find_neighbors_oxynitro(i, atoms):
    #
    # 	import numpy as np
    # 	from ase.data import  covalent_radii
    # 	cov_radii = covalent_radii[atoms.get_atomic_numbers()]
    # 	distances = atoms.get_distances(i, slice(None), mic =True)
    # 	sum_radii = cov_radii[i]+cov_radii
    # 	indices = np.where( distances< (sum_radii+0.5) )[0]
    # 	indices= indices[indices!=i] # * Remove selfe
    # 	return indices.tolist(),np.mean(distances[indices])
    # %%
    import numpy as np
    import pandas as pd

    radius = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
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
    # electronegativity in pauling scale from CRC Handbook of Chemistry and Physics (For elements not having pauling electronegativity, Allred Rochow electronegativity is taken)
    electronegativity = {'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57,
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
    # First ionization energy (from CRC Handbook of Chemistry and Physics)
    first_ip = {'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323,
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
    # pymatgent nearest neighbor to get local enveronment
    # data = read(path_to_cif)
    print("Reading  CIF file {}...".format(path_to_cif))
    # data_pm = pm.Structure.from_file(path_to_cif, primitive=False)
    from ase.io import read
    from pymatgen.io import ase
    data = read(path_to_cif)  # Read using ASE, usually faster than pymatgen
    mof_pm = ase.AseAtomsAdaptor.get_structure(atoms=data)  # Pymatgen structure file
    # data = AseAtomsAdaptor.get_atoms(data_pm)

    cov_radii = np.array([radius[s] for s in data.get_chemical_symbols()])
    en_pauling = np.array([electronegativity[s] for s in data.get_chemical_symbols()])
    ionization_energy = np.array([first_ip[s] for s in data.get_chemical_symbols()])
    # %%
    number_of_atoms = data.get_global_number_of_atoms()

    # * Divide the atoms into different groups based on atomic number (Z) for finding the coordination shell.
    atomic_numbers = data.get_atomic_numbers()

    # * Create a dictionary of functions for the different atomic number ranges
    bins = [0, 7, 9, 120]
    flags = np.digitize(atomic_numbers, bins).tolist()

    # flags = list(map(str, np.digitize(atomic_numbers, bins)))
    # func_dict = {'1': find_neighbors_smallZ, '2': find_neighbors_oxynitro, '3': find_neighbors_largeZ}
    # %%
    print('Computing features for {} ...'.format(path_to_cif))

    # import dask.bag as db
    atom_ids = list(range(number_of_atoms))
    atom_ids = np.array(atom_ids).reshape(number_of_atoms, 1)
    # %%
    # from dask import delayed # Let's not use that because delayed uses threads by default, which breaks GIL
    # futures = []
    # for i in range(number_of_atoms):
    # 	futures.append(func_dict[flags[i]](i,data))
    # import dask
    # %%
    # output_from_dask = np.array(dask.compute(futures))
    # import  dask.bag as db
    # atom_ids_db = db.from_sequence(atom_ids, npartitions=50).map(use_correct_func, flags, data)
    # %%
    #
    #
    # if client == 'dummy': # If there is no client specified used multiprocessing
    #  	print("Client is dummy with partitions!!")
    #  	output_from_dask = np.array(atom_ids_db.compute())
    # else:
    # 	output_from_dask = np.array(atom_ids_db.compute(scheduler=client))
    # with ProgressBar():
    # 	output_from_dask = np.array(atom_ids_db.compute(scheduler=client_name))
    # data_scatter = client_name.scatter(data, broadcast=True)
    # mof_pm_scatter = client_name.scatter(mof_pm, broadcast=True)
    # from tlz import partition_all

    # print("Submitting feature calc. for atoms with Z<7")
    # chunks1 = partition_all(50, atom_ids[np.where(np.array(flags) == 1)])
    # futures1 = client_name.submit(smallZ_many, chunks1, atoms=data_scatter)
    # import dask.array as da
    # indices1 = da.from_array(np.array(atom_ids[np.where(np.array(flags) == 1)]), chunks=10)
    # indices1 = indices1.map_blocks(smallZ_many, atoms=data)
    # if len(atom_ids[np.where(np.array(flags) == 1)])>0:
    # 	futures1 = client_name.persist(indices1)
    # else:
    # 	futures1 = da.from_array(np.array([]))
    # # futures1 = client_name.persist(indices1)
    #
    # # progress(futures1)
    # # futures1 = client_name.map(find_neighbors_smallZ, atom_ids[np.where(np.array(flags) == 1)], atoms=data_scatter)
    #
    # print("Submitting feature calc. for O and N atoms")
    # # chunks2 = partition_all(50, atom_ids[np.where(np.array(flags) == 2)])
    # # futures2 = client_name.submit(oxynitro_many, chunks2, atoms=data_scatter)
    # indices2 = da.from_array(atom_ids[np.where(np.array(flags) == 2)], chunks=10)
    # indices2 = indices2.map_blocks(oxynitro_many, atoms=data)
    # if len(atom_ids[np.where(np.array(flags) == 2)])>0:
    # 	futures2 = client_name.persist(indices2)
    # else:
    # 	futures2 = da.from_array(np.array([]))
    # # progress(futures2)
    # # futures2 = client_name.map(find_neighbors_oxynitro, atom_ids[np.where(np.array(flags) == 2)], atoms=data_scatter)
    #
    #
    # print("Submitting feature calc. for atoms with Z>8")
    # # chunks3 = partition_all(50, atom_ids[np.where(np.array(flags) == 3)])
    # # futures3 = client_name.submit(largeZ_many, chunks3, mof_pm=mof_pm_scatter, atoms=data_scatter)
    #
    # indices3 = da.from_array(atom_ids[np.where(np.array(flags) == 3)], chunks=10)
    # indices3 = indices3.map_blocks(largeZ_many, mof_pm=mof_pm,atoms=data )
    # if len(atom_ids[np.where(np.array(flags) == 3)])>0:
    # 	futures3 = client_name.persist(indices3)
    # else:
    # 	futures3 = da.from_array(np.array([]))

    # Create global variables using the pymatgen dictionary object via serialization
    # this saves data transfer time while parallelization.
#%%
    if client_name != 'dummy':
        struct_as_dict = mof_pm.as_dict()
        [struct_future] = client_name.scatter([struct_as_dict], broadcast=True)
        atoms_as_dict = data.todict()
        [atoms_future] = client_name.scatter([atoms_as_dict], broadcast=True)
        # Divide the calculation into two groups and treate the calculations separately.
        import dask.bag as db
        indices1_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 1)]),
                                        npartitions=100).map(find_neighbors_smallZ, atoms_dict=atoms_future)
        indices2_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 2)]),
                                        npartitions=100).map( find_neighbors_oxynitro, atoms_dict=atoms_future )
        indices3_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 3)]),
                                        npartitions=100).map(find_neighbors_largeZ, struct_dict=struct_future, atoms_dict=atoms_future)
    #%%
    if client_name == 'dummy':
        import dask.bag as db
        indices1_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 1)]),
                                        npartitions=100).map_partitions(
            lambda chk: [find_neighbors_smallZ2(x, struct=mof_pm) for x in chk])
        indices2_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 2)]),
                                        npartitions=100).map_partitions(
            lambda chk: [find_neighbors_oxynitro2(x, struct=mof_pm) for x in chk])
        indices3_bag = db.from_sequence(np.array(atom_ids[np.where(np.array(flags) == 3)]),
                                        npartitions=100).map_partitions(
            lambda chk: [find_neighbors_largeZ2(x, struct=mof_pm) for x in chk])

    index_list = np.array(np.where(np.array(flags) == 1)[0].tolist() + np.where(np.array(flags) == 2)[0].tolist() +
                          np.where(np.array(flags) == 3)[0].tolist())

    from dask.distributed import progress
    from dask.diagnostics import ProgressBar

    if client_name == 'dummy':
        with ProgressBar():
            print("Submitting feature calc. for atoms with Z<7")
            output1 = indices1_bag.compute()  # Now this is a list of tuples
            print("Feature calc. for O and N atoms")
            output2 = indices2_bag.compute()  # Now this is a list of tuples
            print("Feature calc. for atoms with Z>8")
            output3 = indices3_bag.compute()  # Now this is a list of tuples
            outputs_array = np.array(output1 + output2 + output3, dtype='object')

    else:
        futures1 = client_name.compute(indices1_bag)  # This returns a future, not a list.
        futures2 = client_name.compute(indices2_bag)  # This returns a future, not a list.
        futures3 = client_name.compute(indices3_bag)  # This returns a future, not a list.
        print("Feature calc. for atoms with Z<7")
        progress(futures1)
        print("Feature calc. for O and N atoms")
        progress(futures2)
        print("Feature calc. for atoms with Z>8")
        progress(futures3)
        outputs_array = np.array(client_name.gather(futures1) + client_name.gather(futures2) + client_name.gather(futures3), dtype='object')

    output_from_dask = outputs_array[np.argsort(index_list)]

    # progress(futures2)
    # progress(futures3)
    # output_from_dask = np.array(atom_ids.shape)

    # futures_array = np.array(client_name.gather(futures1)+client_name.gather(futures2)+client_name.gather(futures3))
    # futures_array = np.vstack((client_name.gather(futures1).tolist()+client_name.gather(futures2).tolist()+client_name.gather(futures3).tolist()))
    # futures_array = np.array(futures1.compute().tolist()+futures2.compute().tolist()+futures3.compute().tolist())
    # output_from_dask = futures_array[np.argsort(index_list)]

    # if len(futures1) != 0:
    # 	for index in np.where(np.array(flags) == 1):
    # 		output_from_dask[index] =
    # 	output_from_dask[np.where(np.array(flags) == 1)] = client_name.gather(futures1)
    # if len(futures2) != 0:
    # 	output_from_dask[np.where(np.array(flags) == 2)] = client_name.gather(futures2)
    # if len(futures3) != 0:
    # 	output_from_dask[np.where(np.array(flags) == 3)] = client_name.gather(futures3)

    output_from_dask = np.array(output_from_dask)
    # Stack the results from all the futures to get the aggregated list of features
    # output_from_dask = np.vstack((client_name.gather(futures1), client_name.gather(futures2), client_name.gather(futures3)))
    neighbor_list, avg_neighbor_dist = output_from_dask[:, 0], output_from_dask[:, 1]

    # func_dict = {'1':find_neighbors_smallZ, '2':find_neighbors_oxynitro, '3':find_neighbors_largeZ}

    # neighbor_list, avg_neighbor_dist = zip(*[func_dict[flags[i]](i,data) for i in range(number_of_atoms)])

    # neighbor_list, avg_neighbor_dist = list(neighbor_list), list(avg_neighbor_dist)

    # * Find all the atoms with no neighbors, hopefully there aren't any such atoms.
    # * We have to use a for loop since Python's fancy indexing doesn't work so well on lists.

    nl_length = [len(nl) for nl in neighbor_list]
    no_neighbors = np.where(np.array(nl_length) == 0)[0]
    print(
        "Nearest 2 neighbors are considered since no chemically bonded neighbors are found for {} atoms in this case.".format(
            len(no_neighbors)))

    for nn in no_neighbors:
        # print(nn)
        # temp1, temp2 = find_nearest2(nn,data)
        neighbor_list[nn], avg_neighbor_dist[nn] = find_nearest2(nn, data)

    # * We can use pandas to get values from the dictionary
    enSeries = pd.Series(electronegativity)
    ipSeries = pd.Series(first_ip)

    # * Symbols for the neighbors
    neighbor_symbols = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list]

    # average shell electronegativity and ionization potential
    average_en_shell = [np.mean(enSeries[ns].values) for ns in neighbor_symbols]
    average_ip_shell = [np.mean(ipSeries[ns].values) for ns in neighbor_symbols]

    #%% Section added after manuscript revision
    # Second shell neighbors including central atom
    temp = [np.hstack(([neighbor_list[i] for i in neighbor_list[index]])) for index in range(data.get_global_number_of_atoms())]

    # Exclude the central atom
    neighbor_list_2 = [arr[arr != index] for index, arr in enumerate(temp)]  # Exclude the central atom

    # Symbols for the second shell neighbors
    neighbor_symbols_2 = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list_2]

    # Electronegativity of the second shell neighbors
    average_en_shell_2 = [np.mean(enSeries[ns].values) for ns in neighbor_symbols_2]
    #%%

    features = np.vstack(
        (en_pauling, ionization_energy, nl_length, avg_neighbor_dist, average_en_shell, average_ip_shell, average_en_shell_2)).T

    data.info['features'] = features
    data.info['neighbors'] = neighbor_list

    return data  # * Returns the ASE atoms object and the features array.


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
        distances = atoms.get_distances(i, slice(None), mic=True)
        distances = np.sort(distances)
        indices = np.where(distances < distances[2])[0]
        indices = indices[indices != i]  # * Remove self
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
        from ase.data import covalent_radii
        cov_radii = covalent_radii[atoms.get_atomic_numbers()]

        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii[i] + cov_radii
        indices = np.where(distances < (sum_radii + 0.3))[0]
        indices = indices[indices != i]  # * Remove self

        return indices.tolist(), np.mean(distances[indices])

    # * Large Z
    def find_neighbors_largeZ(i, atoms):
        import numpy as np
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.io import ase
        distances = atoms.get_distances(i, slice(None), mic=True)
        mof = ase.AseAtomsAdaptor.get_structure(atoms=atoms)
        import warnings
        warnings.filterwarnings("ignore")
        nn_object = CrystalNN(x_diff_weight=0, distance_cutoffs=(0.3, 0.5))
        local_env = nn_object.get_nn_info(mof, i)
        indices = [local_env[index]['site_index'] for index in range(len(local_env))]
        return indices, np.mean(distances[indices])

    # * Oxygens and nitrogen
    def find_neighbors_oxynitro(i, atoms):
        import numpy as np
        from ase.data import covalent_radii
        cov_radii = covalent_radii[atoms.get_atomic_numbers()]
        distances = atoms.get_distances(i, slice(None), mic=True)
        sum_radii = cov_radii[i] + cov_radii
        indices = np.where(distances < (sum_radii + 0.5))[0]
        indices = indices[indices != i]  # * Remove self
        return indices.tolist(), np.mean(distances[indices])

    import numpy as np
    import pandas as pd

    radius = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
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
    # electronegativity in pauling scale from CRC Handbook of Chemistry and Physics (For elements not having pauling electronegativity, Allred Rochow electronegativity is taken)
    electronegativity = {'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57,
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
    # First ionization energy (from CRC Handbook of Chemistry and Physics)
    first_ip = {'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323,
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
    # pymatgent nearest neighbor to get local enveronment
    from tqdm import tqdm
    from ase.io import read
    # from dask.diagnostics import ProgressBar
    # data = read(path_to_cif)
    print("Reading  CIF file {}...".format(path_to_cif))
    data = read(path_to_cif)
    # data_pm = AseAtomsAdaptor.get_structure(atoms=data)
    number_of_atoms = data.get_global_number_of_atoms()

    cov_radii = np.array([radius[s] for s in data.get_chemical_symbols()])
    en_pauling = np.array([electronegativity[s] for s in data.get_chemical_symbols()])
    ionization_energy = np.array([first_ip[s] for s in data.get_chemical_symbols()])

    # * Divide the atoms into different groups based on atomic number (Z) for finding the coordination shell.
    atomic_numbers = data.get_atomic_numbers()

    # * Create a dictionary of functions for the different atomic number ranges
    bins = [0, 7, 9, 120]
    flags = list(map(str, np.digitize(atomic_numbers, bins)))

    print("Computing features for {}...".format(path_to_cif))
    # import dask.bag as db
    # atom_ids =list(range(number_of_atoms))
    # atom_ids_db = db.from_sequence(atom_ids).map(use_correct_func, flags, data)
    # with ProgressBar():
    # 	output_from_dask = np.array(atom_ids_db.compute())
    # [neighbor_list, avg_neighbor_dist] = output_from_dask[:, 0], output_from_dask[:,1]

    func_dict = {'1': find_neighbors_smallZ, '2': find_neighbors_oxynitro, '3': find_neighbors_largeZ}

    neighbor_list, avg_neighbor_dist = zip(*[func_dict[flags[i]](i, data) for i in tqdm(range(number_of_atoms))])

    neighbor_list, avg_neighbor_dist = list(neighbor_list), list(avg_neighbor_dist)

    # * Find all the atoms with no neighbors, hopefully there aren't any such atoms.
    # * We have to use a for loop since Python's fancy indexing doesn't work so well on lists.
    nl_length = [len(nl) for nl in neighbor_list]
    no_neighbors = np.where(np.array(nl_length) == 0)[0]
    # print(len(no_neighbors))
    for nn in no_neighbors:
        # print(nn)
        # temp1, temp2 = find_nearest2(nn,data)
        neighbor_list[nn], avg_neighbor_dist[nn] = find_nearest2(nn, data)

    # * We can use pandas to get values from the dictionary
    enSeries = pd.Series(electronegativity)
    ipSeries = pd.Series(first_ip)

    # * Symbols for the neighbors
    neighbor_symbols = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list]

    average_en_shell = [np.mean(enSeries[ns].values) for ns in neighbor_symbols]
    average_ip_shell = [np.mean(ipSeries[ns].values) for ns in neighbor_symbols]

    #%% Section added after manuscript revision
    # Second shell neighbors including central atom
    temp = [np.hstack(([neighbor_list[i] for i in neighbor_list[index]])) for index in range(data.get_global_number_of_atoms())]

    # Exclude the central atom
    neighbor_list_2 = [arr[arr != index] for index, arr in enumerate(temp)]  # Exclude the central atom

    # Symbols for the second shell neighbors
    neighbor_symbols_2 = [np.array(data.get_chemical_symbols())[nl] for nl in neighbor_list_2]

    # Electronegativity of the second shell neighbors
    average_en_shell_2 = [np.mean(enSeries[ns].values) for ns in neighbor_symbols_2]

    features = np.vstack(
        (en_pauling, ionization_energy, nl_length, avg_neighbor_dist, average_en_shell, average_ip_shell, average_en_shell_2)).T

    data.info['features'] = features
    data.info['neighbors'] = neighbor_list

    return data  # * Returns the ASE atoms object and the features array.


# %%
def get_charges_single_serial(path_to_cif, create_cif=False, path_to_output_dir='.', add_string='_charged',
                              use_default_model=True, path_to_pickle_obj='dummy_string'):
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

	:rtype: an ase atoms object with the partial charges added as atoms.info['_atom_site_charges'] and the feature vectors in atoms.info['features']
	"""

    import numpy as np
    import joblib
    import os
    # * Get the path of the pickle and load the model
    print("Loading the model...")
    if use_default_model:
        this_dir, this_filename = os.path.split(__file__)
        path_to_pickle_obj = os.path.join(this_dir, "data", "Model_RF_DDEC.pkl")
        # print(path_to_pickle_obj)
        model = joblib.load(path_to_pickle_obj)
    else:
        model = joblib.load(path_to_pickle_obj)
    # print("Computing features...")
    data = get_features_from_cif_serial(path_to_cif)
    features = data.info['features']
    print("Estimating charges for {}...".format(path_to_cif))
    charges = model.predict(features)

    # charges = np.round(charges, decimals=4)
    # * Adjust the charges for neutrality
    charges_adj = charges - np.sum(charges) * np.abs(charges) / np.sum(np.abs(charges))

    data.info['_atom_site_charge'] = charges_adj.tolist()

    if np.any(np.abs(charges - charges_adj) > 0.2):
        print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality!")
    # if write_cif==True:

    if create_cif:
        print('Writing new cif file...')
        path_to_cif = os.path.abspath(path_to_cif)
        old_name = os.path.basename(path_to_cif)
        new_name = old_name.split('.')[-2] + add_string + '.cif'
        # data1 = data_all[i]
        # new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
        path_to_output_dir = os.path.abspath(path_to_output_dir)
        path_to_output_cif = os.path.join(path_to_output_dir, new_name)
        write_cif(path_to_output_cif, data)

    return data


# %%
def get_charges_single_large(path_to_cif, client_name='dummy', create_cif=False, path_to_output_dir='.',
                             add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):
    """ Description Computes the partial charges for a single CIF file and returns an ASE atoms object updated with
	the estimated charges included as atoms.info['_atom_site_charges']. Features for each CIF is calculated in
	parallel using Dask. Options are included for using a different pickled sklearn model and for write an output CIF
	with the new charges.
	
	:type path_to_cif: string
	:param path_to_cif: path to the cif file as input`

	:type create_cif: bool
	:param create_cif: whether to output a new CIF file while '_atom_site_charges' added 

	:type path_to_output_dir: string
	:param path_to_output_dir: path to the output directory for creating the new CIF file.

	:type add_string: string
	:param add_string: A string added to the filename to distinguish the output cif file from the original one.

    :param client_name: Used to run these calculations on a predefined dask cluster.
	:type client_name: Client object from dask distributed

	:type use_default_model: bool
	:param use_default_model: whether  to use the pre-trained model or not. If set to false you can set path to a different pickle file using 'path_to_pickle_obj'.

	:type path_to_pickle_obj: string
	:param path_to_pickle_obj: path to a pickle file containing the scikit-learn
	model one wants to use. Is used only if use_default_model is set to False.

	:raises:

	:rtype: an ase atoms object with the partial charges added as atoms.info['_atom_site_charges'] and the feature
	vectors in atoms.info['features'].

	"""

    import numpy as np
    import joblib
    import os
    # * Get the path of the pickle and load the model
    print("Loading the model...")
    if use_default_model:
        this_dir, this_filename = os.path.split(__file__)
        path_to_pickle_obj = os.path.join(this_dir, "data", "Model_RF_DDEC.pkl")
        # print(path_to_pickle_obj)
        model = joblib.load(path_to_pickle_obj)
    else:
        model = joblib.load(path_to_pickle_obj)
    # print("Computing features...")

    data_out = get_features_from_cif_parallel(path_to_cif, client_name=client_name)

    features = data_out.info['features']
    print('Estimating charges...')
    charges = model.predict(features)
    # round the charges
    # charges = np.round(charges, decimals=4)

    # * Adjust the charges for neutrality
    charges_adj = charges - np.sum(charges) * np.abs(charges) / np.sum(np.abs(charges))

    data_out.info['_atom_site_charge'] = charges_adj.tolist()

    if np.any(np.abs(charges - charges_adj) > 0.2):
        print("WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality!")
    # if write_cif==True:

    if create_cif:
        print('Writing new cif file...')
        path_to_cif = os.path.abspath(path_to_cif)
        old_name = os.path.basename(path_to_cif)
        new_name = old_name.split('.')[-2] + add_string + '.cif'
        # data1 = data_all[i]
        # new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
        path_to_output_dir = os.path.abspath(path_to_output_dir)
        path_to_output_cif = os.path.join(path_to_output_dir, new_name)
        write_cif(path_to_output_cif, data_out)

    return data_out


# %%
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
            # charges = atoms.info['_atom_site_charge']
        else:
            coords = atoms.get_positions().tolist()
        symbols = atoms.get_chemical_symbols()
        occupancies = [1 for i in range(len(symbols))]

        charges = atoms.info['_atom_site_charge']
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
                          '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %s  %6.16f\n'
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
def get_charges_multiple_onebyone(list_of_cifs, client_name='dummy', create_cif=False, path_to_output_dir='.',
                                  add_string='_charged', use_default_model=True, path_to_pickle_obj='dummy_string'):
    """ Description
	Compute the partial charges in a list of CIFs. This function saves time by loading the Random forest model only once. Each of the individual calculatiions
	are parallelized and this function is made for when the CIF files involved are very large (>2000 atoms). The files will
	be processed one by one but each calculation will be parallelized for speed.

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

	:type client_name: Client object from dask distributed
	:param client_name: Used to run these calculations on a predefined dask cluster.

	:raises:

	:rtype:  a list of ase atoms objects with charges added as atoms.info['_atom_site_charges'] to each atoms object and the
	feature vectors added under atoms.info['features']
	"""

    from tqdm import tqdm
    import numpy as np
    import joblib
    import os

    def adjust_charge(charges):
        import numpy as np
        # charges = np.round(charges, decimals=4)
        return charges - np.sum(charges) * np.abs(charges) / np.sum(np.abs(charges))

    # * Get the path of the pickle and load the model
    print("Loading the model...")
    if use_default_model:
        this_dir, this_filename = os.path.split(__file__)
        path_to_pickle_obj = os.path.join(this_dir, "data", "Model_RF_DDEC.pkl")
        # print(path_to_pickle_obj)
        model = joblib.load(path_to_pickle_obj)
    else:
        model = joblib.load(path_to_pickle_obj)

    print('Calculating the features for all the files one by one...')
    # if client == 'dummy':
    # from dask.distributed import Client
    # client = Client()

    data_all = [get_features_from_cif_parallel(l, client_name=client_name) for l in tqdm(list_of_cifs)]

    # else:
    # 	data_all = [get_features_from_cif_parallel(l, client=client) for l in tqdm(list_of_cifs)]

    # data_all = list(data_all)
    charges_all = [model.predict(d.info['features']) for d in data_all]
    # print(len(data_all), len(features_all), len(charges_all))
    print('Estimating charges for all the files...')
    for i in range(len(list_of_cifs)):
        charges_adj = adjust_charge(charges_all[i])
        data_all[i].info['_atom_site_charge'] = charges_adj
        if np.any(np.abs(charges_all[i] - charges_adj) > 0.2):
            print(
                "WARNING: Some charges were adjusted by more than 0.2 to maintain neutrality in file\n" + list_of_cifs[
                    i])

    # * Write the output cif files
    if create_cif:
        print('Writing new cif file...')
        for i in range(len(list_of_cifs)):
            # print(i)
            path_to_cif = os.path.abspath(list_of_cifs[i])
            old_name = os.path.basename(path_to_cif)
            new_name = old_name.split('.')[-2] + add_string + '.cif'
            # data1 = data_all[i]
            # new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
            path_to_output_dir = os.path.abspath(path_to_output_dir)
            path_to_output_cif = os.path.join(path_to_output_dir, new_name)
            write_cif(path_to_output_cif, data_all[i])
    print("Done!")
    return data_all


# %%
def get_charges_multiple_parallel(list_of_cifs, create_cif=False, path_to_output_dir='.', add_string='_charged',
                                  use_default_model=True, path_to_pickle_obj='dummy_string', client_name='dummy'):
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

	:type client_name: Client object from dask distributed
	:param client_name: Used to run these calculations on a predefined dask cluster.

	:raises:

	:rtype:  a list of ase atoms objects with charges added as atoms.info['_atom_site_charges'] to each atoms object
	"""

    import numpy as np
    import joblib
    import os

    def adjust_charge(charges):
        import numpy as np
        # charges = np.round(charges, decimals=4)
        return charges - np.sum(charges) * np.abs(charges) / np.sum(np.abs(charges))

    import dask.bag as db
    from dask.diagnostics import ProgressBar
    # * Get the path of the pickle and load the model
    print("Loading the model...")
    if use_default_model:
        this_dir, this_filename = os.path.split(__file__)
        path_to_pickle_obj = os.path.join(this_dir, "data", "Model_RF_DDEC.pkl")
        # print(path_to_pickle_obj)
        model = joblib.load(path_to_pickle_obj)
    else:
        model = joblib.load(path_to_pickle_obj)

    print('Calculating features for all the files...')

    # Use a distributed scheduler always on the single machine and on a dask cluster
    data_db = db.from_sequence(list_of_cifs).map(get_features_from_cif_serial)
    from dask.distributed import progress
    if client_name == 'dummy':
        # Use the distributed scheduler on a local cluster
        # from dask.distributed import Client
        # client = Client()
        with ProgressBar():
            data_all = data_db.compute()
    else:
        # if we have a dask cluster, then also use the distributed scheduler which is the default
        future_ = client_name.persist(data_db)
        progress(future_)
        data_all = client_name.gather(future_)
    # data_all = data_db.compute(scheduler=client_name)

    # 	# data_all= zip(*[get_features_from_cif(l) for l in tqdm(list_of_cifs)])
    # else:
    #

    data_all = list(data_all)
    charges_all = [model.predict(d.info['features']) for d in data_all]
    # print(len(data_all), len(features_all), len(charges_all))
    print('Estimating charges for all the files...')
    for i in range(len(list_of_cifs)):
        charges_adj = adjust_charge(charges_all[i])
        data_all[i].info['_atom_site_charge'] = charges_adj
        if np.any(np.abs(charges_all[i] - charges_adj) > 0.2):
            print("WARNING: Some charges in {} were adjusted by more than 0.2 to maintain neutrality in file \n".format(
                list_of_cifs[i]))

    # * Write the output cif files
    if create_cif:
        print('Writing new cif files...')
        for i in range(len(list_of_cifs)):
            # print(i)
            path_to_cif = os.path.abspath(list_of_cifs[i])
            old_name = os.path.basename(path_to_cif)
            new_name = old_name.split('.')[-2] + add_string + '.cif'
            # data1 = data_all[i]
            # new_filename = path_to_cif.split('.')[-2].split('\\')[-1]+add_string+ '.cif'
            path_to_output_dir = os.path.abspath(path_to_output_dir)
            path_to_output_cif = os.path.join(path_to_output_dir, new_name)
            write_cif(path_to_output_cif, data_all[i])
    print("Done!")
    return data_all

# if compute==True:
# 	return charges_db.compute()
# else:
# 	return charges_db
