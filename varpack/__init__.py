import numpy as np
import pickle
import os

# min required Python 3.2

PICKLE_FILENAME = 'all_pickled_vars.p'
INTERNAL_VARNAME = '__internal__'

class VarPack:

    def __init__(self, max_dict_keys=1000, min_dict_numpy_size=10000):
        self.INTERNAL_VARNAME = {'max_dict_keys': max_dict_keys,
                                 'min_dict_numpy_size': min_dict_numpy_size}

    def save(self, save_folder: str):

        os.makedirs(save_folder, exist_ok=True)

        # variables that cannot be saved as numpy arrays and need to be saved using pickle
        pickle_vars = list()

        obj_vars = vars(self)
        for var_name in obj_vars:
            # if the variable is a numpy array then try to save it as .npy
            if isinstance(obj_vars[var_name], np.ndarray):
                try:
                    np.save(os.path.join(save_folder, var_name + '.npy'), obj_vars[var_name],
                            allow_pickle=False)  # need to disallow pickle here otherwise all vars are saved
                except:
                    pickle_vars.append(var_name)
            else:  # cannot readily be saved as a numpy array

                # see if it is dictionary made up of numpy arrays (and not has too many keys)
                if type(obj_vars[var_name]) is dict and len(obj_vars[var_name]) < self.INTERNAL_VARNAME['max_dict_keys']:
                    for k in obj_vars[var_name]:
                        if isinstance(obj_vars[var_name][k], np.ndarray) and \
                                obj_vars[var_name][k].size >= self.INTERNAL_VARNAME['min_dict_numpy_size']:
                            pass

                else:
                    pickle_vars.append(var_name)

        # save the rest of variables as pickle
        pickle_dict = dict()
        for v in pickle_vars:
            pickle_dict[v] = self.__getattribute__(self, v)

        with open(os.path.join(save_folder, PICKLE_FILENAME), 'wb') as f:
            pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
