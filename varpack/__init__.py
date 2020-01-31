import numpy as np
import pickle
import os
import json

# min required Python 3.2

PICKLE_FILENAME = 'all_pickled_vars.pickle'
JSON_FILENAME = 'pack.json'


class NumpyArrayPlaceholder:
    def __init__(self, np_arr=None, save_folder=None, var_name=None, key_hash=None):
        if np_arr is not None:
            filename = os.path.join(save_folder, var_name + str(key_hash) + '.npy')
            try:
                # need to allow pickle here since no other way to save mixed numpy and Python objects
                np.save(filename, np_arr, allow_pickle=True)
                self.filename = filename
            except:
                print('Failed in saving numpy placeholder file:', filename)
                self.filename = None  # means it was not successful

    def load(self):
        pass


class VarPack:

    def __init__(self, max_dict_keys=1000, min_dict_numpy_size=10000):
        self.__internal__ = {'max_dict_keys': max_dict_keys,
                             'min_dict_numpy_size': min_dict_numpy_size}

    def save(self, save_folder: str):

        os.makedirs(save_folder, exist_ok=True)

        # variables that cannot be saved as numpy arrays and need to be saved using pickle
        pickle_vars = list()

        obj_vars = vars(self)

        # for each variable as key, contains different information such as its size (in memory) and
        # the file where it is saved.
        self.__internal__['var_info'] = dict()

        for var_name in obj_vars:

            self.__internal__['var_info'][var_name] = dict()
            self.__internal__['var_info'][var_name]['size'] = obj_vars[var_name].__sizeof__()

            # if the variable is a numpy array then try to save it as .npy
            if isinstance(obj_vars[var_name], np.ndarray):
                try:
                    filename = var_name + '.npy'
                    np.save(os.path.join(save_folder, filename), obj_vars[var_name],
                            allow_pickle=False)  # need to disallow pickle here otherwise all vars are saved

                    self.__internal__['var_info'][var_name]['filename'] = filename
                    self.__internal__['var_info'][var_name]['shape'] = obj_vars[var_name].shape
                    self.__internal__['var_info'][var_name]['dtype'] = str(obj_vars[var_name].dtype)
                except:
                    pickle_vars.append(var_name)
            else:  # cannot readily be saved as a numpy array
                pickle_vars.append(var_name)

                # see if it is dictionary made up of numpy arrays (and not has too many keys)
                if type(obj_vars[var_name]) is dict and len(obj_vars[var_name]) < self.__internal__['max_dict_keys']:
                    for k in obj_vars[var_name]:
                        # if the key is a numpy array and has enough elements that makes it worth saving as a
                        # separate file.
                        if isinstance(obj_vars[var_name][k], np.ndarray) and \
                                obj_vars[var_name][k].size >= self.__internal__['min_dict_numpy_size']:
                            numpy_array_placeholder = NumpyArrayPlaceholder(obj_vars[var_name][k],
                                                                            save_folder=save_folder, var_name=var_name,
                                                                            key_hash=k.__hash__())
                            # if saving the value as a numpy array was successful,
                            # put the placeholder object there instead.
                            if numpy_array_placeholder.filename is not None:
                                obj_vars[var_name][k] = numpy_array_placeholder

        # save the rest of variables as pickle
        pickle_dict = dict()
        for v in pickle_vars:
            pickle_dict[v] = self.__getattribute__(v)

            self.__internal__['var_info'][v]['filename'] = PICKLE_FILENAME

        with open(os.path.join(save_folder, PICKLE_FILENAME), 'wb') as f:
            pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save a json file with variable info
        with open(os.path.join(save_folder, JSON_FILENAME), 'w') as outfile:
            json.dump(self.__internal__['var_info'], outfile, indent=4)

    def load(self, load_folder, memory_map=True, stop_on_error=True):

        # read the manifest.json file
        try:
            with open(os.path.join(load_folder, JSON_FILENAME), 'r') as json_file:
                var_info = json.load(json_file)
        except EnvironmentError:
            print('Error when loading ' + JSON_FILENAME + ' file.')
            return None

        files_to_load = [var_info[var_name]['filename'] for var_name in var_info]
        files_to_load = list(set(files_to_load))  # find unique files

        files_with_load_error = list()

        for file_name in files_to_load:
            name, extension = os.path.splitext(file_name)
            if extension == '.pickle':
                try:
                    with open(os.path.join(load_folder, file_name), 'rb') as f:
                        loaded_vars = pickle.load(f)

                    # transfer the variables to the object
                    for v in loaded_vars:
                        self.__setattr__(v, loaded_vars[v])
                except EnvironmentError:
                    print('Error when loading ' + file_name + ' file.')
                    if stop_on_error:
                        return None
                    else:
                        print('Skipping its contents.')
                        files_with_load_error.append(file_name)
            elif extension == '.npy':
                np_arr = np.load(os.path.join(load_folder, file_name))
                var_name, _ = os.path.splitext(os.path.basename(file_name))
                self.__setattr__(var_name, np_arr)
            else:
                print('Unable to load file %s: Unknown file extension %s .' % (file_name, extension))
                files_with_load_error.append(file_name)