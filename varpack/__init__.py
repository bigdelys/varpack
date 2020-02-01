import numpy as np
import pickle
import os
import json
import sys

# min required Python 3.4

MISC_VAR_FILENAME = '__misc_vars__.pickle'
JSON_FILENAME = 'pack.json'
PICKLE_PROTOCOL = 4


def get_total_obj_size(obj, seen=None):
    """Recursively finds size of objects, includes the size of embedded objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_total_obj_size(v, seen) for v in obj.values()])
        size += sum([get_total_obj_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_obj_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_obj_size(i, seen) for i in obj])
    return size


class NumpyArrayPlaceholder:

    def __init__(self, np_arr=None, save_folder=None, var_name=None, key_hash=None):
        if np_arr is not None:
            filename = os.path.join(save_folder, var_name + str(key_hash) + '.npy')
            try:
                # need to allow pickle here since no other way to save mixed numpy and Python objects
                np.save(filename, np_arr, allow_pickle=True)
                self.filename = os.path.basename(filename)
            except EnvironmentError:
                print('Failed in saving numpy placeholder file:', filename)
                self.filename = None  # means it was not successful

    def load(self, load_folder, mmap_mode):

        # need to allow pickle here since no other way to save mixed numpy and Python objects
        full_filename = os.path.join(load_folder, self.filename)

        try:
            np_arr = np.load(full_filename, allow_pickle=True, mmap_mode=mmap_mode)
            return np_arr
        except:
            print('Failed to memory map file:', full_filename)
            if mmap_mode is not None:
                try:
                    np_arr = np.load(full_filename, mmap_mode=None)
                    print('Loaded it all in memory instead.')
                    return np_arr
                except:
                    print('Also failed to load it all in memory.')
                    return self


class VarPack:

    def __init__(self):
        self.__internal__ = dict()

    def save(self, save_folder: str, max_dict_keys=1000, min_dict_numpy_size=10000, sep_var_min_size=1e4,
             sep_vars=None):
        """

        :param save_folder: folder in which all the variables will be saved.
        :param max_dict_keys: do not try to replace large numpy arrays with dictionaries with larger than
                              this number of keys. This is mainly to avoid wasting time checking keys in very large
                              dictionaries.
        :param min_dict_numpy_size: minimum number of elements in a numpy array for it to be replaced with
                                    a placeholder objects that points to a separate numpy file.
        :param sep_var_min_size:  minimum total in-memory size (in bytes) for a variable to be saved in a
                                  separate pickle file (so for example could be excluded during load).
        :param sep_vars: a list containing variables that need to be saved in a different pickle file.
                         All numpy arrays are automatically saved in separate .npy files.
        :return: None
        """

        os.makedirs(save_folder, exist_ok=True)

        # variables that cannot be saved as numpy arrays and need to be saved using pickle
        pickle_vars = list()

        obj_vars = vars(self)

        # for each variable as key, contains different information such as its size (in memory) and
        # the file where it is saved.
        self.__internal__['var_info'] = dict()

        if sep_vars is None:
            sep_vars = list()
        else:
            sep_vars = set(sep_vars)

        for var_name in obj_vars:

            self.__internal__['var_info'][var_name] = dict()
            self.__internal__['var_info'][var_name]['size'] = get_total_obj_size(obj_vars[var_name])

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
                if type(obj_vars[var_name]) is dict and len(obj_vars[var_name]) < max_dict_keys:
                    uses_numpy_placeholders = False
                    for k in obj_vars[var_name]:
                        # if the key is a numpy array and has enough elements that makes it worth saving as a
                        # separate file.

                        if isinstance(obj_vars[var_name][k], np.ndarray) and \
                                obj_vars[var_name][k].size >= min_dict_numpy_size:
                            numpy_array_placeholder = NumpyArrayPlaceholder(obj_vars[var_name][k],
                                                                            save_folder=save_folder, var_name=var_name,
                                                                            key_hash=k.__hash__())
                            # if saving the value as a numpy array was successful,
                            # put the placeholder object there instead.
                            if numpy_array_placeholder.filename is not None:
                                obj_vars[var_name][k] = numpy_array_placeholder
                                uses_numpy_placeholders = True

                    # update the size of the variable now that large numpy arrays have been replaced with placeholders
                    if uses_numpy_placeholders:
                        self.__internal__['var_info'][var_name]['size_before_numpy_placeholders'] = \
                            self.__internal__['var_info'][var_name]['size']

                        self.__internal__['var_info'][var_name]['size'] = get_total_obj_size(obj_vars[var_name])

                    # keep track of whether the dictionary is using numpy placeholder objects (useful when loading)
                    self.__internal__['var_info'][var_name]['uses_numpy_placeholders'] = uses_numpy_placeholders

                # identify variables that are too large and need to be placed in separate pickle files.
                if self.__internal__['var_info'][var_name]['size'] >= sep_var_min_size:
                    sep_vars.add(var_name)

        # save the rest of variables as pickle
        pickle_dict = dict()
        for v in pickle_vars:
            pickle_dict[v] = self.__getattribute__(v)

        # save variables that need to have separate files
        for var_name in sep_vars:
            with open(os.path.join(save_folder, var_name + '.pickle'), 'wb') as f:
                pickle.dump(pickle_dict[var_name], f, protocol=PICKLE_PROTOCOL)
                self.__internal__['var_info'][var_name]['filename'] = var_name + '.pickle'

        misc_vars = set(pickle_vars) - sep_vars
        misc_dict = dict()
        for v in misc_vars:
            misc_dict[v] = pickle_dict[v]
            self.__internal__['var_info'][v]['filename'] = MISC_VAR_FILENAME

        with open(os.path.join(save_folder, MISC_VAR_FILENAME), 'wb') as f:
            pickle.dump(misc_dict, f, protocol=PICKLE_PROTOCOL)

        # save a json file with variable info
        with open(os.path.join(save_folder, JSON_FILENAME), 'w') as outfile:
            json.dump(self.__internal__['var_info'], outfile, indent=4)

    def load(self, load_folder, try_numpy_mmap_mode='r+', stop_on_error=True, skip_loading=None):

        # read the manifest.json file
        try:
            with open(os.path.join(load_folder, JSON_FILENAME), 'r') as json_file:
                var_info = json.load(json_file)
        except EnvironmentError:
            print('Error when loading ' + JSON_FILENAME + ' file.')
            return None

        # get all the files where the variables have been saved to (in case there are extra files in the folder)
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
                    if file_name == MISC_VAR_FILENAME:
                        for v in loaded_vars:
                            if 'uses_numpy_placeholders' in var_info[v] and var_info[v]['uses_numpy_placeholders']:
                                # go over keys in the dictionary and replace the placeholders with numpy arrays
                                for k in loaded_vars[v]:
                                    if isinstance(loaded_vars[v][k], NumpyArrayPlaceholder):
                                        loaded_vars[v][k] = loaded_vars[v][k].load(load_folder, try_numpy_mmap_mode)

                                        if isinstance(loaded_vars[v][k], NumpyArrayPlaceholder):
                                            print('Could load numpy array from the placeholder in variable: %s, key: %s'
                                                  % (v, k))
                                            if stop_on_error:
                                                return None

                            self.__setattr__(v, loaded_vars[v])
                    else:  # if it is not the misc_vars file, then assign it directly
                        self.__setattr__(name, loaded_vars)

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
