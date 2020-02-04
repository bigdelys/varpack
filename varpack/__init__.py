import numpy as np
import pickle
import os
import json
import sys
import shutil

# min required Python 3.4

MISC_VAR_FILENAME = '__misc_vars__.pickle'
JSON_FILENAME = 'varpack.json'
PICKLE_PROTOCOL = 4
from typing import Union, Dict, List
import typing


def get_total_obj_size(obj, seen=None, count_mmap_size=False):
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
    elif isinstance(obj, np.memmap):
        size += os.path.getsize(obj.filename)
    elif hasattr(obj, '__dict__'):
        size += get_total_obj_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_obj_size(i, seen) for i in obj])
    return size


class NumpyArrayPlaceholder:

    def __init__(self, np_arr=None, save_folder=None, var_name=None, key_hash=None):

        if np_arr is not None:

            # if the variable is numpy mmapped
            if isinstance(np_arr, np.memmap):
                np_arr.flush()
                dir_name = os.path.dirname(np_arr.filename)
                self.filename = os.path.basename(np_arr.filename)
                if dir_name != save_folder:   # the mmap file is in a different directory than where we are saving
                    shutil.copyfile(np_arr.filename, os.path.join(save_folder, os.path.basename(np_arr.filename)) )

                return

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

        # for each variable as key, contains different information such as its size (in memory) and
        # the file where it is saved.
        self.__internal__['var_info'] = dict()
        self.__internal__['attached_folder'] = None
        self.__internal__['numpy_mmap_mode'] = 'r+'

    def get_attached_folder(self):
        return self.__internal__['attached_folder']

    def _replace_numpy_placeholders(self, var_info, load_folder, numpy_mmap_mode, stop_on_error,
                                    mmap_vars_list=None):
        # go  over all the loaded variables and replace placeholder numpy arrays with mmap ones
        all_vars = vars(self)
        for v in all_vars:

            # if v != '__internal__':
            #     print(var_info[v])

            if v != '__internal__' and 'uses_numpy_placeholders' in var_info[v] and var_info[v]['uses_numpy_placeholders']:
                # go over keys in the dictionary and replace the placeholders with numpy arrays
                for k in all_vars[v]:
                    if isinstance(all_vars[v][k], NumpyArrayPlaceholder):
                        all_vars[v][k] = all_vars[v][k].load(load_folder, numpy_mmap_mode)

                        if isinstance(all_vars[v][k], NumpyArrayPlaceholder):
                            print('Could not load numpy array from the placeholder in variable: %s, key: %s'
                                  % (v, k))
                            if stop_on_error:
                                return None
                        elif isinstance(all_vars[v][k], np.memmap):
                            if mmap_vars_list is not None:
                                mmap_vars_list.append(v + '[' + k + ']')

    def save(self, save_folder: typing.Optional[str] = None, max_dict_keys: int = 1000,
             min_dict_numpy_size: int = 10000, sep_var_min_size: int = 1e4,
             sep_vars: typing.Optional[Union[Dict, List]] = None):
        """
        Save the pack of variables. If no save folder is provided, the variables are saved in the folder from which
        they were loaded.

        * a subset variables can be marked to be saved in separate files, e.g. so they could later be loaded individually
        or to be excluded from loading the pack.

        * when saving to a new folder, files associated with variables that were excluded during load will be copied
          into the save folder.

        * saving into a new folder does not change the attached_folder (folder from which the variables were loaded).
          Unless data was never loaded from a folder. This values is stored in .__internal__['attached_folder']

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

        if save_folder is None:
            save_folder = self.__internal__['attached_folder']
            print('Saving variables into the attached folder: ', save_folder)

        assert save_folder is not None, 'Missing save_folder input parameter and no loaded folder exists to default to.'

        os.makedirs(save_folder, exist_ok=True)

        # variables that cannot be saved as numpy arrays and need to be saved using pickle
        pickle_vars = list()

        obj_vars = vars(self)

        if sep_vars is None:
            sep_vars = set()
        else:
            sep_vars = set(sep_vars)

        for var_name in obj_vars:
            if var_name != '__internal__':  # do not save __internal__ variable.
                self.__internal__['var_info'][var_name] = dict()
                self.__internal__['var_info'][var_name]['size'] = get_total_obj_size(obj_vars[var_name],
                                                                                     count_mmap_size=False)

                # if the variable is a numpy array then try to save it as .npy
                if isinstance(obj_vars[var_name], np.ndarray):

                    if type(obj_vars[var_name]) is np.memmap:
                        # flush mmap to disk, no need to re-save if
                        # saving to the same folder where it is memory-mapped
                        obj_vars[var_name].flush()

                        # saving to a new folder (not where the data was loaded from)
                        # copy numpy files
                        if save_folder != self.__internal__['attached_folder']:
                            print('Copying...')
                            shutil.copyfile(obj_vars[var_name].filename,
                                            os.path.join(save_folder, os.path.basename(obj_vars[var_name].filename)))

                        self.__internal__['var_info'][var_name]['filename'] = \
                            os.path.basename(obj_vars[var_name].filename)
                        self.__internal__['var_info'][var_name]['shape'] = obj_vars[var_name].shape
                        self.__internal__['var_info'][var_name]['dtype'] = str(obj_vars[var_name].dtype)
                    else:
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

                    # see if it is dictionary made up of numpy arrays (and does not have too many keys)
                    if type(obj_vars[var_name]) is dict and len(obj_vars[var_name]) < max_dict_keys:
                        uses_numpy_placeholders = False

                        for k in obj_vars[var_name]:
                            # if the key is a numpy array and has enough elements that makes it worth saving as a
                            # separate file.

                            if isinstance(obj_vars[var_name][k], np.ndarray) and \
                                    obj_vars[var_name][k].size >= min_dict_numpy_size:

                                print('came here for ', k)
                                print(str(type(obj_vars[var_name][k])))
                                numpy_array_placeholder = NumpyArrayPlaceholder(obj_vars[var_name][k],
                                                                                save_folder=save_folder,
                                                                                var_name=var_name,
                                                                                key_hash=k.__hash__())

                                print('came out for ', k)

                                # if saving the value as a numpy array was successful,
                                # put the placeholder object there instead.
                                if numpy_array_placeholder.filename is not None:
                                    obj_vars[var_name][k] = numpy_array_placeholder
                                    uses_numpy_placeholders = True

                        # update the size of the variable now that large numpy arrays have been replaced
                        # with placeholders
                        if uses_numpy_placeholders:
                            if not 'size_before_numpy_placeholders' in self.__internal__['var_info'][var_name]:
                                self.__internal__['var_info'][var_name]['size_before_numpy_placeholders'] = \
                                    get_total_obj_size(obj_vars[var_name], count_mmap_size=True)

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

        # must copy the files skipped during loading to the save folder (if saving to a different folder)
        if save_folder != self.__internal__['attached_folder'] and \
                self.__internal__['attached_folder'] is not None:
            vars_needs_copying = set(self.__internal__['var_info'].keys()) - set(obj_vars)

            files_need_copying = set()
            for v in vars_needs_copying:
                files_need_copying.add(self.__internal__['var_info'][v]['filename'])

            # copy the files from where variables were loaded to the saved folder
            for filename in files_need_copying:
                shutil.copyfile(os.path.join(self.__internal__['attached_folder'], filename),
                                os.path.join(save_folder, filename))

            print('Copied %d files associated with skipped variables into the save folder.' % len(files_need_copying))

        base_folder = self.__internal__['attached_folder']
        if base_folder is None:
            base_folder = save_folder
        self._replace_numpy_placeholders(self.__internal__['var_info'], base_folder,
                                         numpy_mmap_mode=self.__internal__['numpy_mmap_mode'], stop_on_error=True,
                                         mmap_vars_list=None)

        # if data was never loaded, set the loaded folder to the first save location.
        if self.__internal__['attached_folder'] is None:
            self.__internal__['attached_folder'] = save_folder

    def load(self, load_folder, numpy_mmap_mode='r+', stop_on_error=True, skip_loading=None):

        mmap_vars_list = list()  # the list of variables and dictionary fields that have been numpy memory-mapped

        # read the manifest.json file
        try:
            with open(os.path.join(load_folder, JSON_FILENAME), 'r') as json_file:
                var_info = json.load(json_file)
                print(var_info)
                return

                self.__internal__['var_info'] = var_info
        except EnvironmentError:
            print('Error when loading ' + JSON_FILENAME + ' file.')
            return None

        # get all the files where the variables have been saved to (in case there are extra files in the folder)
        files_to_load = set()
        self.__internal__['skipped_loading_vars'] = set()

        if skip_loading is None:
            skip_loading = set()
        else:
            skip_loading = set(skip_loading)

        for var_name in var_info:
            if var_name in skip_loading:
                # remember which variables where skipped during loading
                self.__internal__['skipped_loading_vars'].add(var_name)
            else:
                files_to_load.add(var_info[var_name]['filename'])
        files_to_load = list(files_to_load)

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
                            if v in skip_loading:  # even if the variable was
                                print('Variable %s was saved in misc. variables file so it was loaded with them.' % v)
                                self.__setattr__(v, loaded_vars[v])

                                # since we ended up loading it anyways
                                self.__internal__['skipped_loading_vars'].remove(v)
                            else:  # if not in skip list
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
                var_name, _ = os.path.splitext(os.path.basename(file_name))

                # first try to mmap, otherwise load regularly
                try:
                    np_arr = np.load(os.path.join(load_folder, file_name), mmap_mode=numpy_mmap_mode)
                    mmap_vars_list.append(var_name)
                except:
                    np_arr = np.load(os.path.join(load_folder, file_name))

                self.__setattr__(var_name, np_arr)
            else:
                print('Unable to load file %s: Unknown file extension %s .' % (file_name, extension))
                files_with_load_error.append(file_name)

        # go  over all the loaded variables and replace placeholder numpy arrays with mmap ones
        self._replace_numpy_placeholders(var_info, load_folder, numpy_mmap_mode=numpy_mmap_mode,
                                         stop_on_error=stop_on_error,
                                         mmap_vars_list=mmap_vars_list)
        self.__internal__['numpy_mmap_mode'] = numpy_mmap_mode

        num_skipped_vars = len(var_info.keys()) - len(vars(self))
        if num_skipped_vars > 0:
            print('Skipped loading %d variables.' % num_skipped_vars)

        if len(mmap_vars_list) > 0:
            print('The following numpy variables have been memory-mapped with option %s:' % numpy_mmap_mode)
            print('    ' + str(mmap_vars_list))
        else:
            print('No properties has been memory-mapped.')

        self.__internal__['attached_folder'] = load_folder
