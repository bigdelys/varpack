import numpy as np
import pickle
import os
import json
import sys
import shutil
import copy

# min required Python 3.4

MISC_VAR_FILENAME = '__misc_vars__.pickle'
JSON_FILENAME = 'varpack.json'
PICKLE_PROTOCOL = 4
from typing import Union, Dict, List
import typing


def mmap_var_to_memory(x):
    """
    Converts a memory-mapped numpy array, or a python dictionary with mmap values, to a regular in-memory array
    :param x: memory-mapped array
    :return: numpy array (in RAM)
    """
    if type(x) is dict:
        y = dict()
        for k in x:
            y[k] = mmap_var_to_memory(x[k])
    elif type(x) is np.memmap:
        y = np.zeros(shape=x.shape, dtype=x.dtype)
        y[:] = x[:]
    else:
        y = x

    return y


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
        if obj.filename is not None:
            size += os.path.getsize(obj.filename)
        else:
            size = np.nan
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
                try:
                    dir_name = os.path.dirname(np_arr.filename)
                    self.filename = os.path.basename(np_arr.filename)

                    if dir_name != save_folder:  # the mmap file is in a different directory than where we are saving
                        shutil.copyfile(np_arr.filename, os.path.join(save_folder, os.path.basename(np_arr.filename)))

                    return
                except TypeError:  # if encountered with problems, copy into memory and save
                    print('Converting a mmaped array with no associated file into regular numpy array.')
                    np_arr_mem = np.zeros(shape=np_arr.shape, dtype=np_arr.dtype)
                    np_arr_mem[:] = np_arr[:]
                    np_arr = np_arr_mem

            filename = os.path.join(save_folder, var_name + str(key_hash) + '.npy')
            try:
                # need to allow pickle here since no other way to save_copy mixed numpy and Python objects
                np.save(filename, np_arr, allow_pickle=True)
                self.filename = os.path.basename(filename)
            except EnvironmentError:
                print('Failed in saving numpy placeholder file:', filename)
                self.filename = None  # means it was not successful

    def load(self, load_folder, mmap_mode):

        # need to allow pickle here since no other way to save_copy mixed numpy and Python objects
        full_filename = os.path.join(load_folder, self.filename)

        try:
            np_arr = np.load(full_filename, allow_pickle=True, mmap_mode=mmap_mode)
            return np_arr
        except:
            print('Could not memory map file:', full_filename)
            if mmap_mode is not None:
                try:
                    np_arr = np.load(full_filename, allow_pickle=True, mmap_mode=None)
                    print('Loaded it all in memory instead.')
                    return np_arr
                except:
                    print('Also failed to load it all in memory.')
                    raise
                    return self


class Varpack:

    def __init__(self, attached_folder=None, **kwargs):
        """
        Initiate Varpack class instance. Optionally attach it to a folder and load the data.
        :param attached_folder: attached/load folder.
        :param kwargs: key-value arguments passed to load()
        """
        self.__internal__ = dict()

        # for each variable as key, contains different information such as its size (in memory) and
        # the file where it is saved.
        self.__internal__['var_info'] = dict()
        self.__internal__['attached_folder'] = None
        self.__internal__['numpy_mmap_mode'] = 'r+'
        self.__internal__['skipped_loading_vars'] = set()
        self.__internal__['skip_saving_vars'] = set()

        if attached_folder is not None:
            # load if json file exists, otherwise attach to it
            if os.path.isfile(os.path.join(attached_folder, JSON_FILENAME)):
                print('Loading from ', attached_folder)
                self.load(load_folder=attached_folder, **kwargs)
            else:
                self.__internal__['attached_folder'] = attached_folder

    def get_attached_folder(self):
        return self.__internal__['attached_folder']

    def detach(self):
        # loads all the variables into memory and detaches from the disk folder. This enables saving it somewhere else.
        if self.__internal__['attached_folder'] is None:   # already detached
            return self

        # loads everything into memory and sets attached_folder to None
        all_vars = vars(self)

        detached_self = copy.deepcopy(self)
        detached_self.__internal__['attached_folder'] = None

        for var_name in all_vars:
            v = mmap_var_to_memory(getattr(detached_self, var_name))
            setattr(detached_self, var_name, v)

        return detached_self

    def _replace_numpy_placeholders(self, var_info, load_folder, numpy_mmap_mode, stop_on_error,
                                    mmap_vars_list=None):
        # go  over all the loaded variables and replace placeholder numpy arrays with mmap ones
        all_vars = vars(self)
        for v in all_vars:
            if (not v in self.__internal__['skip_saving_vars']) and v != '__internal__' and 'uses_numpy_placeholders' in var_info[v] and var_info[v][
                'uses_numpy_placeholders']:
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

    def set_attached_folder(self, attached_folder=None):
        """
        set attached_folder, this is the folder where the pack is saved to and kept up-to-date-with.
        :param attached_folder:
        :return: None
        """
        assert attached_folder is not None, 'attached_folder need to be specified.'

        if self.__internal__['attached_folder'] is not None and self.__internal__['attached_folder'] != attached_folder:
            print('Attached folder is already set to a different directory and cannot be changed.')
            print('Load a new instance of this class to attach to a different directory.')
            print('Skipping...')
            return
        else:
            self.__internal__['attached_folder'] = attached_folder

    def save(self, save_folder=None, max_dict_keys: int = 1000,
             min_dict_numpy_size: int = 10000, sep_var_min_size: int = 1e4,
             sep_vars: typing.Optional[Union[Dict, List]] = None,
             skip_saving_vars: typing.Set = None):
        """
        Save the pack of variables into the 'attached folder'. This folder must have been already set up for the
        var pack using set_attached_folder() method.

        A subset variables can be marked to be saved in separate files, e.g. so they could later be loaded individually
        or to be excluded from loading the pack.

        :param save_folder: if provided and not equal to attached folder, all the data is loaded into memory and
                            then a new folder is made and a copy of the data is saved to that folder. Save then
                            returns the newly created varpack object, and leaves the original object intact.
        :param max_dict_keys: do not try to replace large numpy arrays with dictionaries with larger than
                              this number of keys. This is mainly to avoid wasting time checking keys in very large
                              dictionaries.
        :param min_dict_numpy_size: minimum number of elements in a numpy array for it to be replaced with
                                    a placeholder objects that points to a separate numpy file.
        :param sep_var_min_size:  minimum total in-memory size (in bytes) for a variable to be saved in a
                                  separate pickle file (so for example could be excluded during load).
        :param sep_vars: a list containing variables that need to be saved in a different pickle file.
                         All numpy arrays are automatically saved in separate .npy files.
        :param skip_saving_vars: a set with the variables to be skipped during save.
        :return: None
        """

        # ToDo: add 'skip resaving' and 'fully skip saving' and 'save only' options to save()

        if skip_saving_vars is not None:
            self.__internal__['skip_saving_vars'] = skip_saving_vars

        if save_folder is None:
            save_folder = self.__internal__['attached_folder']

        if save_folder != self.__internal__['attached_folder']:
            detached_self = self.detach()
            detached_self.set_attached_folder(save_folder)

            # must remeber to include all future input params there!
            detached_self.save(save_folder, max_dict_keys, min_dict_numpy_size, sep_var_min_size,
                                sep_vars, skip_saving_vars)
            return detached_self


        print('Saving variables into the attached folder: ', save_folder)

        assert save_folder is not None, 'attached folder has not yet been set'

        os.makedirs(save_folder, exist_ok=True)

        # variables that cannot be saved as numpy arrays and need to be saved using pickle
        pickle_vars = list()

        obj_vars = vars(self)

        if sep_vars is None:
            sep_vars = set()
        else:
            sep_vars = set(sep_vars)

        for var_name in obj_vars:
            if var_name in self.__internal__['skip_saving_vars']:
                print("- Skipping: " + var_name)
                if var_name in self.__internal__['var_info']:
                    del self.__internal__['var_info'][var_name]
            elif var_name != '__internal__':  # do not save_copy __internal__ variable.

                print("Saving: " + var_name)

                self.__internal__['var_info'][var_name] = dict()
                self.__internal__['var_info'][var_name]['size'] = get_total_obj_size(obj_vars[var_name],
                                                                                     count_mmap_size=False)

                # if the variable is a numpy array then try to save_copy it as .npy
                if isinstance(obj_vars[var_name], np.ndarray):

                    if type(obj_vars[var_name]) is np.memmap:
                        # flush mmap to disk, no need to re-save_copy if
                        # saving to the same folder where it is memory-mapped
                        obj_vars[var_name].flush()

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

                                numpy_array_placeholder = NumpyArrayPlaceholder(obj_vars[var_name][k],
                                                                                save_folder=save_folder,
                                                                                var_name=var_name,
                                                                                key_hash=k.__hash__())

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

        # save_copy the rest of variables as pickle
        pickle_dict = dict()
        for v in pickle_vars:
            pickle_dict[v] = self.__getattribute__(v)

        # save_copy variables that need to have separate files
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

        # a json file with variable info
        with open(os.path.join(save_folder, JSON_FILENAME), 'w') as outfile:
            json.dump(self.__internal__['var_info'], outfile, indent=4)

        # Todo: do this for every mmaped variable right after saving, instead of here for all.
        # this would prevent messing up variables if an error occured during save
        base_folder = self.__internal__['attached_folder']
        if base_folder is None:
            base_folder = save_folder
        self._replace_numpy_placeholders(self.__internal__['var_info'], base_folder,
                                         numpy_mmap_mode=self.__internal__['numpy_mmap_mode'], stop_on_error=True,
                                         mmap_vars_list=None)

        # if data was never loaded, set the loaded folder to the first save_copy location.
        if self.__internal__['attached_folder'] is None:
            self.__internal__['attached_folder'] = save_folder

        return self

    def save_then_copy(self, copy_folder, **kwargs):
        """
        save into a new folder and then sane
        :param copy_folder: the folder into which attached folder content is copied to.
        :param args: same arguments as save()
        :return: None
        """

        self.save(**kwargs)
        from distutils.dir_util import copy_tree
        os.makedirs(copy_folder, exist_ok=True)

        print('Copying the attached folder to:', copy_folder)
        copy_tree(self.__internal__['attached_folder'], copy_folder)

    def load(self, load_folder, numpy_mmap_mode='r+', stop_on_error=True,
             skip_loading=None, keep_loaded_skips=False):
        """
        Attach to a folder and load data from it.
        :param load_folder: the varpack folder to load the variables from.
        :param numpy_mmap_mode: must be 'r+', 'r' or 'c'.
               see https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html
        :param stop_on_error: stop if any errors where encountered during load.
        :param skip_loading: a list/set of variable names to skip loading from the folder.
        :param keep_loaded_skips: whether to keep a variable was in __misc__vars.pickle file even if it is in skipped
                                  list. Default: False
        :return: None
        """

        assert numpy_mmap_mode in ['r+', 'r', 'c'], "numpy_mmap_mode must be 'r+', 'r' or 'c'."

        mmap_vars_list = list()  # the list of variables and dictionary fields that have been numpy memory-mapped

        # read the varpack.json file
        try:
            with open(os.path.join(load_folder, JSON_FILENAME), 'r') as json_file:
                var_info = json.load(json_file)

                self.__internal__['var_info'] = var_info
        except EnvironmentError:
            print('Error when loading ' + JSON_FILENAME + ' file.')
            raise

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
                                if keep_loaded_skips:
                                    print(
                                        'Variable %s was saved in misc. variables file so it was loaded with them.' % v)
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
