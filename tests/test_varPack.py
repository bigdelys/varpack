from unittest import TestCase

import os
import sys
# make sure varpack package path is in available
varpack_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if varpack_path not in sys.path:  # add parent dir to paths
    print('Adding ', varpack_path, "to system path.")
    sys.path.append(varpack_path)

import numpy as np
import varpack as vp
import tempfile


class TestVarPack(TestCase):
    def test_basic_save_load(self):
        varpack = vp.Varpack()
        varpack.scalar = 10
        varpack.text = 'test'
        varpack.list = [1, 2, 3, 4]

        # simple saving of small variables
        # should be juts one file named as MISC_VAR_FILENAME
        with tempfile.TemporaryDirectory() as tmpdirname:
            varpack.set_attached_folder(tmpdirname)
            varpack.save()
            filenames = os.listdir(tmpdirname)

            # make sur ethe json file and the misc files are there
            self.assertTrue(len(filenames) == 2)
            self.assertIn(vp.MISC_VAR_FILENAME, filenames)
            self.assertIn(vp.JSON_FILENAME, filenames)

            # add large arrays
            varpack.np_arr = np.zeros((100, 1000, 100))
            varpack.dict_of_np_arr = {'key1': np.zeros((100, 1000, 100)), 'key2': np.zeros((200, 1000, 100))}

            with tempfile.TemporaryDirectory() as tmpdirname2:
                varpack.save_then_copy(tmpdirname2, sep_var_min_size=1000)
                filenames = os.listdir(tmpdirname2)
                print(filenames)

                loaded_varpack = vp.Varpack(tmpdirname2)

                self.assertListEqual(loaded_varpack.list, varpack.list)
                self.assertEqual(loaded_varpack.scalar, varpack.scalar)
                self.assertEqual(loaded_varpack.text, varpack.text)

                self.assertTrue(np.all(loaded_varpack.np_arr - varpack.np_arr==0))

    def test_change_save_load(self):
        varpack = vp.Varpack()
        varpack.scalar = 10
        varpack.text = 'test'
        varpack.list = [1, 2, 3, 4]

        # simple saving of small variables
        # should be juts one file named as MISC_VAR_FILENAME
        with tempfile.TemporaryDirectory() as tmpdirname:
            varpack.set_attached_folder(tmpdirname)

            # add large arrays
            varpack.np_arr = np.zeros((100, 1000, 100))
            varpack.dict_of_np_arr = {'key1': np.zeros((100, 1000, 100)), 'key2': np.zeros((200, 1000, 100))}

            varpack.save()

            varpack.np_arr = varpack.np_arr + 1

            with tempfile.TemporaryDirectory() as tmpdirname2:
                varpack.save_then_copy(tmpdirname2, sep_var_min_size=1000)
                loaded_varpack = vp.Varpack(tmpdirname2)

                self.assertTrue(np.all(loaded_varpack.np_arr - 1 == 0))

    def test_skipped_load(self):
        varpack = vp.Varpack()
        varpack.scalar = 10
        varpack.text = 'test'
        varpack.list = [1, 2, 3, 4]

        # simple saving of small variables
        # should be juts one file named as MISC_VAR_FILENAME
        with tempfile.TemporaryDirectory() as tmpdirname:
            varpack.set_attached_folder(tmpdirname)

            # add large arrays
            varpack.np_arr = np.zeros((100, 1000, 100))
            varpack.dict_of_np_arr = {'key1': np.zeros((100, 1000, 100)), 'key2': np.zeros((200, 1000, 100))}
            varpack.save(sep_var_min_size=1000)

            loaded_varpack = vp.Varpack(tmpdirname, skip_loading=['np_arr'])

            # verify np_arr is not loaded
            self.assertTrue(~hasattr(loaded_varpack, 'np_arr'))

            with tempfile.TemporaryDirectory() as tmpdirname2:
                loaded_varpack.save_then_copy(tmpdirname2)

                loaded_varpack2 = vp.Varpack(tmpdirname2)
                self.assertTrue(hasattr(loaded_varpack2, 'np_arr'))




