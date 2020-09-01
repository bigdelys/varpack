# Varpack

Vidpack class enables fast saving and loading of a set of Python variables to disk. It uses numpy memory-mapped arrays under the hood but adds a number of convenient features. 

## Features:

* Automatically examines Python dictionaries (first level only) for numpy arrays that could be saved as individual memory-mapped files. These are later transparently loaded back into the dictionary using placeholder classes.
 
* During save, numpy variables are saved as in numpy array format `.npy` while other variables are grouped together and saved as pickle.

* User can specify which variables are to be ignored (not loaded) when loading the variable set.

* User can specify which variables are to be saved into separate pickle files, so they could be later skipped, in a time-efficient way, during loading.

## Examples

```python

import varpack
import numpy as np

# create Varpack object 
vp = varpack.Varpack()

# assign some values to it as class properties
vp.var1 = np.ones((100, 1000)) 
vp.var2 = 20
vp.var3 = 'some string'
vp.var4 = {'key1': np.ones((200, 1000)), 'key2': np.zeros((50, 2000))}

vp.set_attached_folder([folder to save all the variables, will be created if needed])

vp.save()

```

Now if we look into the save folder, we see these files:
- `varpack.json`: contains JSON-encoded metadata (size, etc) about saved variables in the vidpack folder.
- `var-44806632781637889188.npy`: a numpy array containing the value of `key1` key in `var4` dictionary.
- `var4-1507539616216130811.npy`: a numpy array containing the value of `key2` key in `var4` dictionary.
- `var1.npy`: a numpy array containing `var1`.
- `__misc_vars__.pickle`: a pickle file containing all non-numpy variables (here, `var3`) 

We can now load the data:

```python
vp2 = varpack.Varpack([the folder where data was saved])
```

We see:

    Loading from  [Varpack folder]
    The following numpy variables have been memory-mapped with option r+:
        ['var1', 'var4[key1]', 'var4[key2]']

and `vp2` now contains the same variables but the large arrays are now memory-mapped.