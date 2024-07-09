import numpy as np


# Let's make two numbers with very similar magnitude:

x = 1.48234
y = 1.48235


# Now let's compute their difference in double precision:

x_dbl = np.float64(x)
y_dbl = np.float64(y)
diff_dbl = x_dbl-y_dbl

print(repr(diff_dbl))


# * What would the correct result be?
# * What has happened here?

# -------------
# Can you predict what will happen in single precision?

x_sng = np.float32(x)
y_sng = np.float32(y)
diff_sng = x_sng-y_sng

print(diff_sng)
