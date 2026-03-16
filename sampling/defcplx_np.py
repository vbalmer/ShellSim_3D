import numpy as np

lbd = 0.67

# --- Vectorized math functions (work for scalars and arrays)
def sqrt(x):
    return np.sqrt(np.asarray(x, dtype=complex))

def sin(x):
    return np.sin(np.asarray(x, dtype=complex))

def cos(x):
    return np.cos(np.asarray(x, dtype=complex))

def tan(x):
    return np.tan(np.asarray(x, dtype=complex))

def asin(x):
    return np.arcsin(np.asarray(x, dtype=complex))

def acos(x):
    return np.arccos(np.asarray(x, dtype=complex))

def atan(x):
    return np.arctan(np.asarray(x, dtype=complex))

# --- Custom complex array class
class cplx(np.ndarray):
    """ ----------------------------------- Custom class for complex numbers/arrays --------------------------------
        - syntax: cplx(scalar), cplx(array)
        - wraps numpy ndarray with dtype=complex
        - supports all standard math operations elementwise
        - comparisons operate on real parts (matching original behaviour)
        -------------------------------------------------------------------------------------------------------------"""

    def __new__(cls, data, imag=None):
        if imag is not None:
            data = np.asarray(data, dtype=complex) + 1j * np.asarray(imag, dtype=complex)
        else:
            data = np.asarray(data, dtype=complex)
        return data.view(cls)

    def __array_finalize__(self, obj):
        pass

    # Arithmetic — np.ndarray handles these, but we ensure cplx output
    def __add__(self, x):        return np.ndarray.__add__(self, x).view(cplx)
    def __radd__(self, x):       return np.ndarray.__radd__(self, x).view(cplx)
    def __sub__(self, x):        return np.ndarray.__sub__(self, x).view(cplx)
    def __rsub__(self, x):       return np.ndarray.__rsub__(self, x).view(cplx)
    def __mul__(self, x):        return np.ndarray.__mul__(self, x).view(cplx)
    def __rmul__(self, x):       return np.ndarray.__rmul__(self, x).view(cplx)
    def __truediv__(self, x):    return np.ndarray.__truediv__(self, x).view(cplx)
    def __rtruediv__(self, x):   return np.ndarray.__rtruediv__(self, x).view(cplx)
    def __pow__(self, x):        return np.ndarray.__pow__(self, x).view(cplx)
    def __rpow__(self, x):       return np.ndarray.__rpow__(self, x).view(cplx)

    # Comparisons on real parts (matching original behaviour)
    def __lt__(self, x):  return self.view(np.ndarray).real < np.asarray(x).view(np.ndarray).real
    def __le__(self, x):  return self.view(np.ndarray).real <= np.asarray(x).view(np.ndarray).real
    def __gt__(self, x):  return self.view(np.ndarray).real > np.asarray(x).view(np.ndarray).real
    def __ge__(self, x):  return self.view(np.ndarray).real >= np.asarray(x).view(np.ndarray).real
    def __eq__(self, x):  return self.view(np.ndarray).real == np.asarray(x).view(np.ndarray).real
    def __ne__(self, x):  return self.view(np.ndarray).real != np.asarray(x).view(np.ndarray).real

    def __repr__(self):
        if self.ndim == 0:
            v = self.item()
            return f'cplx({v.real!r}, {v.imag!r})'
        return f'cplx({np.array2string(np.array(self))})'