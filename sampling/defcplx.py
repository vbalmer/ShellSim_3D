import cmath
lbd = 0.67

def sqrt(x):
    return cplx(cmath.sqrt(x))


def sin(x):
    return cplx(cmath.sin(x))


def cos(x):
    return cplx(cmath.cos(x))


def tan(x):
    return cplx(cmath.tan(x))


def asin(x):
    return cplx(cmath.asin(x))


def acos(x):
    return cplx(cmath.acos(x))


def atan(x):
    return cplx(cmath.atan(x))

# ---

class cplx(complex):
    """ ----------------------------------- Custom class for complex numbers ----------------------------------------
        - syntax: cplx(real,imag)
        - supported mathematical operations: "+", "-", "*", "/", "**"
        - supported comparisons: ">", ">=", "==", "<=", "<", "!="
        -------------------------------------------------------------------------------------------------------------"""
    def __repr__(self):
        return 'cplx(%r, %r)' % (self.real, self.imag)

    def __add__(self,x):
        return cplx(complex.__add__(self, x))

    def __radd__(self,x):
        return cplx(complex.__radd__(self, x))

    def __sub__(self,x):
        return cplx(complex.__sub__(self, x))

    def __rsub__(self,x):
        return cplx(complex.__rsub__(self, x))

    def __mul__(self,x):
        return cplx(complex.__mul__(self, x))

    def __rmul__(self,x):
        return cplx(complex.__rmul__(self, x))

    def __truediv__(self,x):
        return cplx(complex.__truediv__(self,x))

    def __rtruediv__(self,x):
        return cplx(complex.__rtruediv__(self,x))

    def __pow__(self,x):
        return cplx(complex.__pow__(self, x))

    def __rpow__(self,x):
        return cplx(complex.__rpow__(self, x))

    def __lt__(self,x):
        return self.real < x.real

    def __le__(self,x):
        return self.real <= x.real

    def __gt__(self,x):
        return self.real > x.real

    def __ge__(self,x):
        return self.real >= x.real

    def __eq__(self,x):
        return self.real == x.real

    def __ne__(self,x):
        return self.real != x.real

