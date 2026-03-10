# constitutive_laws in vectorised form according to stresses_mixedreinf (Andreas)
# vb, 10.03.2026

# note: Only the material classes relevant for my sampling methodology are transferred, namely: 
#       cm_klij = 3 (reinforced concrete)
#       cmcc_klij = 1 (fib Model code)
#       cmcs_klij = 1 (concrete strength considering softening according to CSFM)
#       cms_klij = 1 (tension chord model with bilinear bare relationship)
#       cmtn_klij = None (not required here)


import numpy as np
from defcplx import *
from defcplx import cplx


class ConstitutiveLaws():
    def __init__(self, constants, mat_dict):
        # self.E = constants['E']
        # self.nu = constants['nu']


        self.ect = mat_dict['ect']
        
        # constants:
        self.tole = 1e-9


    
    def principal(self, ex, ey, gxy):
        """ ------------------------------ Calculation of Principal Strains and Direction ------------------------------
            ----------------------------------------    INPUT: -------------------------------------------------
            - ex, ey, gxy:  In plane strains
            ------------------------------------------- OUTPUT:-------------------------------------------------
            - e1, e3:       Principal strains
            - th:           Principal direction (in range -pi/2 to pi/2
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Mohr's Circle of imposed total strains

        re = 1/2*np.sqrt((ex - ey) ** 2 + gxy ** 2)
        me = (ex + ey) * 1/2

        e1 = me  + re
        e3 = me  - re

        if abs(self.e1 - ex) < 10**-10:
            if ex > ey:
                th = np.pi/2-10**(-10)
            elif ex < ey:
                th = 10**(-10)
            elif ex == ey:
                th = np.pi/4
        else:
            th = np.atan(gxy / (2 * (e1 - ex)))

        # 1.1 Correct theta to not be exactly 0 or 90° (optional)
        # if abs(self.th) > 0:
            # if abs(self.th)>pi/2*89/90:
            #     self.th = self.th/(abs(self.th))*89/90*pi/2
            #
            # if abs(self.th) < pi/2*1/90:
            #     self.th = self.th / (abs(self.th)) * 1 / 90 * pi / 2

        # 2 Submodel based on imposed strain state
        if e1 > self.ect*0+self.tole and e3 > self.ect*0+self.tole:
            submodel = 3
        elif e1 <= self.ect*0+self.tole and e3 <= self.ect*0+self.tole:
            submodel = 1
        else:
            submodel = 2

        # 3 Mohr's Circle of concrete strains
        # 3.1 Crack inclinations depending on rotating or fixed cracks (here: always rotating)
        thr = th
        thc = th

        # 3.2 Concrete strains in n-t and x-y. For Fixed cracks: Starting values for iteration.
        ec3 = e3
        ec1 = 0
        if e1 < self.tole:
            ec1 = e1
        gctn = 0
        ecn = 0

        if ec3 < 0:
            ecx = ec1 + (ec3-ec1)*np.cos(thc)**2
            ecy = ec1 + (ec3-ec1)*np.sin(thc)**2

        else:
            ecx = 0
            ecy = 0

        return e1, e3, th


    def out(self, e):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - e:        [ex_klij,ey_klij,gxy_klij,gxz_klij,gyz_klij] strain state in integration point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - s:        [sx_klij,sy_klij,txy_klij,txz_klij,tyz_klij] stress state in integration point
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Assign Strain Values
        self.ex = e[0]*cplx(1,0)
        self.ey = e[1]*cplx(1,0)
        self.gxy = e[2]*cplx(1,0)
        self.gxz = e[3]*cplx(1,0)
        self.gyz = e[4]*cplx(1,0)

        # 2 Calculate Stresses based on given constitutive model and strain state
        self.principal()

        # Flag
        # if do_cracked == False:
        #     if self.e1 * self.Ec < self.fct * 1.5:
        #         self.cm_klij = 1

        if do_cracked == False:
            if hasattr(self.sprev_klij,'crackflag'):
                self.crackflag = 1
                pass
            elif self.e1*self.Ec < self.fct:
                self.cm_klij = 1
                self.principal()
            else:
                self.crackflag = 1

        if self.cm_klij == 1:
            self.sigma_cart_1()
        elif self.cm_klij == 3:
            if self.submodel == 3:
                self.sigma_cart_33()
            elif self.submodel == 1:
                self.sigma_cart_31()
            else:
                self.sigma_cart_32()
        self.sigma_shear()

        # # 3 Output
        # # Flag
        # ff = .0001
        # # if abs(self.txy)<ff*self.Ec/2*abs(self.gxy):
        # #     self.txy = ff*self.Ec/2*self.gxy

        # if abs(self.txy)<ff*self.rhox*self.Esx/2*abs(self.gxy):
        #     self.txy = ff*self.rhox * self.Esx / 2 * self.gxy


        # self.s = [self.sx,self.sy,self.txy,self.txz,self.tyz]

        # 3 Output
        ff=self.Ec/100
        if abs(self.txy.imag) < abs(self.gxy.imag)*ff/2:
            self.txy = cplx(self.txy.real,self.gxy.imag*ff/2)
        # if abs(self.sy.imag) < abs(self.ey.imag)*ff:
        #     self.sy = cplx(self.sy.real,self.ey.imag*ff)
        s = [self.sx,self.sy,self.txy,self.txz,self.tyz]

        return s
