# constitutive_laws in vectorised form according to stresses_mixedreinf (Andreas)
# vb, 10.03.2026

# note: Only the material classes relevant for my sampling methodology are transferred, namely: 
#       cm_klij = 3 (reinforced concrete)
#       cmcc_klij = 1 (fib Model code)
#       cmcs_klij = 1 (concrete strength considering softening according to CSFM)
#       cms_klij = 1 (tension chord model with bilinear bare relationship)
#       cmtn_klij = None (not required here)


import numpy as np
from defcplx_np import *
from defcplx_np import cplx


class ConstitutiveLaws():
    def __init__(self, e, constants, mat_dict, cm_klij = 3):
        # General
        self.cm_klij = cm_klij          # material type: 1 - lin.el., 3 - nonlinear RC
        self.e = e                      # strain vector per layer, shape: (n_tot, nl, 3)

        # Material parameters
        self.ect = mat_dict['ect']
        self.Ec = mat_dict['Ec']
        self.fct = mat_dict['fct']

        self.Esx = mat_dict['Es']
        self.Esy = mat_dict['Es']

        self.v = constants['nu']

        # Geometrical parameters
        self.rho_x = constants['rho_x']      #TODO: ensure that rho is not the same for every layer? define rho as vector directly at the start
        self.rho_y = constants['rho_y']


        # Other constants:
        self.tole = 1e-9
        self.ff = .0001


    
    def principal(self, ex, ey, gxy):
        """
        Vectorised principal strains and direction.
        Args:
            ex, ey, gxy:  shape (n_tot, nl, 3)
        Returns:
            e1, e3:       shape (n_tot, nl, 3)
            th:           shape (n_tot, nl, 3)
            submodel:     shape (n_tot, nl, 3), int (1, 2, or 3)
        """
        # 1 Mohr's Circle
        re = 0.5 * np.sqrt((ex - ey)**2 + gxy**2)
        me = 0.5 * (ex + ey)
        e1 = me + re
        e3 = me - re

        # 2 Principal direction th
        # Default: atan(gxy / (2*(e1-ex)))
        denom = 2 * (e1 - ex)
        th = np.arctan(np.where(np.abs(denom) > 1e-10, gxy / denom, 0.0))

        # Override where e1 ≈ ex (degenerate cases)
        near_ex = np.abs(e1 - ex) < 1e-10
        th = np.where(near_ex & (ex > ey),  np.pi/2 - 1e-10, th)
        th = np.where(near_ex & (ex < ey),  1e-10,            th)
        th = np.where(near_ex & (ex == ey), np.pi/4,          th)

        # 3 Submodel
        tole = self.tole
        threshold = self.ect * 0 + tole          # scalar (as in original)
        submodel = np.where(
            (e1 > threshold) & (e3 > threshold), 3,
            np.where(
                (e1 <= threshold) & (e3 <= threshold), 1,
                2
            )
        ).astype(int)

        # 4 Concrete strains
        thc = th                                 # rotating cracks: thc = th
        ec3 = e3
        ec1 = np.where(e1 < tole, e1, 0.0)

        ecx = np.where(ec3 < 0, ec1 + (ec3 - ec1) * np.cos(thc)**2, 0.0)
        ecy = np.where(ec3 < 0, ec1 + (ec3 - ec1) * np.sin(thc)**2, 0.0)

        return e1, e3, th, submodel


    def sigma_cart_1(self):
        '''
        Calculate linear elastic stresses
        '''

        ex = self.e[:,:,0]
        ey = self.e[:,:,1]
        gxy = self.e[:,:,2]

        E = self.Ec
        v = self.v

        sx_x = E/(1-v**2)*ex
        sx_y = E/(1-v**2)*v*ey
        sx_xy = np.zeros_like(self.e[:,:,0])
        sy_x = E/(1-v**2)*v*ex
        sy_y = E/(1-v**2)*ey
        sy_xy = np.zeros_like(self.e[:,:,0])
        txy_x = np.zeros_like(self.e[:,:,0])
        txy_y = np.zeros_like(self.e[:,:,0])
        txy_xy = E/(1-v**2)*(1-v)/2*gxy

        sx = sx_x + sx_y + sx_xy
        sy = sy_x + sy_y + sy_xy
        txy = txy_x + txy_y + txy_xy

        # ssx = ex*self.Esx
        # ssy = ey*self.Esy
        # sc3 = e3*self.Ec
        # sc1 = e1*self.Ec

        return sx, sy, txy


    def out(self, sprev_klij = None, do_cracked=True):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - e:        [ex_klij,ey_klij,gxy_klij,gxz_klij,gyz_klij] strain state in integration point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - s:        [sx_klij,sy_klij,txy_klij,txz_klij,tyz_klij] stress state in integration point
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Assign Strain Values
        ex = self.e[:,:,0]*cplx(np.ones_like(self.e[:,:,0]),np.zeros_like(self.e[:,:,1]))
        ey = self.e[:,:,1]*cplx(np.ones_like(self.e[:,:,0]),np.zeros_like(self.e[:,:,1]))
        gxy = self.e[:,:,2]*cplx(np.ones_like(self.e[:,:,0]),np.zeros_like(self.e[:,:,1]))

        # 2 Calculate Stresses based on given constitutive model and strain state
        e1, e3, th, submodel = self.principal(ex, ey, gxy)

        if do_cracked == False:
            raise UserWarning('Not tested for do_cracked == False')
            if hasattr(sprev_klij,'crackflag'):
                crackflag = 1
                pass
            elif e1*self.Ec < self.fct:
                e1, e3, th, submodel = self.principal(ex, ey, gxy)
                self.cm_klij = 1
            else:
                crackflag = 1

        if self.cm_klij == 1:
            sx, sy, txy = self.sigma_cart_1()
        elif self.cm_klij == 3:
            if submodel == 3:
                sx, sy, txy = self.sigma_cart_33()
            elif submodel == 1:
                sx, sy, txy = self.sigma_cart_31()
            else:
                sx, sy, txy = self.sigma_cart_32()


        # 3 Output
        mask = np.abs(txy) < self.ff * self.rho_x * self.Esx / 2 * np.abs(gxy)
        txy = np.where(mask, self.ff * self.rho_x * self.Esx / 2 * gxy, txy)
        
        s = np.stack((sx, sy, txy), axis = 2)

        return s
