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
        # self.e = e                      # strain vector per layer, shape: (n_tot, nl, 3)
        self.e = e*cplx(np.ones_like(e),np.zeros_like(e))

        # Material parameters
        self.ect = mat_dict['ect']
        self.ec0 = mat_dict['ec0']
        self.Ec = mat_dict['Ec']
        self.fct = mat_dict['fct']
        self.fcp = mat_dict['fcp']
        self.tb0 = mat_dict['tb0']
        self.tb1 = mat_dict['tb1']

        self.Esx = mat_dict['Es']
        self.Esy = mat_dict['Es']
        self.Es = mat_dict['Es']
        self.Esh = mat_dict['Esh']
        self.fsy = mat_dict['fsy']
        self.fsu = mat_dict['fsu']

        self.v = constants['nu']
        self.nl = constants['n_layer']

        # Geometrical parameters
        ntot = self.e.shape[0]
        self.rho_x = np.broadcast_to(np.array(constants['rho_x'])[np.newaxis,:,np.newaxis], (ntot, self.nl, 1))
        self.rho_y = np.broadcast_to(np.array(constants['rho_y'])[np.newaxis,:,np.newaxis], (ntot, self.nl, 1))
        self.D     = constants['D']


        # Other constants:
        self.tole = 1e-9
        self.fff = .0001
        self.ff = 0.01
        self.ecsx = -.3 / 1000*0
        self.ecsy = -.05 / 1000*0
        self.tole = 1e-9
        self.lbd = 0.67
    
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
        threshold = self.ect * 0 + tole         
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

        return e1, e3, th, submodel, ecx, ecy, ec1, ec3

    ''' -------------------------------------CONSTITUTIVE LAWS-------------------------------------'''

    def fcs(self, e1):
        """ ---------------------- Calculation concrete strength with softening as a function of e1---------------------
        ----------------------------------------    INPUT (self.): -------------------------------------------------
        - fcp:          Cylinder compressive strength of concrete, int
        - e1:           Principal tensile strain, shape (ntot, nl, 1)

        NOTE: Only Concrete strength considering softening according to CSFM (fib Eq.(7.3- 40) /SIA modified)
        ----------------------------------------------- OUTPUT:-----------------------------------------------------
        - fc:           Concrete strength, shape (ntot, nl, 1)
        -------------------------------------------------------------------------------------------------------------"""

        # Precompute scalar thresholds
        elim1 = 0.2 * self.ec0 / 0.34
        elim2 = 0.22 / 35.75
        kc2 = 1 / (1.2 + 55 * elim2)

        # Allocate kc array
        kc = np.empty_like(e1)

        # Masks for each branch
        mask_neg   = e1 < 0
        mask_low   = ~mask_neg & (e1 <= elim1)
        mask_high  = ~mask_neg & (e1 >= elim2)
        mask_mid   = ~mask_neg & ~mask_low & ~mask_high

        kc[mask_neg]  = 1.0
        kc[mask_low]  = 1 + (elim1 - e1[mask_low]) * 0.001
        kc[mask_high] = 1 / (1.2 + 55 * e1[mask_high])
        kc[mask_mid]  = 1 - (1 - kc2) * (e1[mask_mid] - elim1) / (elim2 - elim1)

        fc = kc * self.fcp

        return fc

    def sc(self, e1):
        """ --------------------------------- Calculation concrete compressive stress-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - e1:           Normal strain in regarded direction (shape: ntot, nl, 1)
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - sc3:          Concrete compressive stress according to fib model code
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (ff*e for numerical stability)
            ---------------------------------------------------------------------------------------------------------"""
        # fc = self.fcs(e1)
        fc = self.fcp
        ec0 = self.ec0 * fc / self.fcp  # scalar

        sc3 = np.empty_like(e1)

        # Masks
        mask_comp       = e1 < 0
        mask_inner      = mask_comp & (np.abs(e1) < ec0)
        mask_outer      = mask_comp & ~mask_inner
        mask_elastic    = ~mask_comp & (e1 <= self.ect)
        mask_tension    = ~mask_comp & ~mask_elastic

        # Compressive inner (parabolic branch)
        eta = -e1[mask_inner] / ec0
        k = self.Ec / (fc / ec0)
        k = max(k, 1)
        sc3[mask_inner] = -fc * (k * eta - eta ** 2) / (1 + (k - 2) * eta)

        # Compressive outer (post-peak branch)
        sc3[mask_outer] = -fc + (e1[mask_outer] + ec0) / 1000

        # Elastic (tension/compression near zero)
        sc3[mask_elastic] = self.Ec * e1[mask_elastic]

        # Tension softening
        sc3[mask_tension] = self.ff * e1[mask_tension]

        return sc3
    
    def ssr(self, e, ec, ecs, srm, rho):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - e:            strain at integration point in one layer, just at locations where there is tension-tension and rho != 0
                            (shape: n_tot, nl*mask, 3)
            - ec:           output from principal (principal concrete strains?), just at locations where there is tension-tension and rho != 0
                            (shape: n_tot, nl*mask, 1)
            - ecs:          constant, currently zero (what is it?)
            - sr:           crack spacing
            - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
            NOTE: Only for cms_klij = 1 (bilinear + TCM)
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - ss:           steel stressses
        -------------------------------------------------------------------------------------------------------------"""

        ssr = self.ssr_tcm_bilin(e+ecs, ec+ecs, srm, rho)
        # ssr shape: (n_tot, nl*mask, 1)

        # 1. Select tbi based on ssr > fsy                              # only required for fixed cracks
        # tbi = np.where(ssr.real > self.fsy, self.tb1, self.tb0)

        # 2. Compute dsk
        # dsk = 4 * tbi / self.D * srm / 4                               # only required for fixed cracks

        # 3. Compute ssk with three branches                             # only required for fixed cracks
        # ssk = np.where(
        #     ssr.real > dsk * 3/2,
        #     ssr - dsk,
        #     np.where(
        #         ssr.real > dsk / 2,
        #         ssr - dsk * (ssr - dsk/2) / dsk,
        #         ssr
        #     )
        # )

        # 4. Catch yielding in compression
        comp_yield_mask = ssr < -self.fsy
        if np.any(comp_yield_mask):
            ssr_comp = self.ss_bilin(e)
            ssr = np.where(comp_yield_mask, ssr_comp, ssr)  
            # ssk = np.where(comp_yield_mask, ssr_comp, ssk)               # only required for fixed cracks

        return ssr

    def ss_bilin(self, e):
        """ ------------------------ Calculation of bare reinforcing steel stress: bilinear-----------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - e:            Strains only at given layers and for one dimension (ntot, nl*masks, 1)
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ss:           Steel stress
        -------------------------------------------------------------------------------------------------------------"""
        esy = self.fsy / self.Es

        ss = np.empty_like(e)

        mask_elastic = np.abs(e) <= esy
        mask_tension = ~mask_elastic & (e > 0)
        mask_compress = ~mask_elastic & (e < 0)

        ss[mask_elastic]  = e[mask_elastic] * self.Es
        ss[mask_tension]  = self.fsy + self.Esh * (e[mask_tension] - esy)
        ss[mask_compress] = -self.fsy + self.Esh * (e[mask_compress] + esy)

        return ss

    def ssr_seelhofer(self, e, ec, srm, rho):
        """ --------------------- Calculate Steel Stress with Seelhofer for non-stabilised crack -----------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - ec: axial concrete compressive strain from mohr's circle of concrete strains
                - srm: Crack spacing: srm = lambda * sr0
                - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel/CFRP stress at the crack according to Seelhofer
            ---------------------------------------------------------------------------------------------------------"""
        # 2 Elastic crack element
        a = 4 * self.tb0 / (self.Es * self.D)
        b = 4 * self.tb0 / (self.Ec * self.D) * rho / (1 - rho) * srm
        c = srm * (ec - e)
        x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        ssr = (ec + 4 * self.tb0 / (self.Ec * self.D) * rho / (1 - rho) * x1 + 4 * self.tb0 / (self.Es * self.D) * x1) * self.Es

        # 3 Elastic-Plastic crack element
        mask_ep = (x1 < srm / 2) & (ssr > self.fsy)

        if np.any(mask_ep):
            ec_ep = ec.copy()
            ec_ep[mask_ep] = ec[mask_ep] + (self.fsy / self.Es - ec[mask_ep]) * self.Es * rho[mask_ep] / (1 - rho[mask_ep]) / self.Ec

            x1_ep = (self.fsy - ec_ep * self.Es) * self.D / (4 * self.tb0)

            c_ep = -e * srm + 2 * ec_ep * (srm / 2 - x1_ep) + (self.fsy / self.Es + ec_ep) * x1_ep
            b_ep = 2 * (self.fsy / self.Es - ec_ep)
            a_ep = 4 * self.tb1 / (self.D * self.Esh)
            discriminant = np.where(mask_ep,b_ep**2 - 4 * a_ep * c_ep,0.0)
            x2_ep = (-b_ep + np.sqrt(discriminant)) / (2 * a_ep)

            ssr_ep = ec_ep * self.Es + x1_ep * 4 * self.tb0 / self.D + x2_ep * 4 * self.tb1 / self.D

            x1[mask_ep]  = x1_ep[mask_ep]
            ssr[mask_ep] = ssr_ep[mask_ep]


        return x1, ssr

    def ssr_tcm_bilin(self, e, ec, srm, rho):
        """ ------------------------------------ Calculate Steel Stress with the TCM -----------------------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e:        normal strain (just at masked locations, shape: (ntot, nl*mask, 1))
                - ec:       axial concrete compressive strain from mohr's circle of concrete strains, shape: (ntot, nl*mask, 1)
                - srm:      Crack spacing: srm = lambda * sr0, shape: (ntot, nl*mask, 1)
                - rho:          rho only at given locations in element and for one dimension (shape: n_tot, nl*mask, 1)
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ss_out:   steel stress at the crack according to TCM, shape: (ntot, nl*mask, 1)
            ---------------------------------------------------------------------------------------------------------"""

        # 1 Seelhofer
        x1, ssr = self.ssr_seelhofer(e, ec, srm, rho)

        # 2. TCM
        mask_tcm = x1 >= srm / 2
        ss_out = np.empty_like(e)
        ss_out[~mask_tcm] = ssr[~mask_tcm]

        # 2.1 Bare steel stress
        st_naked = self.ss_bilin(e)

        # 2.2 Steel stress for fully elastic crack element
        s1 = st_naked + self.tb0 * srm / self.D

        # 2.3 Steel stress for fully plastic element
        s3 = self.fsy + self.Esh * (e - self.fsy / self.Es) + self.tb1 * srm / self.D

        # 2.4 Masks for stress regimes (only where mask_tcm is active)
        mask_elastic  = mask_tcm & (s1 <= self.fsy)
        mask_plastic  = mask_tcm & (s1 > self.fsy) & (s3 - (2 * self.tb1 * srm / self.D) >= self.fsy)
        mask_partial  = mask_tcm & ~mask_elastic & ~mask_plastic

        # 2.4.1 Fully elastic
        ss_out[mask_elastic] = s1[mask_elastic]

        # 2.4.2 Fully plastic
        ss_out[mask_plastic] = s3[mask_plastic]

        # 2.4.3 Partially elastic
        s2 = (self.fsy - self.Es * e) * self.tb1 * srm / self.D * (self.tb0 / self.tb1 - self.Es / self.Esh)
        s2 = s2 + self.Es / self.Esh * self.tb0 * self.tb1 * srm ** 2 / self.D ** 2
        discriminant = np.where(mask_partial,s2,0.0)
        s2 = self.tb0 * srm / self.D - np.sqrt(discriminant)                # to avoid warning in output.
        s2 = self.fsy + 2 * s2 / (self.tb0 / self.tb1 - self.Es / self.Esh)
        ss_out[mask_partial] = s2[mask_partial]

        # 2.5 If stress > ultimate stress, assign ultimate stress
        mask_ultimate = ss_out.real > self.fsu
        ss_out[mask_ultimate] = self.fsu + 100 * e[mask_ultimate]

        return ss_out

    def sr0_vc(self, th):
        """ --------------------------------- Calculate diagonal crack spacing------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - th: Principal direction (shape: ntot, nl, 1)
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - srx, sry: Crack spacing in reinforcement direction (shape: ntot, nl, 1)
        -------------------------------------------------------------------------------------------------------------"""

        # 0.1 Manipulate (mirror) negative angles
        th = np.where(th < 0, -th, th)

        # 1 Initial Assumption
        # 1.1 in x
        mask_rhox = self.rho_x > 1e-9
        denom_x = np.where(mask_rhox, 2 * self.tb0 / self.D * self.rho_x, 1.0)  # safe dummy in false branch, to avoid divide by zero warning.
        srx0_mask = np.where(mask_rhox, self.fct * (1 - self.rho_x) / denom_x, 1e3)
        srxmax = np.where(mask_rhox, self.D / (2 * self.tb0) * self.fsy + self.D / (2 * self.tb1) * (self.fsu - self.fsy), 1e3)
        srx0 = np.minimum(srx0_mask, srxmax)

        # 1.2 in y
        mask_rhoy = self.rho_y > 1e-9
        denom_y = np.where(mask_rhoy, 2 * self.tb0 / self.D * self.rho_y, 1.0)  # safe dummy in false branch, to avoid divide by zero warning.
        sry0_mask = np.where(mask_rhoy, self.fct * (1 - self.rho_y) / denom_y, 1e3)
        srymax = np.where(mask_rhoy, self.D / (2 * self.tb0) * self.fsy + self.D / (2 * self.tb1) * (self.fsu - self.fsy), 1e3)
        sry0 = np.minimum(sry0_mask, srymax)

        # 2 Actual crack spacing as a function of lambda
        sr = self.lbd / ((np.sin(th) / srx0) + (np.cos(th) / sry0))

        # 3 Recalculate spacings in reinforcement directions
        # 3.1 x-direction
        srx_ = np.where(
            np.abs(np.sin(th)) > 0,
            sr / np.sin(th),
            sr / 1e-10
        )
        srx = np.minimum(srx_, srxmax)

        # 3.2 y-direction
        sry_ = np.where(
            np.abs(np.cos(th)) > 0,
            sr / np.cos(th),
            sr / 1e-10
        )
        sry = np.minimum(sry_, srymax)

        return srx, sry


    ''' -------------------------------------COMBINED IN-PLANE STATES-------------------------------------'''


    def sigma_cart_1(self):
        '''
        Calculate linear elastic stresses
        '''

        ex = self.e[:,:,0:1]
        ey = self.e[:,:,1:2]
        gxy = self.e[:,:,2:3]

        E = self.Ec
        v = self.v

        sx_x = E/(1-v**2)*ex
        sx_y = E/(1-v**2)*v*ey
        sx_xy = np.zeros_like(self.e[:,:,0:1])
        sy_x = E/(1-v**2)*v*ex
        sy_y = E/(1-v**2)*ey
        sy_xy = np.zeros_like(self.e[:,:,0:1])
        txy_x = np.zeros_like(self.e[:,:,0:1])
        txy_y = np.zeros_like(self.e[:,:,0:1])
        txy_xy = E/(1-v**2)*(1-v)/2*gxy

        sx = sx_x + sx_y + sx_xy
        sy = sy_x + sy_y + sy_xy
        txy = txy_x + txy_y + txy_xy

        # ssx = ex*self.Esx
        # ssy = ey*self.Esy
        # sc3 = e3*self.Ec
        # sc1 = e1*self.Ec

        return sx, sy, txy

    def sigma_cart_31(self, mask):
        """
        in-plane stresses for compression - compression
        
        Args:
            self.e:     Strains (n_tot, nl, 3)
            mask:       Location where submodel = 3 (20,)
        
        Returns:
            sx, sy, txy Stresses (n_tot, nl*mask, 3)    only at masked locations.
        """
        
        e = self.e
        _,_,th,_,_,_,ec1,ec3 = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3]) 
        
        # 2 Steel Contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        ssx[mask&mask_rhox] = self.ss_bilin(e[:,:,0:1][mask&mask_rhox])

        ssy = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhoy = (self.rho_y != 0)
        ssy[mask&mask_rhoy] = self.ss_bilin(e[:,:,1:2][mask&mask_rhoy])

        # 3 CFRP Contribution (skipped)

        # 4 Concrete Constitutive Matrix
        # 4.1 Principal Concrete stresses
        sc1 = self.sc(ec1)
        sc3 = self.sc(ec3)

        # 4.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: assign "zero" (not possible in compression-compression case)
        scx = sc3 * np.cos(th) ** 2 + sc1 * np.sin(th) ** 2
        scy = sc3 * np.sin(th) ** 2 + sc1 * np.cos(th) ** 2
        txy = (sc1 - sc3) * np.sin(th) * np.cos(th)

        # 5 Assemble resulting stresses
        # 5.1 Stress assembly
        sx = 1/(1-self.v**2)*(scx[mask] + self.v*scy[mask])*(1-self.rho_x[mask]) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy[mask] + self.v*scx[mask])*(1-self.rho_y[mask]) + self.rho_y[mask]*ssy[mask]

        # 6 Manipulate parameters that do not occurr in compression-compression
        # self.sr0_vc()
        # self.dn = 0
        # self.dt = 0

        return sx, sy, txy[mask]

    def sigma_cart_32(self, mask):

        e = self.e
        _,_,th,_,ecx,ecy,ec1,ec3 = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3])  

        srx, sry = self.sr0_vc(th)

        # 2 Steel Contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        mask_e = e[:,:,0:1] > ecx
        mask_1_x = mask & mask_rhox & mask_e
        ssx[mask_1_x] = self.ssr(self.e[:,:,0:1][mask_1_x], ecx[mask_1_x], self.ecsx, 
                                srx[mask_1_x], self.rho_x[mask_1_x])
        mask_2_x = mask & mask_rhox & ~mask_e
        ssx[mask_2_x] = self.ss_bilin(self.e[:,:,0:1][mask_2_x])

        ssy = np.zeros((e.shape[0], e.shape[1],1), dtype=np.complex64)
        mask_rhoy = (self.rho_y != 0)
        mask_e_y = e[:,:,1:2] > ecy
        mask_1_y = mask & mask_rhoy & mask_e_y
        ssy[mask_1_y] = self.ssr(self.e[:,:,1:2][mask_1_y], ecy[mask_1_y], self.ecsy, 
                                sry[mask_1_y], self.rho_y[mask_1_y])
        mask_2_y = mask & mask_rhoy & ~mask_e
        ssy[mask_2_y] = self.ss_bilin(self.e[:,:,1:2][mask_2_y])

        # 3 CFRP Contribution (skipped)

        # 4 Concrete Constitutive Matrix
        # 4.1 Principal Concrete stresses
        sc1 = self.sc(ec1)
        sc3 = self.sc(ec3)

        # 4.2 Concrete secant stiffness matrix
        scx = sc3 * np.cos(th) ** 2 + sc1 * np.sin(th) ** 2
        scy = sc3 * np.sin(th) ** 2 + sc1 * np.cos(th) ** 2
        txy = (sc1 - sc3) * np.sin(th) * np.cos(th)

        # 5 Assemble resulting stresses
        sx = 1/(1-self.v**2)*(scx[mask] + self.v*scy[mask]) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy[mask] + self.v*scx[mask]) + self.rho_y[mask]*ssy[mask]

        return sx, sy, txy[mask]

    def sigma_cart_33(self, mask):
        """
        in-plane stresses for tension - tension
        
        Args:
            self.e:     Strains (n_tot, nl, 3)
            mask:       Location where submodel = 3 (20,)
        
        Returns:
            sx, sy, txy Stresses (n_tot, nl*mask, 3)    only at masked locations.
        """

        e = self.e
        _,_,th,_,ecx,ecy,_,_ = self.principal(e[:,:,0:1], e[:,:,1:2], e[:,:,2:3])      
        
        # 1 Crack spacing
        srx, sry = self.sr0_vc(th)

        # 2 Steel contribution
        ssx = np.zeros((e.shape[0], e.shape[1],1), dtype = np.complex64)
        mask_rhox = (self.rho_x != 0)
        ssx[mask&mask_rhox] = self.ssr(self.e[:,:,0:1][mask&mask_rhox], ecx[mask&mask_rhox], self.ecsx, 
                                       srx[mask&mask_rhox], self.rho_x[mask&mask_rhox])

        ssy = np.zeros((self.e.shape[0], self.e.shape[1],1), dtype = np.complex64)
        mask_rhoy = (self.rho_y != 0)
        ssy[mask&mask_rhoy] = self.ssr(self.e[:,:,1:2][mask&mask_rhoy], ecy[mask&mask_rhoy], self.ecsy, 
                                       sry[mask&mask_rhoy], self.rho_y[mask&mask_rhoy])

        # 3 - CFRP contribution (skipeed)

        # 4 - Concrete contribution
        ex = e[:,:,0:1][mask]                   # only use strains in those layers, where given submodel applies.
        ey = e[:,:,1:2][mask]
        gxy = e[:,:,2:3][mask]
        scx = self.ff*ex
        scy = self.ff*ey

        # 5 - Resulting stresses
        # rho_x = self.rho_x[np.newaxis,:,np.newaxis]
        # rho_y = self.rho_y[np.newaxis,:,np.newaxis]
        sx = 1/(1-self.v**2)*(scx + self.v*scy) + self.rho_x[mask]*ssx[mask]
        sy = 1/(1-self.v**2)*(scy + self.v*scx) + self.rho_y[mask]*ssy[mask]
        txy = self.ff/2*gxy

        return sx, sy, txy

    ''' -------------------------------------OUTPUT (STRESSES)-------------------------------------'''

    def out(self, sprev_klij = 0, do_cracked=True):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - self.e:       [ex_klij,ey_klij,gxy_klij] strain state in integration point
            - sprev:        set s_prev to zero everywhere. Should not be required as cm_klij is never 4 (fixed cracks)
            - do_cracked:   set to True also in previous file (as fixed parameter)
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - s:            [sx_klij,sy_klij,txy_klij] stress state in integration point
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Assign Strain Values
        # ex = self.e[:,:,0:1]*cplx(np.ones_like(self.e[:,:,0:1]),np.zeros_like(self.e[:,:,1:2]))
        # ey = self.e[:,:,1:2]*cplx(np.ones_like(self.e[:,:,0:1]),np.zeros_like(self.e[:,:,1:2]))
        # gxy = self.e[:,:,2:3]*cplx(np.ones_like(self.e[:,:,0:1]),np.zeros_like(self.e[:,:,1:2]))
        ex = self.e[:,:,0:1]
        ey = self.e[:,:,1:2]
        gxy= self.e[:,:,2:3]

        # 2 Calculate Stresses based on given constitutive model and strain state
        e1, _, _, submodel,_,_,_,_ = self.principal(ex, ey, gxy)

        if do_cracked == False:
            raise UserWarning('Not tested for do_cracked == False')
            if hasattr(sprev_klij,'crackflag'):
                crackflag = 1
                pass
            elif e1*self.Ec < self.fct:
                e1, e3, th, submodel,_,_ = self.principal(ex, ey, gxy)
                self.cm_klij = 1
            else:
                crackflag = 1

        if self.cm_klij == 1:
            sx, sy, txy = self.sigma_cart_1()
        elif self.cm_klij == 3:
            n_tot = self.e.shape[0]
            mask3 = (submodel == 3)
            mask1 = (submodel == 1)
            mask2 = ~mask3 & ~mask1

            sx  = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)
            sy  = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)
            txy = np.zeros((n_tot, self.nl, 1), dtype = np.complex64)

            sx[mask3], sy[mask3], txy[mask3] = self.sigma_cart_33(mask3)
            sx[mask1], sy[mask1], txy[mask1] = self.sigma_cart_31(mask1)
            sx[mask2], sy[mask2], txy[mask2] = self.sigma_cart_32(mask2)


        # 3 Output
        # rho_x = self.rho_x[np.newaxis, :]
        mask = np.abs(txy) < self.fff * self.rho_x * self.Esx / 2 * np.abs(gxy)
        txy = np.where(mask, self.fff * self.rho_x * self.Esx / 2 * gxy, txy)
        
        s = np.stack((sx, sy, txy), axis = 2)

        return s
