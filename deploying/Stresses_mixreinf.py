import numpy as np
from math import pi
# from math import e as euler
from cmath import e as euler
import matplotlib.pyplot as plt
# lbd = 0.6234
# lbd = 1
lbd = .67
ff = .01
# Flag
ff_e3 = 1e-4*0    # Kann geändert werden
ff_ev = 1
tole = 1e-9
do_cracked = True  # True oder False
do_itkin = False
doplot = False
do_char = 0

# Flag
# ecsx = -.23 / 1000
ecsx = -.3 / 1000*0
ecsy = -.05 / 1000*0

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

import sys
# sys.path.insert(1,'C:/Users/naesboma/00_an/FEM_Q/01_Code')
from defcplx import *
from defcplx import cplx

class stress():
    def __init__(self,sprev_klij,cm_klij,l, k, i, j, MAT,GEOM,cmcc_klij = 1,cmcs_klij = 1,cms_klij = 1,cmtn_klij = 1, count = [0, 0, 0]):
        """ --------------------------------- Initiate instance of stress class-----------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - sprev_klij     Stress state in integration point klij from previous iteration

            - cm_klij        Material Model to be applied
                             1 = Linear elastic
                             3 = CMM with rotating, stress-free cracks
                             4 = CMM with fixed, interlocked cracks
                             10 = glass

            - cmcc_klij:     Constitutive Model for Concrete in Compression
                             1 = Stress - strain relation according to
                                 fib Model Code with adaptions according to CSFM
                             2 = Parabolic stress - strain relation according to
                                 Kaufmann (1998)
                             3 = Linear elastic-perfectly plastic stress - strain relationship
                             4 = Quadratic stress - strain relationship according to Sargin

            - cmcs_klij:     Strain softening law for concrete
                             0 = Concrete strength considering no softening
                             1 = Concrete strength considering softening according to CSFM (fib/SIA modified)
                             2 = Concrete strength considering softening according to Kaufmann (1998)
                             3 = Concrete strength considering softening according to Vecchio & Collins (1986)

            - cms_klij:      Constitutive Model for Steel in tension
                             0 = Bilinear stress-strain relationship with no tension stiffening
                             1 = TCM (or POM) with bilinear bare relationship
                             2 = TCM with cold-worked bare relationship according to p. 11/89 Diss Alvarez
                             3 = TCM with yielding plateau according to p.89 ff. Diss Alvarez
                             4 = TCM with custom defined sigma-epsilon and tau-delta relationships
                                 (numerical tension stiffening)

            - cmtn_klij:     Aggregate interlock Model for normal and shear stresses in the crack
                             1 = RCM (Bazant and Gambarova 1980, later refined by Gambarova and Karakoc 1983)
                             2 = CDM (Li and Maekawa 1989)
                             3 = TPM (Walraven 1981)

            - MAT:     Material Information for Integration Point
            - GEOM:    Geometry Information for Integration Point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - Material, geometry and strain information of instance
        -------------------------------------------------------------------------------------------------------------"""
        self.sprev_klij = sprev_klij
        self.cm_klij = cm_klij
        self.l=l

        self.Ec = MAT[0]
        self.v = MAT[1]
        self.fcp = MAT[2]
        self.fct = MAT[3]
        self.ec0 = MAT[4]
        self.ect = MAT[5]
        self.Dmax = MAT[6]

        self.Esx = MAT[7]
        self.Eshx = MAT[8]
        self.fsyx = MAT[9]
        self.fsux = MAT[10]
        self.Esy = MAT[11]
        self.Eshy = MAT[12]
        self.fsyy = MAT[13]
        self.fsuy = MAT[14]
        self.tb0 = MAT[15]
        self.tb1 = MAT[16]

        self.Epx = MAT[17]
        self.Epy = MAT[18]
        self.tbp0 = MAT[19]
        self.tbp1 = MAT[20]
        self.ebp1 = MAT[21]
        self.fpux = MAT[22]
        self.fpuy = MAT[23]

        self.Ec2 = MAT[24]
        self.vc2 = MAT[25]

        self.rhox = GEOM[0]
        self.rhoy = GEOM[1]
        self.dx = GEOM[2]
        self.dy = GEOM[3]
        self.spacx = GEOM[4]
        self.spacy = GEOM[5]

        self.rhopx = GEOM[6]
        self.rhopy = GEOM[7]
        self.dpx = GEOM[8]
        self.dpy = GEOM[9]

        self.t = GEOM[10]
        self.t2 = GEOM[11]

        self.cmcc_klij = cmcc_klij
        self.cmcs_klij = cmcs_klij
        self.cms_klij  =  cms_klij
        self.cmtn_klij = cmtn_klij
        self.count = count

        self.k = k
        self.i = i
        self.j = j
        self.l = l

    # def principal(self):
    #     """ ------------------------------ Calculation of Principal Strains and Direction ------------------------------
    #         ----------------------------------------    INPUT (self.): -------------------------------------------------
    #         - ex, ey, gxy:  In plane strains
    #         ------------------------------------------- OUTPUT (self.):-------------------------------------------------
    #         - e1, e3:       Principal strains
    #         - th:           Principal direction (in range -pi/2 to pi/2
    #     -------------------------------------------------------------------------------------------------------------"""
    #     ex = self.ex
    #     ey = self.ey
    #     gxy = self.gxy
    #     r = 1/2*sqrt((ex - ey) ** 2 + gxy ** 2)
    #     m = (ex + ey) *1/2

    #     self.e1 = m + r
    #     self.e3 = m - r
    #     if abs(gxy) < 10**-8:
    #         if ex > ey:
    #             self.th = pi/2-10**(-10)
    #         elif ex < ey:
    #             self.th = 10**(-10)
    #         elif ex == ey:
    #             self.th = pi/4
    #     else:
    #         self.th = atan(gxy / (2 * (self.e1 - ex)))

    def principal(self):
        """ ------------------------------ Calculation of Principal Strains and Direction ------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - ex, ey, gxy:  In plane strains
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - e1, e3:       Principal strains
            - th:           Principal direction (in range -pi/2 to pi/2
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Mohr's Circle of imposed total strains
        ex = self.ex
        ey = self.ey
        gxy = self.gxy
        # print(ex)
        self.re = 1/2*sqrt((ex - ey) ** 2 + gxy ** 2)
        self.me = (ex + ey) * 1/2

        self.e1 = self.me  + self.re
        self.e3 = self.me  - self.re

        if abs(self.e1 - ex) < 10**-10:
            if ex > ey:
                self.th = pi/2-10**(-10)
            elif ex < ey:
                self.th = 10**(-10)
            elif ex == ey:
                self.th = pi/4
        else:
            self.th = atan(gxy / (2 * (self.e1 - ex)))

        # 1.1 Correct theta to not be exactly 0 or 90° (optional)
        # if abs(self.th) > 0:
            # if abs(self.th)>pi/2*89/90:
            #     self.th = self.th/(abs(self.th))*89/90*pi/2
            #
            # if abs(self.th) < pi/2*1/90:
            #     self.th = self.th / (abs(self.th)) * 1 / 90 * pi / 2

        # 2 Submodel based on imposed strain state
        if self.e1 > self.ect*0+tole and self.e3 > self.ect*0+tole:
            self.submodel = 3
        elif self.e1 <= self.ect*0+tole and self.e3 <= self.ect*0+tole:
            self.submodel = 1
        else:
            self.submodel = 2

        # 3 Mohr's Circle of concrete strains
        # 3.1 Crack inclinations depending on rotating or fixed cracks
        if self.cm_klij == 4 and self.submodel == 2 and self.sprev_klij.submodel == 2:
            self.thr = self.sprev_klij.thr.real
            self.thc = self.sprev_klij.thc.real  # Starting value for iteration
        else:
            self.thr = self.th
            self.thc = self.th

        # 3.2 Concrete strains in n-t and x-y. For Fixed cracks: Starting values for iteration.
        self.ec3 = self.e3
        self.ec1 = 0
        if self.e1 < tole:
            self.ec1 = self.e1
        self.gctn = 0
        self.ecn = 0

        if self.ec3 < 0:
            self.ecx = self.ec1 + (self.ec3-self.ec1)*cos(self.thc)**2
            self.ecy = self.ec1 + (self.ec3-self.ec1)*sin(self.thc)**2

        else:
            self.ecx = 0
            self.ecy = 0

    def crack_kin(self,testmethod = False):
        """ ---------------------- Crack kinematics principal directions and aggregate interlock -----------------------
            ---------------------------------------------    INPUT: ----------------------------------------------------
            - (self.) ex, ey, gxy:  In plane strains
            - (self.) s_prev: crack direction from previous iteration
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - Crack kinematics, aggregate interlock stresses, strain and stress components of crack and concrete
        -------------------------------------------------------------------------------------------------------------"""

        # --------------------------------------------------------------------------------------------------------------
        # 0 Auxiliary Functions
        # --------------------------------------------------------------------------------------------------------------
        def plot_mohr():
            """ -------------------------------- Plot Mohr's Circles for Validation  --------------------------------"""
            fig1, ax = plt.subplots(1, 2)
            fig1.set_figheight(8)
            fig1.set_figwidth(16)

            circle1 = plt.Circle((self.me, 0), self.re, color='r', fill=False)
            circle1k = plt.Circle(((self.ec1k + self.ec3k) / 2, 0), (self.ec1k - self.ec3k) / 2, color='gray',
                                  fill=False)
            ax[0].add_patch(circle1)
            ax[0].add_patch(circle1k)
            ax[0].scatter(self.en, self.gtn / 2, color='r')
            ax[0].annotate('$N$', xy=(self.en.real, self.gtn.real / 2), color='r')
            ax[0].scatter(self.et, -self.gtn / 2, color='b')
            ax[0].annotate('$T$', xy=(self.et.real, -self.gtn.real / 2), color='b')
            ax[0].scatter(self.ex, self.gxy / 2, color='k')
            ax[0].annotate('$X$', xy=(self.ex.real, self.gxy.real / 2), color='k')
            ax[0].scatter(self.ey, -self.gxy / 2, color='gray')
            ax[0].annotate('$Y$', xy=(self.ey.real, -self.gxy.real / 2), color='gray')
            ax[0].scatter(self.ecnk, self.gctnk / 2, color='gray')
            # ax[0].scatter(self.ecnk_i, self.gctnk_i / 2, color='y')
            ax[0].annotate('$N^{(ck)}$', xy=(self.ecnk.real, self.gctnk.real / 2), color='gray')
            ax[0].scatter(self.ectk, -self.gctnk / 2, color='gray')
            # ax[0].scatter(self.ectk_i, -self.gctnk_i / 2, color='y')

            me_contr = (self.ecnk + self.ern + self.ectk)/2
            re_contr = sqrt((self.gctnk+self.grtn)**2/4+(self.ectk-me_contr)**2)
            circle1_contr = plt.Circle((me_contr, 0), re_contr, color='k',linestyle='--', fill=False)
            ax[0].add_patch(circle1_contr)

            rc = 1 / 2 * sqrt((self.ecn - self.ectt) ** 2 + self.gctn ** 2)
            mc = (self.ecn + self.ectt) * 1 / 2
            circle2 = plt.Circle((mc, 0), rc, color='k', fill=False)
            ax[0].add_patch(circle2)
            ax[0].scatter(self.ecn, self.gctn / 2, color='r')
            ax[0].annotate('$N^{(cr)}$', xy=(self.ecn.real, self.gctn.real / 2), color='r')
            ax[0].scatter(self.ectt, -self.gctn / 2, color='b')
            ax[0].annotate('$T^{(cr)}$', xy=(self.ectt.real, -self.gctn.real / 2), color='b')
            ax[0].scatter(self.ec1, 0, color='k')
            ax[0].scatter(self.ec3, 0, color='k')
            ax[0].scatter(self.ecx, self.gcxy / 2, color='k')
            ax[0].scatter(self.ecx, -self.gcxy / 2, color='k')
            ax[0].annotate('$Q^{(cr)}$', xy=(self.ecx.real, -self.gcxy.real / 2), color='k')
            ax[0].annotate('$X^{(cr)}$', xy=(self.ecx.real, self.gcxy.real / 2))
            ax[0].plot([self.ecx, self.ecx + 2 * rc * cos(self.thr)],
                       [-self.gcxy / 2, -self.gcxy / 2 + 2 * rc * sin(self.thr)], color='k', linewidth=0.5, linestyle='dashdot')
            ax[0].plot([self.ecx, self.ecx], [self.gcxy / 2, -self.gcxy / 2], color='k', linewidth=0.5)
            ax[0].plot([self.ecx, self.ecx + 2 * rc * cos(self.thc)],
                       [-self.gcxy / 2, -self.gcxy / 2 + 2 * rc * sin(self.thc)], color='k', linewidth=0.5)

            rr = 1 / 2 * sqrt((self.ern - 0) ** 2 + self.grtn ** 2)
            mr = (self.ern + 0) * 1 / 2
            circle3 = plt.Circle((mr, 0), rr, color='m', fill=False)
            ax[0].add_patch(circle3)
            ax[0].scatter(self.ern, self.grtn / 2, color='m')
            ax[0].annotate('$N^{(r)}$', xy=(self.ern.real, self.grtn.real / 2), color='m')
            ax[0].scatter(0, -self.grtn / 2, color='b')
            ax[0].annotate('$T^{(r)}$', xy=(0, -self.grtn.real / 2), color='b')
            ax[0].grid(True)

            ax[0].set_title('Mohrs Circle of Strains')
            ax[0].set(xlabel="$\epsilon$ [-]")
            ax[0].set(ylabel="$\gamma/2$ [-]")

            circle4 = plt.Circle((self.sc3 / 2 + self.sc1 / 2, 0), (self.sc1 - self.sc3) / 2, color='k', fill=False)
            circle5 = plt.Circle((self.scxk / 2 + self.scyk / 2, 0),
                                 1 / 2 * sqrt((self.scxk - self.scyk) ** 2 + 4 * self.tcxyk ** 2), color='gray',
                                 fill=False)
            ax[1].add_patch(circle4)
            ax[1].add_patch(circle5)
            ax[1].scatter(self.scnr, self.tctnr, color='r')
            ax[1].scatter(self.sctr, -self.tctnr, color='r')
            ax[1].scatter(self.scx,self.tcxy,color='k')
            ax[1].plot([self.scnr,self.scnr+self.dsnrk],[self.tctnr,self.tctnr],color='b')
            ax[1].plot([self.scnr + self.dsnrk, self.scnr + self.dsnrk], [self.tctnr, self.tctnr+self.dttnrk], color='b')
            ax[1].plot([self.sctr, self.sctr], [-self.tctnr, -self.tctnr - self.dttnrk], color='b')
            ax[1].scatter(self.scy, -self.tcxy, color='k')
            ax[1].annotate('$N^{(cr)}$', xy=(self.scnr.real, self.tctnr.real), color='r')
            ax[1].annotate('$T^{(cr)}$', xy=(self.sctr.real, -self.tctnr.real), color='r')
            ax[1].scatter(self.scnk, self.tctnk, color='gray')
            ax[1].scatter(self.sctk, -self.tctnk, color='gray')
            ax[1].annotate('$N^{(ck)}$', xy=(self.scnk.real, self.tctnk.real), color='gray')
            ax[1].annotate('$T^{(ck)}$', xy=(self.sctk.real, -self.tctnk.real), color='gray')
            ax[1].scatter(self.sc(self.ec1), 0, color='c',s=75,alpha = 0.5)
            ax[1].scatter(self.sc(self.ec3), 0, color='c',s=75,alpha = 0.5)
            ax[1].scatter(self.sc(self.ec1k), 0, color='c',s=75,alpha = 0.5)
            ax[1].scatter(self.sc(self.ec3k), 0, color='c',s=75,alpha = 0.5)
            ax[1].scatter(self.sc1k, 0, color='gray')
            ax[1].scatter(self.sc3k, 0, color='gray')
            ax[1].scatter(self.sc1, 0, color='k')
            ax[1].scatter(self.sc3, 0, color='k')
            rr = self.sc1-self.sc3
            ax[1].plot([self.scx,self.scx+rr*cos(self.thr)],[-self.tcxy,-self.tcxy+rr*sin(self.thr)],color='k', linewidth=0.5, linestyle='dashdot')
            ax[1].plot([self.scx, self.scx + rr * cos(self.thc)], [-self.tcxy, -self.tcxy + rr * sin(self.thc)],
                       color='k', linewidth=0.5)
            ax[1].plot([self.scxk, self.scxk + rr * cos(self.thr)], [-self.tcxyk, -self.tcxyk + rr * sin(self.thr)],
                       color='gray', linewidth=0.5, linestyle='dashdot')
            ax[1].plot([self.scxk, self.scxk + rr * cos(self.thck)], [-self.tcxyk, -self.tcxyk + rr * sin(self.thck)],
                       color='gray', linewidth=0.5)
            ax[1].set_title('Mohrs Circle of Concrete Stresses')
            ax[1].set(xlabel="$\sigma$ [MPa]")
            ax[1].set(ylabel="$\\tau$ [MPa]")
            ax[1].grid(True)
            plt.show()

        def mohr_x(ux,uy,uxy,isstress = True):
            if isstress:
                # Mohr's Circle of stress
                f = 1
            else:
                # Mohr's Circle of Strain
                f = 2

            r = 1 / 2 * sqrt((ux - uy) ** 2 + (2*uxy/f) ** 2)
            m = (ux + uy) * 1 / 2

            u1 = m+r
            u3 = m-r

            phi = pi/2-self.thr
            un = m + 0.5 * (ux - uy) * cos(2 * phi) + uxy / f * sin(2 * phi)
            ut = m - 0.5 * (ux - uy) * cos(2 * phi) - uxy / f * sin(2 * phi)
            utn = f * (-0.5 * (ux - uy) * sin(2 * phi) + uxy / f * cos(2 * phi))

            return un,ut,utn,u1,u3

        def mohr_n(un,ut,unt,isstress = True):
            if isstress:
                # Mohr's Circle of stress
                f = 1
            else:
                # Mohr's Circle of Strain
                f = 2

            r = 1 / 2 * sqrt((un - ut) ** 2 + (2*unt/f) ** 2)
            m = (un + ut) * 1 / 2

            u1 = m+r
            u3 = m-r

            phi = -(pi/2-self.thr)
            ux = m + 0.5 * (un - ut) * cos(2 * phi) + unt / f * sin(2 * phi)
            uy = m - 0.5 * (un - ut) * cos(2 * phi) - unt / f * sin(2 * phi)
            uxy = f * (-0.5 * (un - ut) * sin(2 * phi) + unt / f * cos(2 * phi))

            return ux,uy,uxy,u1,u3

        def mohr_1(u1,u3,th,isstress = True):
            if isstress:
                # Mohr's Circle of stress
                f = 1
            else:
                # Mohr's Circle of Strain
                f = 2

            ux = u3 * cos(th) ** 2 + u1 * sin(th) ** 2
            uy = u1 * cos(th) ** 2 + u3 * sin(th) ** 2
            uxy = f * (u1 - u3) * cos(th) * sin(th)

            return ux,uy,uxy

        def tct_scnr():

            Dmax = self.Dmax
            if self.dn < 0.00001:
                self.dn = self.dn / abs(self.dn) * 0.00001

            # Yang: Reduction factor for high strength concrete
            if self.fcp >= 65:
                R_ai = 0.85 * sqrt((7.2 / (self.fcp - 40) + 1) ** 2 - 1) + 0.34
                if R_ai > 1:
                    R_ai = 1
            else:
                R_ai = 1
            # 1 Interlock stresses according to Gambarova and Karakoc
            if self.cmtn_klij == 1:
                t0 = 0.275 * self.fcp
                if abs(self.dt) > 10 ** -10:
                    self.tctnr = R_ai * t0 * (1 - sqrt(2 * self.dn / Dmax)) * self.dt / self.dn * (
                                2.45 / t0 + 2.44 * (1 - 4 / t0) * abs(self.dt / self.dn) ** 3) / (
                                             1 + 2.44 * (1 - 4 / t0) * (self.dt / self.dn) ** 4)
                    self.scnr = -0.62 * self.dt / (self.dn ** 2 + self.dt ** 2) ** (1 / 4) * self.tctnr
                else:
                    self.tctnr = 0
                    self.scnr = 0

            # 2 Interlock stresses according to Li and Maekawa
            elif self.cmtn_klij == 2:
                Kd = 1-euler**(1-Dmax/(2*self.dn))
                Kd = max(0.001,Kd)
                tlim = R_ai *Kd*3.83*self.fcp**(1/3)
                if abs(self.dt) > 10**-10:
                    self.tctnr = self.dt / abs(self.dt) * tlim * self.dt ** 2 / (self.dn ** 2 + self.dt ** 2)
                    self.scnr = -tlim*(pi/2-atan(abs(self.dn/self.dt))-abs(self.dn*self.dt/(self.dn**2+self.dt**2)))
                else:
                    self.tctnr = 0
                    self.scnr = 0

            # 3 Interlock stresses according to Walraven
            elif self.cmtn_klij == 3:
                from scipy.integrate import quad

                def TPM_ContactArea(w, v, Dmax, p_k):

                    if w < 10**-10:
                        w = cplx(10**-10,0)
                    if v < 10**-10:
                        v = cplx(10**-10,0)

                    def u_max(D, v, w):
                        u_max = (-1 / 2 * w * (w ** 2 + v ** 2) + 1 / 2 * sqrt(
                            w ** 2 * (w ** 2 + v ** 2) ** 2 - (w ** 2 + v ** 2) * (
                                        (w ** 2 + v ** 2) ** 2 - v ** 2 * D ** 2))) / (
                                        w ** 2 + v ** 2)
                        return u_max

                    def F(D, Dmax, v, w):
                        F = 0.532 * (D / Dmax) ** 0.5 - 0.212 * (D / Dmax) ** 4 - 0.072 * (D / Dmax) ** 6 - 0.036 * (
                                    D / Dmax) ** 8 - 0.025 * (D / Dmax) ** 10
                        return F

                    def G1(D, Dmax, v, w):
                        G1 = D ** -3 * (sqrt(D ** 2 - (w ** 2 + v ** 2)) * v / (sqrt(w ** 2 + v ** 2)) * u_max(D, v, w) - w * u_max(
                            D, v, w) - u_max(D, v, w) ** 2)
                        return G1

                    def G2(D, Dmax, v, w):
                        G2 = D ** -3 * ((v - sqrt(D ** 2 - (w ** 2 + v ** 2)) * w / sqrt(w ** 2 + v ** 2)) * u_max(D, v, w) + (
                                                    u_max(D, v, w) + w) * sqrt(
                            1 / 4 * D ** 2 - (w + u_max(D, v, w)) ** 2) -
                                        w * sqrt(1 / 4 * D ** 2 - w ** 2) + 1 / 4 * D ** 2 * asin(
                                    (w + u_max(D, v, w)) / (1 / 2 * D)) - D ** 2 / 4 * asin(2 * w / D))
                        return G2

                    def G3(D, Dmax, v, w):
                        G3 = D ** -3 * (1 / 2 * D - w) ** 2
                        return G3

                    def G4(D, Dmax, v, w):
                        G4 = D ** -3 * (np.pi / 8 * D ** 2 - w * sqrt(1 / 4 * D ** 2 - w ** 2) - D ** 2 / 4 * asin(
                            2 * w / D))
                        return G4

                    def integrate_cplx(function, Dmax, v, w, a, b):
                        steps = 10
                        x = np.linspace(a.real, b.real, steps)
                        integral = 0
                        for i in range(steps - 1):
                            x0 = x[i]
                            x1 = x[i + 1]
                            y = function((x0 + x1) / 2, Dmax, v, w)
                            integral += y*(x1-x0)
                        return integral

                    if v < w:
                        if (w ** 2 + v ** 2) / v < Dmax:
                            def ftemp1(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G1(D, Dmax, v, w)

                            a = (w ** 2 + v ** 2) / v
                            b = Dmax
                            A_y_bar = integrate_cplx(ftemp1, Dmax, v, w, a, b)

                            def ftemp2(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G2(D, Dmax, v, w)

                            a = (w ** 2 + v ** 2) / v
                            b = Dmax
                            A_x_bar = integrate_cplx(ftemp2, Dmax, v, w, a, b)
                        else:

                            A_x_bar = 0
                            A_y_bar = 0
                    else:
                        if (w ** 2 + v ** 2) / w < Dmax:
                            def ftemp31(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G1(D, Dmax, v, w)

                            def ftemp32(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G3(D, Dmax, v, w)

                            a31 = (w ** 2 + v ** 2) / w
                            b31 = Dmax
                            a32 = 2 * w
                            b32 = (w ** 2 + v ** 2) / w
                            A_y_bar = integrate_cplx(ftemp31, Dmax, v, w, a31, b31) + integrate_cplx(ftemp32, Dmax, v,
                                                                                                     w, a32, b32)

                            def ftemp41(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G2(D, Dmax, v, w)

                            def ftemp42(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G4(D, Dmax, v, w)

                            a41 = (w ** 2 + v ** 2) / w
                            b41 = Dmax
                            a42 = 2 * w
                            b42 = (w ** 2 + v ** 2) / w
                            A_x_bar = integrate_cplx(ftemp41, Dmax, v, w, a41, b41) + integrate_cplx(ftemp42, Dmax, v,
                                                                                                     w, a42, b42)

                        elif 2 * w < Dmax:
                            def ftemp5(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G3(D, Dmax, v, w)

                            a = 2 * w
                            b = Dmax
                            A_y_bar = integrate_cplx(ftemp5, Dmax, v, w, a, b)

                            def ftemp6(D, Dmax, v, w):
                                return p_k * 4 / np.pi * F(D, Dmax, v, w) * G4(D, Dmax, v, w)

                            a = 2 * w
                            b = Dmax
                            A_x_bar = integrate_cplx(ftemp6, Dmax, v, w, a, b)

                        else:
                            A_x_bar = 0
                            A_y_bar = 0

                    return [A_y_bar, A_x_bar]

                def TPM_Stresses(dn, dt):

                    fc = self.fcp
                    fcc = fc/0.85

                    sigma_pu = 6.39 * fcc ** 0.56
                    p_k = 0.75
                    my = 0.4

                    sign_dt = dt/abs(dt)
                    sign_dt = sign_dt.real


                    [Ay_bar, Ax_bar] = TPM_ContactArea(dn, sqrt(dt*dt), Dmax, p_k)
                    sigma = -sigma_pu * (Ax_bar - my * Ay_bar)
                    tau = sign_dt * sigma_pu * (Ay_bar + my * Ax_bar)

                    sigma = R_ai * sigma
                    tau = R_ai * tau

                    if sigma > 0:
                        sigma = 0

                    return [sigma, tau]

                w = cplx(self.dn)
                v = cplx(self.dt)

                [self.scnr, self.tctnr] = TPM_Stresses(w,v)

            if self.tctnr*self.dt < 0:
                self.tctnr = 0
            if self.scnr > 0:
                self.scnr = 0
            # 4 Linear activation of interlock stresses
            # if self.s1prev < self.fct * 2:
            #     acti = (-self.fct+self.s1prev)/(self.fct)
            #     self.scnr = self.scnr * acti
            #     self.tctnr = self.tctnr * acti
            #     print(acti)
            # if self.e3 > -ff_e3 * 2:
            #     acti = (-ff_e3+abs(self.e3))/ff_e3
            #     self.scnr = self.scnr * acti
            #     self.tctnr = self.tctnr * acti

            # 3 Custom cubic formula
            # tlim = 7*max(0.1,self.dn)/0.2
            # slim = -4*max(0.1,self.dn)/0.2
            # dtlim = 1
            # at = -2*tlim/dtlim**3
            # bt = 3*tlim/dtlim**2
            # ass = -2*slim/dtlim**3
            # bss = 3*slim/dtlim**2
            # self.scnr = ass*abs(self.dt)**3 + bss*abs(self.dt)**2
            # self.tctnr = self.dt/abs(self.dt)*(at*abs(self.dt)**3 + bt*abs(self.dt)**2)

        def tct_dowel():

            # 4 Dowel Action according to Vintzileou in combination with bending problem
            # 4.1 Assumption for spacings
            sy = self.spacx
            sz = self.dx ** 2 * pi / 4 / (self.rhox * sy)
            rsz = sz/(2*30+self.dx)
            if rsz > 1:
                sz = sz/rsz
                sy = sy*rsz

            # 4.2 Failure mode I: Concrete crushing beneath bar
            fsyred = self.fsyx - (self.ex * self.Esx + self.tb0 * self.sr / self.dx)
            if fsyred < 0:
                fsyred = 0
            Du1 = 1.3 * self.dx ** 2 * sqrt(self.fcp * fsyred)
            du1 = 2 * self.ec0 * self.dx

            # 4.3 Failure mode II: Concrete splitting
            Du2 = 2 * (sz - self.dx) * self.dx * self.fct * 0.5
            du2 = 2 * Du2 / self.Ec

            # 4.4 Resulting maximum stress
            if Du1 < Du2:
                tdow_max = Du1 / (sy * sz)
                dmax = du1
            else:
                tdow_max = Du2 / (sy * sz)
                print(tdow_max)
                dmax = du2

            # 4.5 Stress in dependence of crack opening in y-direction
            # 4.5.1 Activation factor for inclusion of dn in dowel action (numerical stability)
            if abs(self.dt) < 1:
                act_dt = abs(self.dt) / 1
            else:
                act_dt = 1

            # 4.5.2 Crack opening in y
            dy = abs(self.dt * cos(self.thr)) + abs(self.dn * sin(self.thr)) * act_dt

            # 4.5.3 Dowel action depending on maximum stress and crack displacement
            if dy < dmax:
                tdow = self.dt / (abs(self.dt)) * dy / dmax * tdow_max
            else:
                tdow = self.dt / abs(self.dt) * tdow_max
            self.tctndow = tdow * cos(self.thr)
            self.tctnr += self.tctndow
        # --------------------------------------------------------------------------------------------------------------
        "-"
        # 0 If testmethod: run vectors of dn and dt, return tau and sigma, do not run method
        if testmethod:
            steps = 100
            dn_all = np.linspace(0.3, 0.3, steps)
            dt_all = np.linspace(0.001, 3, steps)
            tau_all = np.zeros_like(dt_all)
            sigma_all = np.zeros_like(dt_all)
            for j in range(len(dt_all)):
                self.dn = dn_all[j]
                self.dt = dt_all[j]
                tct_scnr()
                tau_all[j] = self.tctnr.real
                sigma_all[j] = self.scnr.real
            return dn_all, dt_all, tau_all.real, sigma_all.real

        # Total applied strains in n and t
        # 1 Mohr's Circle of Applied Strains, given ex, ey, gxy
        self.en,self.et,self.gtn,self.e1,self.e3 = mohr_x(self.ex,self.ey,self.gxy,False)

        # 2 Stress increments introduced by bond to chracteristic points in quarter points of crack elements
        # 2.1 Concrete stress increments in x and y
        self.qxint = (self.ssx - self.ssxk) * self.rhox / (1 - self.rhox) + (self.spx - self.spxk) * self.rhopx / (
                1 - self.rhopx)
        self.qyint = (self.ssy - self.ssyk) * self.rhoy / (1 - self.rhoy) + (self.spy - self.spyk) * self.rhopy / (
                1 - self.rhopy)
        self.qxint = self.qxint*do_char
        self.qyint = self.qyint*do_char


        # 2.2 Mohr's Circle of characteristic stress increments
        self.dsnrk = self.qxint*sin(self.thr) + self.qyint*cos(self.thr)
        self.dstrk = self.v*self.dsnrk
        self.dttnrk = -self.qxint * cos(self.thr) + self.qyint * sin(self.thr)

        # 2.4 First estimates of strain increments
        self.denrk = self.dsnrk / self.Ec
        self.detrk = 0
        self.dgtnrk = self.dttnrk / self.Ec * 2

        # 3 Mohr's Circle of concrete strains at the crack
        if do_itkin:
            self.ecn = self.sprev_klij.ecn
            self.gctn = self.sprev_klij.gctn
        else:
            self.ecn = 0
            self.gctn = 0
        self.ectt = self.et - self.dstrk / self.Ec
        self.ecx, self.ecy, self.gcxy, self.ec1, self.ec3 = mohr_n(self.ecn, self.ectt, self.gctn, False)

        # 4 Mohr's Circle of characteristic concrete strains
        self.ecnk = self.ecn + self.denrk
        self.ectk = self.et + self.detrk
        self.gctnk = self.gctn + self.dgtnrk
        self.ecxk, self.ecyk, self.gcxyk, self.ec1k, self.ec3k = mohr_n(self.ecnk, self.ectk, self.gctnk, False)

        # 5 Mohr's Circle of strains due to crack kinematics
        # 5.1 ert is not updated in iteration, always ert = 0
        self.ert = 0

        # 5.2 First iteration is made in any case, if no iteration is required, it is aborted after the first step
        contit = True
        itcount = 0
        while contit:
            # 5.3 Update grtn and ern based on characteristic concrete strains
            self.grtn = self.gtn - self.gctnk
            self.ern = self.en - self.ecnk
            # if self.ern < 10 ** -10:
            #     print('fdsa')
            #     self.ern = 10 ** -10

            # 6 Crack kinematics
            self.dn = self.ern * self.sprev_klij.sr
            self.dt = self.grtn * self.sprev_klij.sr
            if abs(self.dn) < 0.01:
                self.dn = np.sign(self.dn) * 0.01

            # 7 Interlock stresses for given crack kinematics
            tct_scnr()

            # 8 Mohr's Circle of concrete stresses at the crack, accounting for the fact that Nc from interlock stresses
            #   must lie on the circle: adjust s_c1 accordingly
            # 8.1 Concrete stresses from aggregate interlock
            self.sc3 = self.sc(self.ec3)
            a = sqrt((self.scnr-self.sc3)**2+self.tctnr**2)
            if abs((self.scnr - self.sc3)) < 10 ** -10:
                self.scnr += 0.001
            alpha = atan(self.tctnr/(self.scnr-self.sc3))
            self.sc1 = self.sc3 + a/cos(alpha)

            self.sctr = self.sc3 + (self.sc1-self.scnr)
            self.scx, self.scy, self.tcxy, _, _ = mohr_n(self.scnr, self.sctr, self.tctnr, True)
            if abs((self.sc1 - self.scx)) < 10 ** -10:
                self.scx += 0.001
            self.thc = atan(self.tcxy/(self.sc1-self.scx))

            # 8.2 Dowel Action (depending on concrete stress at the crack due to aggregate interlock due to criterion
            # sc1 < fct
            tct_dowel()
            if self.cmd_klij > 0.5:
                # 8.3 Concrete Stresses due to aggregate interlock and dowel action
                a = sqrt((self.scnr-self.sc3)**2+self.tctnr**2)
                if abs((self.scnr-self.sc3)) < 10**-10:
                    self.scnr += 0.001
                alpha = atan(self.tctnr/(self.scnr-self.sc3))
                self.sc1 = self.sc3 + a/cos(alpha)

                self.sctr = self.sc3 + (self.sc1-self.scnr)
                self.scx, self.scy, self.tcxy, _, _ = mohr_n(self.scnr, self.sctr, self.tctnr, True)
                if abs((self.sc1-self.scx)) < 10**-10:
                    self.scx += 0.001
                self.thc = atan(self.tcxy/(self.sc1-self.scx))

            # 9 Mohr's Circle of characteristic concrete stresses
            self.scnk = self.scnr + self.dsnrk
            self.sctk = self.sctr + self.dstrk
            self.tctnk = self.tctnr + self.dttnrk
            self.scxk, self.scyk, self.tcxyk, self.sc1k, self.sc3k = mohr_n(self.scnk, self.sctk, self.tctnk, True)
            self.thck = atan(self.tcxyk / (self.sc1k - self.scxk))

            itcount += 1
            r=0.67
            if max(self.ssx,self.ssy)>500:
                r = r/4
            if do_itkin:

                # 10 Mohr's Circle of characteristic concrete strains taking into account that ectk = et
                self.ec1k = self.ec(self.sc1k)
                a = sqrt((self.ec1k - self.et) ** 2 + (self.gctnk/2) ** 2)
                alpha = atan(self.gctnk/2 / (self.ec1k - self.et))
                self.ec3k = self.ec1k - a / cos(alpha)
                self.ecxk,self.ecyk,self.gcxyk = mohr_1(self.ec1k,self.ec3k,self.thck,False)
                self.ecnk_i, self.ectk_i, self.gctnk_i, _, _ = mohr_x(self.ecxk, self.ecyk, self.gcxyk, False)

                # 11 Update characteristic concrete stresses based on obtained strains (with ectk = et)
                self.sc3k = self.sc(self.ec3k)
                self.scxk, self.scyk, self.tcxyk = mohr_1(self.sc1k, self.sc3k, self.thck, True)
                self.scnk, self.sctk, self.tctnk, _,_ = mohr_x(self.scxk, self.scyk, self.tcxyk, True)

                # 12 Mohr's Circle of Concrete stresses at the crack
                self.scnr = self.scnk - self.dsnrk
                self.sctr = self.sctk - self.dstrk
                self.scx, self.scy, self.tcxy, self.sc1, self.sc3 = mohr_n(self.scnr, self.sctr, self.tctnr, True)

                # 13 Mohr's Circle of Concrete strains at the crack
                self.ec1 = self.ec(self.sc1)
                self.ec3 = self.ec(self.sc3)
                self.ecx,self.ecy,self.gcxy = mohr_1(self.ec1,self.ec3,self.thc,False)
                self.ecn, self.ectt, self.gctn, _, _ = mohr_x(self.ecx, self.ecy, self.gcxy, False)

                # 14 If iteration required: check convergence. Otherwise: continue
                # 14.1 Check converence if required

                # Maximum deviation w.r.t. previous iteration
                maxd = np.max([abs(self.ecnk-self.ecnk_i),abs(self.et-self.ectk_i),abs(self.gctnk-self.gctnk_i)])

                # Update Characteristic concrete strains (numerical damping of 1-r)
                self.ecnk = (1-r)*self.ecnk + r*self.ecnk_i
                self.gctnk = (1-r)*self.gctnk + r*self.gctnk_i
                self.ectk = self.et

                if maxd < 10**-8:
                    # print('converged!!')
                    contit = False
                if itcount > 200:
                    contit = False

            # 14.2 Assign calculated characteristic concrete strains if no iteration is required and continue
            else:
                contit = False

        # 15 If crack opening negative --> uncracked linear elastic analysis
        if self.ern <= 0:
            self.submodel = 1

        # 16 Plot Mohr's circle if option is checked
        # if doplot and self.ex.imag > 0 and self.cm_klij == 4 and self.submodel == 2 and self.gxy > 0.0156:
        if doplot and self.cm_klij == 4:
            plot_mohr()

    def t_mat(self):
        """ ----------------------------------- Calculation of Transformation Matrices ---------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - thc:           Principal direction of concrete stresses
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - Tsigma:       Stress transformation matrix. [sc1,sc3,tc13]' = Tsigma@[scx,scy,tcxy]'
            - Tepsilon:     Strain transformation matrix. [e1,e3,g13]' = Tepsilon@[ex,ey,txy]'
        -------------------------------------------------------------------------------------------------------------"""
        th = self.thc
        self.Tsigma = np.array([[sin(th) ** 2, cos(th) ** 2, 2 * sin(th) * cos(th)],
                                [cos(th) ** 2, sin(th) ** 2, -2 * sin(th) * cos(th)],
                                [- sin(th) * cos(th), sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
        self.Tepsilon = np.array([[sin(th) ** 2, cos(th) ** 2, sin(th) * cos(th)],
                                  [cos(th) ** 2, sin(th) ** 2, -sin(th) * cos(th)],
                                  [-2 * sin(th) * cos(th), 2 * sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])

    def fcs(self):
        """ ---------------------- Calculation concrete strength with softening as a function of e1---------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - fcp:          Cylinder compressive strength of concrete
            - e1:           Principal tensile strain
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - fc(0):        Concrete strength considering no softening
            - fc(1):        Concrete strength considering softening according to CSFM (fib Eq.(7.3- 40) /SIA modified)
            - fc(2):        Concrete strength considering softening according to Kaufmann (1998)
            - fc(3):        Concrete strength considering softening according to Vecchio & Collins (1986)
        -------------------------------------------------------------------------------------------------------------"""
        if self.cmcs_klij == 0:
            fc = self.fcp
        if self.cmcs_klij == 1:
            # CSFM
            if self.e1 < 0:
                kc = 1
            else:
                elim1 = 0.2 * self.ec0 / 0.34
                elim2 = 0.22 / 35.75
                kc2 = 1 / (1.2 + 55 * elim2)
                if self.e1 <= elim1:
                    kc = 1 + (elim1 - self.e1) * 0.001
                elif self.e1 >= elim2:
                    kc = 1 / (1.2 + 55 * self.e1)
                else:
                    # x = 228906275 / (64 * (5281250 * self.ec0 ** 2 + 357))
                    # y = -13 * (377609375 * self.ec0 ** 2 + 160973) / (40 * (5281250 * self.ec0 ** 2 + 357))
                    # z = (8376062500 * self.ec0 ** 2 + 837097) / (2000 * (5281250 * self.ec0 ** 2 + 357))
                    # kc = x * self.e1 ** 2 + y * self.e1 + z
                    kc = 1-(1-kc2)*(self.e1-elim1)/(elim2-elim1)
            fc = kc * self.fcp
        elif self.cmcs_klij == 2:
            # Kaufmann
            fc = min((pow(self.fcp, 2 / 3) / (0.4 + 30 * max(self.e1,0))), self.fcp)
        elif self.cmcs_klij == 3:
            # Vecchio & Collins
            fc = min((self.fcp/(0.8+0.34*self.e1/self.ec0)),self.fcp)
        self.fc_soft = fc

    def sc(self,e):
        """ --------------------------------- Calculation concrete compressive stress-----------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - cmcc_klij:     Constitutive Model for Concrete in Compression
                             1 = Quadratic stress - strain relation according to
                                 fib Model Code with adaptions according to CSFM
                             2 = Quadratic stress - strain relation according to
                                 Kaufmann (1998)
                             3 = Linear elastic stress - strain relationship
                             4 = Quadratic stress - strain relationship according to Sargin
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - sc3:          Concrete compressive stress according to given model
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Methods for determining concrete stresses with different approaches
        # 1.1 fib
        def sc_fib(e):
            """ --------------------------------- Calculation concrete compressive stress-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - fcs:          Concrete strength considering softening according to fib model code and CSFM
                                (Ch. 5.1.8.1, p. 82) fib Model Code 2010)
                - ec0:          Concrete strain at fc
                - ect:          Assumed tensile strain at cracking
                - Ec:           Concrete E-Modulus
                - e:            Normal strain in regarded direction
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - sc3:          Concrete compressive stress according to fib model code
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (ff*e for numerical stability)
            ---------------------------------------------------------------------------------------------------------"""
            # print('concrete fib')
            self.fcs()
            fc = self.fc_soft
            if e < 0:
                ec0 = self.ec0 * fc / self.fcp
                # ec0 = self.ec0
                if abs(e) < ec0:
                    eta = 0 - e / ec0
                    k = self.Ec / (fc / ec0)
                    k = max(k, 1)  # k cannot be s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    maller than 1! 1 means linear elastic
                    sc3 = 0 - fc * (k * eta - eta ** 2) / (1 + (k - 2) * eta)
                else:
                    sc3 = 0 - fc + (e + ec0) / 1000
            elif e <= self.ect+tole*0:
                # print(e)
                sc3 = self.Ec * e
            else:
                sc3 = ff * e
            return sc3

        # 1.2 Kaufmann
        def sc_kfm(e):
            """ --------------------------------- Calculation concrete compressive stress-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - fcs:          Concrete strength considering softening according to Kaufmann (1998)
                - ec0:          Concrete strain at fc
                - ect:          Assumed tensile strain at cracking
                - Ec:           Concrete E-Modulus
                - e:            Normal strain in regarded direction
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - sc3:          Concrete compressive stress according to Kaufmann (1998)
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (10*e for numerical stability)
                                - alpha parameter: from e = alpha*ec0, a linear course with the inclination of the
                                  derivative
                                  at alpha*ec0 is assumed for numerical stability
            ---------------------------------------------------------------------------------------------------------"""
            self.fcs()
            fc = self.fc_soft
            alpha = 0.99
            # ec0 = self.ec0 * fc / self.fcp
            ec0 = self.ec0
            if e < 0:
                if abs(e) < alpha * ec0:
                    sc3 = fc * (e ** 2 + 2 * e * ec0) / (ec0 ** 2)
                else:
                    sc3 = fc * (alpha ** 2 - 2 * alpha) + fc * (2 * alpha - 2) / ec0 * (- e - alpha * ec0)
            elif e < self.ect:
                sc3 = self.Ec * e
            else:
                sc3 = ff * e
                # sc3 = self.Ec * e
            return sc3

        # 1.3 Sargin
        def sc_sargin(e):
            # Formula 3.16, p36 in Sargin, 1971
            self.fcs()
            fc = self.fc_soft
            A = self.Ec * self.ec0 / fc
            D = 0.3
            if e < 0:
                x = 0 - e / self.ec0
                sc3 = -fc * (A * x + (D - 1) * x ** 2) / (1 + (A - 2) * x + D * x ** 2)
                if sc3 > 0:
                    sc3 = 0
                if x > 1:
                    sc3 = -fc * (1 + (x - 1) / 100)
            elif e < self.ect:
                sc3 = self.Ec * e
            else:
                sc3 = ff * e
            return sc3

        # 1.4 Linear elastic
        def sc_linel(e):
            self.fcs()
            fc = self.fc_soft
            if e < 0:
                ec0 = fc/self.Ec
                if abs(e) < ec0:
                    sc3 = self.Ec * e
                else:
                    sc3 = - fc + (e + ec0) / 1000
                    # sc3 = 0
            elif e < self.ect:
                sc3 = self.Ec * e
            else:
                sc3 = ff * e
            return sc3

        # 2 Get stress depending on cmcc_klij
        if self.cmcc_klij == 1:
            sc3 = sc_fib(e)
        elif self.cmcc_klij == 2:
            sc3 = sc_kfm(e)
        elif self.cmcc_klij == 3:
            sc3 = sc_linel(e)
        elif self.cmcc_klij == 4:
            sc3 = sc_sargin(e)

        return sc3

    def ec(self,s):
        """ --------------------------------- Calculation concrete compressive strain-----------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - cmcc_klij:     Constitutive Model for Concrete in Compression
                             1 = Quadratic stress - strain relation according to
                                 fib Model Code with adaptions according to CSFM
                             2 = Quadratic stress - strain relation according to
                                 Kaufmann (1998)
                             3 = Linear elastic stress - strain relationship
                             4 = Quadratic stress - strain relationship according to Sargin
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ec3:          Concrete compressive strain according to given model
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Methods for determining concrete stresses with different approaches
        # 1.1 fib
        def ec_fib(s):
            """ --------------------------------- Calculation concrete compressive strain-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - fcs:          Concrete strength considering softening according to fib model code and CSFM
                                (Ch. 5.1.8.1, p. 82) fib Model Code 2010)
                - ec0:          Concrete strain at fc
                - ect:          Assumed tensile strain at cracking
                - Ec:           Concrete E-Modulus
                - s:            Normal stress in regarded direction
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - ec3:          Concrete compressive strain according to fib model code
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (ff*e for numerical stability)
            ---------------------------------------------------------------------------------------------------------"""
            self.fcs()
            fc = self.fc_soft
            if s < 0:
                ec0 = self.ec0 * fc / self.fcp
                # ec0 = self.ec0
                if abs(s) < abs(self.sc(-ec0)):
                    k = self.Ec / (fc / ec0)
                    k = max(k, 1)  # k cannot be smaller than 1! 1 means linear elastic
                    sn = s/fc
                    eta = 0.5*(k*sn-sqrt((k*sn+k-2*sn)**2+4*sn)+k-2*sn)
                    ec3 = -eta*ec0
                else:
                    ec3 = (s+fc)*1000-ec0
            elif s < self.ect*self.Ec:
                ec3 = s/self.Ec
            else:
                ec3 = s/ff
            return ec3

        # 1.2 Kaufmann
        def ec_kfm(s):
            """ --------------------------------- Calculation concrete compressive strain-------------------------------
                ----------------------------------------    INPUT (self.): ---------------------------------------------
                - fcs:          Concrete strength considering softening according to Kaufmann (1998)
                - ec0:          Concrete strain at fc
                - ect:          Assumed tensile strain at cracking
                - Ec:           Concrete E-Modulus
                - s:            Normal stress in regarded direction
                ----------------------------------------------- OUTPUT:-------------------------------------------------
                - ec3:          Concrete compressive strain according to Kaufmann (1998)
                                - e < 0:         sc3 = f(e) according to compression parabola
                                - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                                - e > ect:       sc3 = "0" (10*e for numerical stability)
                                - alpha parameter: from e = alpha*ec0, a linear course with the inclination of the
                                  derivative
                                  at alpha*ec0 is assumed for numerical stability
            ---------------------------------------------------------------------------------------------------------"""
            # self.fcs()
            fc = self.fc_soft
            alpha = 0.99
            # ec0 = self.ec0 * fc / self.fcp
            ec0 = self.ec0
            if s < 0:
                if abs(s) < abs(self.sc(-alpha*ec0)):
                    a=1
                    b=2*ec0
                    c=-s*ec0**2/fc
                    ec3 = (-b+sqrt(b**2-4*a*c))/(2*a)
                else:
                    ec3=-(ec0/(fc*(2*alpha-2))*(s-fc*(alpha**2-2*alpha))+alpha*ec0)
            elif s < self.ect*self.Ec:
                ec3 = s/self.Ec
            else:
                ec3 = s/ff
                # ec3 = s/self.Ec
            return ec3

        # 1.4 Linear elastic
        def ec_linel(s):
            self.fcs()
            fc = self.fc_soft
            if s < 0:
                if s < -self.ec0*self.Ec:
                    ec3 = (s+fc)*1000-self.ec0
                else:
                    ec3 = s/self.Ec
            elif s < self.ect*self.Ec:
                ec3 = s/self.Ec
            else:
                ec3 = s/ff
            return ec3

        # 2 Get strain depending on cmcc_klij
        if self.cmcc_klij == 1:
            ec3 = ec_fib(s)
        elif self.cmcc_klij == 2:
            ec3 = ec_kfm(s)
        elif self.cmcc_klij == 3:
            ec3 = ec_linel(s)
        elif self.cmcc_klij == 4:
            print('Error: Sargin strain from stress calculation not implemented: cannot be used together '
                  'with iteration in fixed crack model')

        # 3 Control
        s_control = self.sc(ec3)
        if abs(s_control-s) > 0.001*abs(s):
            print('attention: concrete strain calculation not correct')
            print(s)
            print(s_control)
            print(ec3)
        return ec3

    def ssr(self, e, ec, ecs, srm, rho, rhop, d, fsy, fsu, Es, Esh, tb0, tb1, Ec):
        """ -------------------------------------- Calculation of steel stress -----------------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - cms_klij:      Constitutive Model for Steel in tension
                             0 = Bilinear stress-strain relationship
                             1 = TCM (or POM) with bilinear bare relationship
                             2 = TCM with cold-worked bare relationship according to p. 11/89 Diss Alvarez
                             3 = TCM with yielding plateau according to p.89 ff. Diss Alvarez
                             4 = TCM with custom defined sigma-epsilon and tau-delta relationships
                                 (numerical tension stiffening)
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ssr:            Steel stress (at crack)
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Steel stresses at crack
        if self.cms_klij == 0 or e <= ec + 1e-15:
            ssr = self.ss_bilin(e, fsy, Es, Esh)
        elif self.cms_klij == 1:
            ssr = self.ssr_tcm_bilin(e+ecs, ec+ecs, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec)
        elif self.cms_klij == 2:
            # Flag
            # if d < 140:
            if d == 14:
                ssr = self.ssr_tcm_ro(e+ecs, ec+ecs, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec)
            else:
                ssr = self.ssr_tcm_bilin(e+ecs, ec+ecs, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec)
        elif self.cms_klij == 4:
            ssr = self.ssr_tcm_custom(e,srm,rho,d,tb0,tb1,Es)

        # 2 Characteristic steel stresses
        if ssr.real > fsy:
            tbi = tb1
        else:
            tbi = tb0
        dsk = 4*tbi/d*srm/4
        if ssr.real > dsk*3/2:
            ssk = ssr-dsk
        elif ssr.real > dsk/2:
            ssk = ssr-dsk*(ssr-dsk/2)/dsk
        else:
            ssk = ssr

        # 3 Catch yielding in compression
        if ssr < -fsy:
            ssr = self.ss_bilin(e, fsy, Es, Esh)
            ssk = ssr

        # 4 Return values
        return ssr,ssk

    def spr(self, e, ec, ecs, rhop, srm, dp, fpu, Ep, tb):
        """ -------------------------------------- Calculation of CFRP stress -----------------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            -
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - spr:            CFRP stress (at crack)
        -------------------------------------------------------------------------------------------------------------"""
        # Flag
        e += ecs
        ec += ecs
        if self.cms_klij == 0 or e <= ec + 1e-15:
            if e > 0:
                spr = e*Ep
            else:
                # spr = e*2000
                spr = e*Ep
            spk = spr
        elif self.cms_klij == 4:
            spr = self.ssr_tcm_custom(e, srm, self.rhopx, dp, self.tbp0, self.tbp1, Ep)

            spk = spr-4*tb/dp*srm/2
            if spk < Ep*ec:
                spk = Ep*ec
        else:
            # 1 Seelhofer
            x1, spr, _ = self.ssr_seelhofer(e, ec, rhop, srm, dp, fpu, Ep, Ep, tb, tb)
            # 2 TCM
            if x1 >= srm / 2:
                spr = e*Ep+tb*srm/dp
            if spr > fpu:
                # Flag
                spr = fpu*1 + (e-fpu/Ep)*ff

            spk = spr-4*tb/dp*srm/2
            if spk < Ep*ec:
                spk = Ep*ec


        # spr = self.ssr_tcm_bilin(e, ec, srm, rhop, d, fpu, fpu*2, Ep, 20000, tb, tb/2, self.Ec)
        # spk = spr-10

        return spr,spk

    def ss_bilin(self, e, fsy, Es, Esh):
        """ ------------------------ Calculation of bare reinforcing steel stress: bilinear-----------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - Es, Esh:      Reinforcing steel Young's Modulus and Hardening Modulus
            - fsy:          Reinforcing steel yield stress
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ss:           Steel stress
        -------------------------------------------------------------------------------------------------------------"""
        esy = fsy / Es
        if abs(e) <= esy:
            ss = e * Es
        elif e > 0:
            ss = fsy + Esh * (e - esy)
        elif e < 0:
            ss = -fsy + Esh * (e + esy)
        return ss

    def ssr_seelhofer(self, e, ec, rho, srm, d, fsy, Es, Esh, tb0, tb1):
        """ --------------------- Calculate Steel Stress with Seelhofer for non-stabilised crack -----------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - ec: axial concrete compressive strain from mohr's circle of concrete strains
                - srm: Crack spacing: srm = lambda * sr0
                - d: Reinforcement diameter
                - fsy,Es: Reinforcing steel/CFRP parameters (for CFRP: fsy = fpu, Es = Ep)
                - tb0,tb1: Bond stresses elastic and plastic (for CFRP: tb0 = tb1 = tbp
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel/CFRP stress at the crack according to Seelhofer
            ---------------------------------------------------------------------------------------------------------"""
        # Seelhofer original

        # # 1 Material and Geometric properties
        # n = Es / self.Ec
        # alpha = 1 + n * rho
        # esy = fsy / Es
        #
        # # 2 Elastic crack element
        # c1 = sqrt(n * n * rho * rho + Es * e / tb0 * d / srm) - n * rho
        # x1 = srm / 2 * c1
        # x1 = min(max(x1, 0), srm / 2)
        # ssr = x1 * 4 * tb0 / d * (1 + n * rho)
        #
        # # 3 Elastic - Plastic crack element
        # if ssr > fsy:
        #     c2 = sqrt(4 * alpha * Es / Esh * (
        #             srm * tb1 / (d * fsy) * (alpha * Es * e / fsy - n * rho) - tb1 / (4 * alpha * tb0)) + 1) - 1
        #     x2 = d * fsy * Esh / (4 * tb1 * alpha * Es) * c2
        #     x2 = min(max(x2, 0), srm / 2)
        #     ssr = fsy + x2 * 4 * tb1 / d
        #     x21 = (fsy - ssr * n * rho / (1 + n * rho)) * d / (4 * tb0)
        #     x1 = x2 + x21

        # Seelhofer adapted

        # 2 Elastic crack element
        a = 4*tb0/(Es*d)
        b = 4*tb0/(self.Ec*d)*rho/(1-rho)*srm
        c = srm*(ec-e)
        x1 = (-b+sqrt(b**2-4*a*c))/(2*a)
        ssr = (ec+4*tb0/(self.Ec*d)*rho/(1-rho)*x1+4*tb0/(Es*d)*x1)*Es

        # 3 Elastic - Plastic crack element
        if x1 < srm / 2 and ssr > fsy:
            # print('Seelhofer elastisch-plastisch')
            ec = ec+(fsy/Es-ec)*Es*rho/(1-rho)/self.Ec
            x1 = (fsy - ec * Es) * d / (4 * tb0)
            c = -e*srm+2*ec*(srm/2-x1)+(fsy/Es+ec)*x1
            b = 2*(fsy/Es-ec)
            a = 4*tb1/(d*Esh)
            x2 = (-b+sqrt(b**2-4*a*c))/(2*a)
            ssr = ec*Es+x1*4*tb0/d+x2*4*tb1/d

        return x1,ssr,tb0

    def ssr_tcm_bilin(self, e, ec, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec, flag = 0):
        """ ------------------------------------ Calculate Steel Stress with the TCM -----------------------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - srm: Crack spacing: srm = lambda * sr0
                - rho: Reinforcement content
                - d: Reinforcement diameter
                - fsy,fsu,Es,Esh: Reinforcing steel parameters
                - tb0,tb1: Bond stresses elastic and plastic
                - Ec: Concrete E-modulus
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel stress at the crack according to TCM
            ---------------------------------------------------------------------------------------------------------"""
        # 1 Seelhofer
        x1, ssr, _ = self.ssr_seelhofer(e, ec, rho, srm, d, fsy, Es, Esh, tb0, tb1)

        # 2. TCM
        if x1 >= srm / 2:

            # 2.1 Bare steel stress
            st_naked = self.ss_bilin(e, fsy, Es, Esh)

            # 2.2 Steel stress for fully elastic crack element
            s1 = st_naked + tb0 * srm / d

            # 2.3 Steel stress for fully plastic element
            s3 = fsy + Esh * (e - fsy / Es) + tb1 * srm / d

            # 2.4 Assign according to stress level
            # 2.4.1 Fully elastic
            if s1 <= fsy:
                ssr = s1

            # 2.4.2 Fully plastic
            elif s1 > fsy and s3 - (2 * tb1 * srm / d) >= fsy:
                ssr = s3

            # 2.4.3 Partially elastic
            else:
                s2 = (fsy - Es * e) * tb1 * srm / d * (tb0 / tb1 - Es / Esh)
                s2 = s2 + Es / Esh * tb0 * tb1 * srm ** 2 / d ** 2
                s2 = tb0 * srm / d - sqrt(s2)
                s2 = fsy + 2 * s2 / (tb0 / tb1 - Es / Esh)
                ssr = s2

        # 2.5 If stress > ultimate stress, assign ultimate stress
        if ssr > fsu:
            # print('steel failure?')
            ssr = fsu + 100*e
            return ssr
        
        return ssr

    def ssr_tcm_ro(self, e, ec, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec):
        """ ------------------- Calculate Steel Stress with the TCM with Ramberg-Osgood/Hill Law------------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - srm: Crack spacing: srm = lambda * sr0
                - rho: Reinforcement content
                - d: Reinforcement diameter
                - fsy,fsu,Es,Esh: Reinforcing steel parameters
                - tb0,tb1: Bond stresses elastic and plastic
                - Ec: Concrete E-modulus
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel stress at the crack according to TCM
            ---------------------------------------------------------------------------------------------------------"""
        def esr_tcm_ro(ssr, srm, d, fsy, Es, tb0, tb1):
            # 0.1 Tension chord tests
            # if d == 8:
            #     fs1 = 545
            # elif d == 10:
            #     fs1 = 519
            # elif d == 12:
            #     fs1 = 489
            # elif d == 10.2:
            #     fs1 = 485

            # Flag
            # 0.2 Slab strip tests
            # if d == 10:
            #     fs1 = 483.67
            #     fsy = self.fsyx
            # elif d == 12:
            #     fs1 = 480.37
            #     fsy = self.fsyx

            # 0.3 Prototype
            # Flag
            # fs1 = 467

            # 0.4 Single girders
            if d == 10:
                fs1 = fsy-60
            elif d == 14:
                fs1 = fsy-150
            else:
                fs1 = fsy-100

            esy2 = 0.002
            esy1 = 0.0001
            fs2 = fsy
            a = log(esy2 / esy1) / (log(fs2 / fs1))
            fsy = fs1

            if ssr <= 2*tb0*srm/d:
                eres = ssr / Es - tb0 * srm / (Es * d)
            elif ssr <= fsy:
                eres = ssr / Es - tb0 * srm / (Es * d) + d / (2 * tb0 * srm) * esy2 / ((a + 1) * fs2 ** a) * (
                            ssr ** (a + 1) - (ssr - 2 * tb0 * srm / d) ** (a + 1))
            elif ssr <= fsy + 2 * tb1 * srm / d:
                eres = d / (4 * Es * tb1 * srm) * (
                            (ssr - fsy) ** 2 * (1 - tb0 / tb1) + 2 * Es * esy2 / ((a + 1) * fs2 ** a) * (
                                ssr ** (a + 1) -
                                fsy ** (a + 1) * (1 - tb1 / tb0) - tb1 / tb0 * (
                                            fsy + tb0 / tb1 * (ssr - fsy) - 2 * tb0 * srm / d) ** (a + 1))) + \
                       tb0 / tb1 * (ssr / Es - fsy / Es * (1 - tb1 / tb0)) - tb0 * srm / (Es * d)
            else:
                eres = ssr / Es - tb1 * srm / (Es * d) + d / (2 * tb1 * srm) * esy2 / ((a + 1) * fs2 ** a) * (
                            ssr ** (a + 1) - (ssr - 2 * tb1 * srm / d) ** (a + 1))

            # emin = ssr / Es - tb0 * srm / (Es * d)
            # if eres < emin:
            #     eres = emin

            return eres

        # 0 Catch Case of fsu
        esr_max = esr_tcm_ro(fsu, srm, d, fsy, Es, tb0, tb1)
        if e >= esr_max:
            return fsu + 100*e

        # 1 Seelhofer bilin
        x1, ssr, tb0 = self.ssr_seelhofer(e, ec, rho, srm, d, fsy, Es, Esh, tb0, tb1)

        # 2 Seelhofer RO
        if x1 < srm / 2:
            ec = ec+4*tb0/d*rho/(1-rho)*x1/Ec
            kit = 2
            count = 1
            while abs(kit - 1) > 0.000001:
                eres = esr_tcm_ro(ssr, 2*x1, d, fsy, Es, tb0, tb1)
                eres2 = esr_tcm_ro(ssr-2, 2*x1, d, fsy, Es, tb0, tb1)

                eres = (eres * x1 + ec * (srm / 2 - x1)) / (srm/2)
                eres2 = (eres2 * x1 + ec * (srm / 2 - x1)) / (srm / 2)
                kit = e / eres

                Etemp = 2/(eres-eres2)
                ssr = ssr + (e - eres) * Etemp
                count += 1
                if count > 1000:
                    print('Problem with Ramberg-Osgood Seelhofer iteration')
                    print(d)
                    print(ssr)
                    print(e)
                    print(ec)
                    print(srm)
                    print(self.t)
                    kit = 1
                    _, ssr, _ = self.ssr_seelhofer(e, ec, rho, srm, d, fsy, Es, Esh, tb0, tb1)
        # 3. TCM
        else:
            # 2 Find stress pertaining to strain according to Formulas 104-106 Diss Alvarez
            kit = 2
            ssr = self.ssr_tcm_bilin(e, ec, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec)
            count = 1
            while abs(kit-1) > 0.001:
                eres = esr_tcm_ro(ssr, srm, d, fsy, Es, tb0, tb1)
                eres2 = esr_tcm_ro(ssr-2, srm, d, fsy, Es, tb0, tb1)
                kit = e/eres

                # if e < fsy/Es:
                #     ssr = ssr + (e-eres)*Es/4
                # else:
                #     ssr = ssr + (e - eres) * Esh*2

                Etemp = 2 / (eres - eres2)
                ssr = ssr + (e-eres)*Etemp

                count +=1
                if count > 1000:
                    print('Problem with Ramberg-Osgood iteration')
                    print(kit)
                    print(d)
                    print(ssr)
                    print(srm)
                    print(e)
                    print(ec)
                    print(self.submodel)
                    print(self.t)
                    print(self.thr)
                    print(self.thc)
                    print(self.e1)
                    print(self.e3)
                    print(self.ec1)
                    print(self.ec3)
                    print(Etemp)
                    kit=1
        # if ssr > fsu:
            # print('steel failure?')
            # print(ssr)
            # print(e)
            # ssr = fsu-10+1000*e
        # if ssr <= fsy + 2 * tb1 * srm / d and ssr > fsy:
        #     print(e)
        # if x1 < srm/2:
        #     print(e)
        # print('RO converged')
        # print(eres,e)
        # print(ssr)
        return ssr

    def ssr_tcm_custom(self, e, srm, rho, d, tb0, tb1, Es):
        """ ------------------------------------ Calculate Steel Stress with the TCM -----------------------------------
                ------------------------------------------ INPUT (self.): ----------------------------------------------
                - e: normal strain
                - srm: Crack spacing: srm = lambda * sr0
                - rho: Reinforcement content
                - d: Reinforcement diameter
                - tb0,tb1: Bond stresses elastic and plastic
                - Es: Reinforcement (steel or CFRP) Young's Modulus
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel stress at the crack according to TCM
                ------------------------------------- DEFINED IN CLASS: ------------------------------------------------
                - es_bare, ss_bare: custom bare steel law
                - delta, tau: bond stress - slip relationship. If not defined: tb0 and tb1 according to input
            ---------------------------------------------------------------------------------------------------------"""
        # 0 Functions
        def gets(e,es_bare,ss_bare):
            """ Get stress for given strain in costum material law """
            for i in range(len(es_bare)):
                if e <= es_bare[i]:
                    s = ss_bare[i-1] + (e-es_bare[i-1])/(es_bare[i]-es_bare[i-1])*(ss_bare[i]-ss_bare[i-1])
                    return s

        def gete(s,es_bare,ss_bare):
            """ Get strain for given stress in costum material law """
            for i in range(len(ss_bare)):
                if s <= ss_bare[i]:
                    e = es_bare[i-1] + (s-ss_bare[i-1])/(ss_bare[i]-ss_bare[i-1])*(es_bare[i]-es_bare[i-1])
                    return e

        def gett(s,d,fsy,tb0,tb1,ds_bare,ts_bare):
            """ Get bond stress for given slip or normal stress in costum law """
            # law = 1 for assumptions of TCM
            # law = 2 for custom delta-tau relationship
            law = 1

            if law == 1:
                if s < fsy * 9 / 10:
                    tbi = tb0
                elif s > fsy * 11 / 10:
                    tbi = tb1
                else:
                    tbi = tb0 - (tb0 - tb1) * (s - fsy * 9 / 10) / (2 / 10 * fsy)

                return tbi
            elif law == 2:
                for i in range(len(ds_bare)):
                    if d <= ds_bare[i]:
                        tbi = ts_bare[i - 1] + (d - ds_bare[i - 1]) / (ds_bare[i] - ds_bare[i - 1]) * (
                                    ts_bare[i] - ts_bare[i - 1])
                        # print(tbi)
                        return tbi

        # 1 Bare steel stress - strain relationship
        # 1.1 Input
        if Es == self.Esy:
            # d8
            # es_input = [-0.0022,0,0.001,0.002,0.00227,0.003,0.004,0.0046,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,
            #             0.017,0.018,0.019,0.02,0.0291,0.0391]
            # ss_input = [-399,0,196.99,381.01,420,483.37,519.87,529.78,536.38,545.86,552.25,557.49,562.98,568.29,572.86,574.60,576.92,578.22,579.78,581.10,
            #             582.44,584.01,585.16,586.82,593.34,594.00]

            # es_input = [-0.00278238,0,0.00278238,0.062]
            # ss_input = [-537,0,537,603]

            # esy2 = 0.002
            # esy1 = 0.0001
            # fs2 = 537+50
            # fs1 = 400+50
            # a = log(esy2 / esy1) / (log(fs2 / fs1))
            # ss_input = [*range(0,650,10)]
            # es_input = [i/Es+0.002*(i/fs2)**a for i in ss_input]

            # Berücksichtigung prestrain: de = ((de_S1+de_S2)/2)
            # de = 0.3e-3
            # es_input = [i+de for i in es_input]
            # ss_input = [i+de*self.Esx for i in ss_input]
            # indyield = 4

            es_input = [-300*0.002605, 0.002605, 0.05, 0.1,1]
            ss_input = [-300*521, 521,521.1,521*1.2,521*1.201]
            # es_input = [-0.1,0.1]
            # ss_input = [-20000,20000]
            indyield = 1
        elif Es == self.Epx:
            es_input = [-0.03,0,0.03]
            ss_input = [-4470,0,4470]
            indyield = 2

        # 1.2 Define Yield stress
        fsy = ss_input[indyield].real

        # 1.3 Manipulate es_bare and ss_bare to get derivatives. No action required
        es_bare = [cplx(es_input[0], 1e-17)]
        ss_bare = [cplx(ss_input[0], (ss_input[1] - ss_input[0]) / (es_input[1] - es_input[0]))]
        for i in range(len(es_input)):
            if i == 0:
                es_bare[i] = cplx(es_input[i], 1e-17)
                ss_bare[i] = cplx(ss_input[i],(ss_input[i+1]-ss_input[i])/(es_input[i+1]-es_input[i])*1e-17)
            elif i < len(es_input)-1:
                es_bare.append(cplx(es_input[i], 1e-17))
                es_bare.append(cplx(es_input[i], 1e-17))
                ss_bare.append(cplx(ss_input[i], (ss_input[i] - ss_input[i-1]) / (es_input[i] - es_input[i-1])*1e-17))
                ss_bare.append(cplx(ss_input[i], (ss_input[i+1] - ss_input[i]) / (es_input[i+1] - es_input[i])*1e-17))
            else:
                es_bare.append(cplx(es_input[i], 1e-17))
                ss_bare.append(cplx(ss_input[i], (ss_input[i] - ss_input[i - 1]) / (es_input[i] - es_input[i - 1])*1e-17))

        # 2 Local bond stress - slip relationship
        # 2.1 Input
        if Es == self.Esy:
            # d8 (TC1) over entire test
            ds_bare = [0, 0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,1]
            ts_bare = [0,6.39,6.30,4.79,3.66,3.01,2.07,1.74,1.21,0.77,0.77]

            # d8 (TC1) for evaluated crack elements with e_mean = 1.5
            # ds_bare = [0, 0.005,0.015,0.03,0.045,0.06,0.075,0.9,0.105,0.12,0.135,1]
            # ts_bare = [0,3.24,7.20,10.17,7.78,6.33,5.22,2.65,1.30,1.50,0.88,0.70]
        elif Es == self.Epx:
            # CFK TC1 over entire test
            ds_bare = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 1]
            ts_bare = [0,6.32, 6.75, 5.41, 3.79, 3.28, 2.96, 3.00, 1.87, 1.87, 1.87]

            # CFK TC1 for evaluated crack elements with e_mean = 1.5
            # ds_bare = [0, 0.005,0.015,0.03,0.045,0.06,0.075,0.9,0.105,0.12,0.135,1]
            # ts_bare = [0,3.79,7.08,8.02,7.75,7.91,4.85,1.48,1.38,1.28,1.18,1.08]
        # 3 Iterate through half of crack element, determine strains and stresses

        # 3.1 Number of steps to iterate through sr/2
        numsteps = 5
        delta_x = srm / 2 / (numsteps-1)

        # 3.2 Starting point for ssr and slip
        ssr = gets(e + 4 * tb1 / d * srm / 4 / Es, es_bare, ss_bare)
        d_all = range(0,numsteps)*e*srm/2/(numsteps-1)
        contit = 1
        countit = 0

        # 3.3 Iteration until equilibrium and compatibility fulfilled
        while contit == 1 and countit < 1000:
            countit += 1
            ss_all = np.zeros(numsteps, dtype=cplx)
            es_all = np.zeros_like(ss_all)
            tb_all = np.zeros_like(ss_all)
            ss_all[-1] = ssr
            es_all[-1] = gete(ssr, es_bare, ss_bare)

            # Backwards iteration from crack towards zero slip point
            for i in list(range(numsteps - 1, 0, -1)):

                tbi = gett(ss_all[i],d_all[i],fsy,tb0,tb1,ds_bare,ts_bare)
                tb_all[i] = tbi
                # Next normal stress value from equilibrium
                ss_all[i - 1] = ss_all[i] - delta_x * 4 * tbi / d
                if ss_all[i - 1] < 0:
                    ss_all[i - 1] = cplx(0, 0)
                elif ss_all[i - 1] > ss_bare[-1]:
                    ss_all[i - 1] = ss_bare[-1]
                es_all[i - 1] = gete(ss_all[i - 1], es_bare, ss_bare)

            # Compare resulting mean strain with given epsilon (must be equal)
            de_it_real = abs((vecmean(es_all).real-e.real)/e.real)
            de_it_imag = abs((vecmean(es_all).imag-e.imag)/e.imag)
            if abs(e.imag) > 0:
                de_it = de_it_real + de_it_imag
            else:
                de_it = de_it_real

            # Decision, whether iteration is continued
            if de_it < 0.00001:
                contit = 0
                print(e)
            else:
                es_all = es_all * e / vecmean(es_all)
                ssr = gets(es_all[-1], es_bare, ss_bare)
                d_all[0] = 0
                for i in range(1,numsteps):
                    d_all[i] = d_all[i-1] + (es_all[i-1]+es_all[i])/2*delta_x


        # 4 Safe results in stress object
        xcr = range(0,numsteps)
        self.xcr = [i * srm/2/(numsteps-1) for i in xcr]
        if Es == self.Esx:
            self.ds_all = [i.real for i in d_all]
            self.ss_all = [i.real for i in ss_all]
            self.es_all = [i.real for i in es_all]
            self.tbs_all = [i.real for i in tb_all]
        elif Es == self.Epx:
            self.dp_all = [i.real for i in d_all]
            self.sp_all = [i.real for i in ss_all]
            self.ep_all = [i.real for i in es_all]
            self.tbp_all = [i.real for i in tb_all]

        # 5 return steel stress at crack
        print('conveeeeerge')
        print(ss_all[-1])
        if countit > 998:
            ss_all = [fsy for i in ss_all]
        return ss_all[-1]

    def sr0_vc(self):
        """ --------------------------------- Calculate diagonal crack spacing------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - th: Principal direction
            - fct: concrete tensile strength
            - rhox, rhoy: Reinforcement contents
            - dx,dy: Reinforcement diameters
            - tb0: bond stress elastic
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - srx0, sry0: Maximum crack spacing of tension chord in x and y direction
            - sr: Diagonal crack spacing
            - srx, sry: Crack spacing in reinforcement direction
        -------------------------------------------------------------------------------------------------------------"""
        # 0 Assign
        th = self.thr
        rhox = self.rhox
        rhopx = self.rhopx
        rhoy = self.rhoy
        rhopy = self.rhopy
        tb0 = self.tb0
        tb1 = self.tb1
        tbp0 = self.tbp0
        dx = self.dx
        dy = self.dy
        dpx = self.dpx
        dpy = self.dpy
        fct = self.fct
        fsyx = self.fsyx
        fsyy = self.fsyy
        fsux = self.fsux
        fsuy = self.fsuy


        # Flag Prototype and Girders
        # if self.dx == 14:
        #     rhox = 0.02138
        #     rhopx = 0.00880

        # 0.1 Manipulate (mirror) negative angles
        if th < 0:
            th = -th

        # 1 Initial Assumption (Formula 5.7 in Kaufmann, 1998 extended for prestressing)
        # 1.1 in x
        if rhox + rhopx > 10**-9:
            if abs(rhopx) <1e-9:
                self.srx0 = (2*tb0/dx*rhox)**(-1)*fct*(1-rhox-rhopx)
            else:
                self.srx0 = (2*tb0/dx*rhox+2*tbp0/dpx*rhopx)**(-1)*fct*(1-rhox-rhopx)
        else:
            self.srx0 = 10**3

        # 1.1 Maximum possible crack spacing
        if rhox + rhopx > 10 ** -9:
            srxmax = dx/(2*tb0)*fsyx + dx/(2*tb1)*(fsux-fsyx)
        else:
            srxmax = 10 ** 3
        if self.srx0 > srxmax:
            self.srx0 = srxmax

        # 1.2 in y
        if rhoy + rhopy > 10**-9:
            if abs(rhopy) < 1e-9:
                self.sry0 = (2 *tb0/dy*rhoy)**(-1)*fct*(1-rhoy-rhopy)
            else:
                self.sry0 = (2 *tb0/dy*rhoy+2*tbp0/dpy*rhopy)**(-1)*fct*(1-rhoy-rhopy)
        else:
            self.sry0 = 10**3

        # 1.2 Maximum possible crack spacing
        if rhoy + rhopy > 10 ** -9:
            srymax = dy/(2*tb0)*fsyy + dy/(2*tb1)*(fsuy-fsyy)
        else:
            srymax = 10**3
        if self.sry0 > srymax:
            self.sry0 = srymax

        # 2 Actual crack spacing as a function of lambda
        self.sr = lbd / ((sin(th) / self.srx0) + (cos(th) / self.sry0))

        # 3 Recalculate spacings in reinforcement directions
        # 3.1 x-direction
        if abs(sin(th)) > 0:
            self.srx = self.sr/(sin(th))
        else:
            self.srx = self.sr/10**-10

        if self.srx > srxmax:
            self.srx = srxmax

        # 3.2 y-direction
        if abs(cos(th)) > 0:
            self.sry = self.sr/(cos(th))
        else:
            self.sry = self.sr / 10 ** -10

        if self.sry > srymax:
            self.sry = srymax



        # if self.rhoy + self.rhopy < 10**-9:
        #     self.srx = lbd*self.srx0
        #     self.sr = self.srx*sin(th)
        #     self.sry = self.sr/cos(th)
        #     self.sry0 = self.sry/lbd

        # Custom definition for Slab strips!!
        # Flag
        # self.sry = 100
        # self.sry0 = self.sry/lbd

    def sigma_cart_1(self):
        """ ---------------------- Get In Plane Stresses for Linear Elastic Material Law ----------------------------"""
        self.sr0_vc()  # potentially remove this line of code again, as it produces errors. And it is not required for the lin.el. law to calculate these properties.
        self.sx_xy = 0
        self.sy_xy = 0
        self.txy_x = 0
        self.txy_y = 0
        
        if self.cm_klij == 1 or self.l % 2 == 0:
            # material 1 = lin.el. steel 
            # or even layer number (0, 2,..) --> material 1 (glass)
            E = self.Ec
            v = self.v        
        else: 
            # uneven layer number --> material 2 (PVB),
            E = self.Ec2
            v = self.vc2

        self.sx_x = E/(1-v**2)*self.ex
        self.sx_y = E / (1 - v ** 2) * v * self.ey
        self.sx_xy = 0
        self.sy_x = E/(1-v**2)*v*self.ex
        self.sy_y = E / (1 - v ** 2) * self.ey
        self.sy_xy = 0
        self.txy_x = 0
        self.txy_y = 0
        self.txy_xy = E/(1-v**2)*(1-v)/2*self.gxy

        self.sx = self.sx_x + self.sx_y + self.sx_xy
        self.sy = self.sy_x + self.sy_y + self.sy_xy
        self.txy = self.txy_x + self.txy_y + self.txy_xy

        self.ssx = (self.ex+ecsx)*self.Esx
        self.ssy = self.ey*self.Esy
        self.spx = (self.ex+ecsy)*self.Epx
        self.spy = self.ey*self.Epy
        self.sc3 = self.e3*self.Ec
        self.sc1 = self.e1*self.Ec
        self.fc_soft = self.fcp
        self.dn = 0
        self.dt = 0

    def sigma_cart_31(self):
        """ -------------------------- Get In Plane Stresses for Compression - Compression -----------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Layer info
        self.cm_klij = 3
        # 2 Steel Contribution
        if self.rhox > 0:
            self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
        else:
            self.ssy = 0

        # 3 CFRP Contribution
        if self.rhopx > 0:
            self.spx,_ = self.spr(self.ex, self.ecx, ecsx, self.rhopx, 100, self.dpx, self.fpux, self.Epx, self.tbp0)
        else:
            self.spx = 0
        if self.rhopy > 0:
            self.spy,_ = self.spr(self.ey, self.ecy, ecsy, self.rhopy, 100, self.dpy, self.fpuy, self.Epy, self.tbp0)
        else:
            self.spy = 0

        # 4 Concrete Constitutive Matrix
        # 4.1 Principal Concrete stresses
        self.sc1 = self.sc(self.ec1)
        self.sc3 = self.sc(self.ec3)

        # 4.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: assign "zero" (not possible in compression-compression case)
        self.scx = self.sc3 * cos(self.thc) ** 2 + self.sc1 * sin(self.thc) ** 2
        self.scy = self.sc3 * sin(self.thc) ** 2 + self.sc1 * cos(self.thc) ** 2
        self.txy = (self.sc1 - self.sc3) * sin(self.thc) * cos(self.thc)

        # 5 Assemble resulting stresses
        # 5.1 Stress assembly
        self.v=0
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy)*(1-self.rhox-self.rhopx) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx)*(1-self.rhox-self.rhopx) + self.rhoy*self.ssy + self.rhopy*self.spy

        # 6 Manipulate parameters that do not occurr in compression-compression
        self.sr0_vc()
        self.dn = 0
        self.dt = 0

    def sigma_cart_32(self):
        """ ---------------------------- Get In Plane Stresses for Cracked Membrane Model ------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""
        # 0 Adjust fixed crack model if (i) existing crack has the opposite direction of the new one (if it could form)
        #   or (ii) the principal stress of the last iteration does not exceed the concrete tensile strength
        if self.cm_klij == 4:
            mprev = (self.sprev_klij.sx + self.sprev_klij.sy)/2
            rprev = sqrt((self.sprev_klij.sx - self.sprev_klij.sy)**2/4 + self.sprev_klij.txy**2)
            self.s1prev = mprev + rprev

            # if do_cracked:
            if self.rhox + self.rhopx > 0 and self.rhoy + self.rhopy > 0:
                if self.s1prev < self.fct:
                    self.cm_klij = 3
                    self.principal()
            if self.sprev_klij.cm_klij == 1:
                self.cm_klij = 3
                self.principal()
            elif self.sprev_klij.e3 > -ff_e3:
                self.cm_klij = 3
                self.principal()
            elif self.sprev_klij.submodel == 3 or self.sprev_klij.submodel == 1:
                self.cm_klij = 3
                self.principal()
            elif abs(self.sprev_klij.th) > 80 / 90 * pi / 2 or abs(self.sprev_klij.th) < 10 / 90 * pi / 2:
                self.cm_klij = 3
                self.principal()

        # Flag
        # elif self.cm_klij == 3:
        #     if self.rhoy + self.rhopy < 10**-9:
        #         self.cm_klij = 1
        #         self.out([self.ex,self.ey,self.gxy,self.gxz,self.gyz])

        # 1 Crack Spacing and Kinematics
        self.sr0_vc()

        # 2 Steel Contribution
        if self.rhox > 0:
            if self.ex > self.ecx:
                self.ssx, self.ssxk = self.ssr(self.ex, self.ecx, ecsx, self.srx, self.rhox, self.rhopx, self.dx, self.fsyx, self.fsux, self.Esx,
                                        self.Eshx, self.tb0, self.tb1, self.Ec)
            else:
                self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
                self.ssxk = self.ssx
        else:
            self.ssx = 0
            self.ssxk = 0
        if self.rhoy > 0:
            if self.ey > self.ecy:
                self.ssy, self.ssyk = self.ssr(self.ey, self.ecy, ecsy, self.sry, self.rhoy, self.rhopy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                        self.Eshy, self.tb0, self.tb1, self.Ec)
            else:
                self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
                self.ssyk = self.ssy
        else:
            self.ssy = 0
            self.ssyk = 0

        # 3 CFRP Contribution
        if self.rhopx > 0:
            self.spx, self.spxk = self.spr(self.ex, self.ecx, ecsx, self.rhopx, self.srx, self.dpx, self.fpux, self.Epx, self.tbp0)
        else:
            self.spx = 0
            self.spxk = 0

        if self.rhopy > 0:
            self.spy, self.spyk = self.spr(self.ey, self.ecy, ecsy, self.rhopy, self.sry, self.dpy, self.fpuy, self.Epy, self.tbp0)
        else:
            self.spy = 0
            self.spyk = 0

        # 4 Concrete Constitutive Matrix
        # 4.0 If CMM_F: calculate crack kinematics and check whether submodel has changed due to resulting
        #               characteristic concrete stresses
        if self.cm_klij == 4:
            self.crack_kin()
            if self.submodel == 1:
                self.cm_klij = 3
                self.principal()
                self.sigma_cart_31()
            elif self.submodel == 3:
                self.cm_klij = 3
                self.principal()
                self.sigma_cart_33()

        # 4.1 Principal Concrete stresses (if not already calculated in crack kinematics)
        if self.cm_klij == 3:
            self.sc1 = self.sc(self.ec1)
            self.sc3 = self.sc(self.ec3)

        # if self.sc1 > 0:
        #     self.sc1 = ff*self.ec1
        # if self.sc3 > 0:
        #     self.sc3 = ff*self.ec1

        # 4.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: Ec if uncracked
        #           "zero" if ec1 > cracking strain
        self.scx = self.sc3 * cos(self.thc) ** 2 + self.sc1 * sin(self.thc) ** 2
        self.scy = self.sc3 * sin(self.thc) ** 2 + self.sc1 * cos(self.thc) ** 2
        self.txy = (self.sc1 - self.sc3) * sin(self.thc) * cos(self.thc)

        # if self.cm_klij == 4:
        #     scx_contr = self.sctr*cos(self.thr) ** 2 + self.scnr*sin(self.thr) ** 2 - 2*self.tctnr*sin(self.thr)*cos(self.thr)
        #     scy_contr = self.sctr*sin(self.thr) ** 2 + self.scnr*cos(self.thr) ** 2 + 2*self.tctnr*sin(self.thr)*cos(self.thr)
        #     tcxy_contr = (self.scnr-self.sctr)*sin(self.thr)*cos(self.thr)-self.tctnr*(cos(self.thr)**2-sin(self.thr)**2)
        #
        #     print(self.txy.real,tcxy_contr.real)

        # txy_contr = (self.scnr-self.sctr)*sin(self.thr)*cos(self.thr)-self.tctnr*(cos(self.thr)**2-sin(self.thr)**2)
        # if abs(self.txy-txy_contr)> 0.001:
        #     print('control value of txy not coinciding')
        #     print(self.txy)
        #     print(txy_contr)

        # 5 Assemble resulting stresses
        # 5.1 Set Poisson's Ratio to zero
        self.v = 0

        # 5.2 Stress assembly
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy + self.rhopy*self.spy

        # 6 Crack kinematics for rotating cracks
        if self.cm_klij == 3:
            self.dn = self.sr*self.e1
            self.dt = 0

    def sigma_cart_33(self):
        """ ------------------------------ Get In Plane Stresses for Tension - Tension ---------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""
        # 0 Layer info
        self.cm_klij = 3

        # 1 Crack spacing
        self.sr0_vc()

        # 2 Steel Contribution
        if self.rhox > 0:
            self.ssx, self.ssxk = self.ssr(self.ex, self.ecx, ecsx, self.srx0*lbd, self.rhox, self.rhopx, self.dx, self.fsyx, self.fsux, self.Esx,
                                    self.Eshx, self.tb0, self.tb1, self.Ec)
        else:
            self.ssx = 0
            self.ssxk = 0
        if self.rhoy > 0:
            self.ssy, self.ssyk = self.ssr(self.ey, self.ecy, ecsy, self.sry0*lbd, self.rhoy, self.rhopy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                    self.Eshy, self.tb0, self.tb1, self.Ec)
        else:
            self.ssy = 0
            self.ssyk = 0

        # 3 CFRP Contribution
        if self.rhopx > 0:
            self.spx, self.spxk = self.spr(self.ex, self.ecx, ecsx, self.rhopx, self.srx0*lbd, self.dpx, self.fpux, self.Epx, self.tbp0)
        else:
            self.spx = 0
            self.spxk = 0
        if self.rhopy > 0:
            self.spy, self.spyk = self.spr(self.ey, self.ecy, ecsy, self.rhopy, self.sry0*lbd, self.dpy, self.fpuy, self.Epy, self.tbp0)
        else:
            self.spy = 0
            self.spyk = 0

        # 4 Concrete Contribution
        # 4.1 If ex < cracking strain: Assign linear elastic Law in x/y
        #     Else: Set "zero" stiffness in tensile direction: ff for numerical stability
        self.fcs()
        self.sc1 = 0
        self.sc3 = 0

        self.scx = ff*self.ex
        self.scy = ff*self.ey

        # 5 Assemble resulting stresses
        # 5.1 Set Poisson's Ratio to zero
        self.v = 0

        # 5.2 Stress assembly
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy + self.rhopy*self.spy
        self.txy = ff/2*self.gxy

        # 6 Crack Kinematics
        self.dn = self.srx0*lbd*self.ex
        self.dt = 0
        self.sr = self.srx0*lbd

    def sigma_shear(self):
        """ ---------------------- Generate Shear Stiffness Matrix in xz and yz Direction -------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - E: Young's Modulus
            - v: Poisson's Ratio
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -ET: Elastic Matrix
        -----------------------------------------------------------------------------------------------------------------"""
        G = (self.Ec + self.Ec) / (4 * (1 + self.v))
        self.txz = 5/6*G*self.gxz
        self.tyz = 5/6*G*self.gyz

    def out(self,e):
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

        # 3 Output
        ff=self.Ec/100
        if abs(self.txy.imag) < abs(self.gxy.imag)*ff/2:
            self.txy = cplx(self.txy.real,self.gxy.imag*ff/2)
        # if abs(self.sy.imag) < abs(self.ey.imag)*ff:
        #     self.sy = cplx(self.sy.real,self.ey.imag*ff)
        self.s = [self.sx,self.sy,self.txy,self.txz,self.tyz]