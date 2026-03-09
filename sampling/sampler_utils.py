import numpy as np
from Stresses_mixreinf import stress
import pyDOE as doe
import pandas as pd
import wandb

class SamplerUtils:
    def __init__(self, t_1, t_2, nl, mat, nel, E1, nu1, E2, nu2, other=None, mat_dict=None):
        self.t_1 = t_1
        self.t_2 = t_2
        self.nl = nl
        self.mat = mat
        self.nel = nel
        self.E1 = E1
        self.nu1 = nu1
        self.E2 = E2
        self.nu2 = nu2
        self.other = other
        self.mat_dict = mat_dict

        if self.E2 == None:
            # in the case of lin.el. or glass         
            self.E = E1
            self.nu = nu1

        if self.other is None:
            pass
        elif self.other.shape[1] == 10:
            self.add = 1       # t, rho_x, rho_y, CC
        elif self.other.shape[1] == 9:
            self.add = 0       # t, rho, CC


    def find_e(self, e0,eh,go):
            """ ------------------------------------------- Membrane Elements --------------------------------------------------
                ------------------------------------- Calculate element strains ---------------------------------------------
                --------------------------------------------    INPUT: ------------------------------------------------------
                - u         --> Node deformations
                --------------------------------------------- OUTPUT:--------------------------------------------------------
                - e        --> Matrix of element strains [e_xi e_yi g_xyi]
                - ex       --> Element normal strains in x-Direction
                - ey       --> Element normal strains in y-Direction
                - gxy      --> Element shear strains
                - e1       --> Element principal tensile strains
                - e3       --> Element principal compressive strains
                - th       --> Element principal directions
            -----------------------------------------------------------------------------------------------------------------"""

            nel = self.nel
            if self.mat == 10: 
                nlk_max = int(max(self.nl)[0])
            else: 
                nlk_max = self.nl
            e = np.zeros((nel, nlk_max, go, go, 5))
            ex = np.zeros((nel, nlk_max, go, go))
            ey = np.zeros((nel, nlk_max, go, go))
            gxy = np.zeros((nel, nlk_max, go, go))
            e1 = np.zeros((nel, nlk_max, go, go))
            e3 = np.zeros((nel, nlk_max, go, go))
            th = np.zeros((nel, nlk_max, go, go))
            for k in range(nel):
                if self.mat == 10:
                    nlk = int(self.nl[k])
                else: 
                    nlk = self.nl
                for l in range(nlk):
                    for i in range(go):
                        for j in range(go):
                            # would be for triangular elements
                            # if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            #     e[k][l][i][j][:] = -10**5*np.ones_like(e[k][l][i][j][:])
                            #     ex[k][l][i][j] = -10**5*np.ones_like(ex[k][l][i][j])
                            #     ey[k][l][i][j] = -10**5*np.ones_like(ey[k][l][i][j])
                            #     gxy[k][l][i][j] = -10**5*np.ones_like(gxy[k][l][i][j])
                            #     e1[k][l][i][j] = -10**5*np.ones_like(e1[k][l][i][j])
                            #     e3[k][l][i][j] = -10**5*np.ones_like(e3[k][l][i][j])
                            #     th[k][l][i][j] = -10**5*np.ones_like(th[k][l][i][j])
                            # else:
                            eh_kij = eh[k, i, j, :]
                            e_klij = self.find_e_klij(eh_kij, k, l, i, j, [self.t_1[k,0,0], self.t_2[k,0,0]])
                            e0_klij = e0[k][l][i][j][:]
                            e[k][l][i][j][:]=np.transpose(e_klij)-np.transpose(e0_klij)
                            ex[k][l][i][j] = e_klij[0]-e0_klij[0]
                            ey[k][l][i][j] = e_klij[1]-e0_klij[1]
                            gxy[k][l][i][j] = e_klij[2]-e0_klij[2]
                            [e1_klij, e3_klij, th_klij] = self.e_principal(ex[k][l][i][j], ey[k][l][i][j], gxy[k][l][i][j])
                            e1[k][l][i][j] = e1_klij
                            e3[k][l][i][j] = e3_klij
                            th[k][l][i][j] = th_klij
            return e, ex, ey, gxy, e1, e3, th


    def find_e_klij(self, eh_kij, k, l, i, j, t_elem):
        """ ------------------------------------------- Shell Elements --------------------------------------------------
            --------------------------- Find strains in element k in layer l and gauss point (i,j) -------------------"""
        
        cm_k = self.mat
        if cm_k == 10:
            nlk = self.nl[k][0]
            t_k1, t_k2 = t_elem[0], t_elem[1]
            t_k_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
            t = np.zeros((1,int(nlk)))
            for l_ in range(int(nlk)):
                if l_%2 == 0:
                    t[0,l_] = t_k1
                else: 
                    t[0,l_] = t_k2
                t_cum = np.cumsum(t)
            if l == 0:
                z = -t_k_tot/2 + 0.5*t[0,l]
            else: 
                z = -t_k_tot/2 + t_cum[l-1] + 0.5*t[0,l]
        else: 
            t_k = t_elem[0]
            nlk = self.nl
            z = -t_k/2+(2*l+1)*t_k/(2 * nlk)
            
        S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                    [0, 1, 0, 0, -z, 0, 0, 0],
                    [0, 0, 1, 0, 0, -z, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])
        e_klij = np.matmul(S,eh_kij)
        # e_klij[e_klij == 0] = 1e-13
        # # control sizes of strains in x, y:
        # if abs(e_klij[0])<0.5e-6: 
        #     e_klij[0] = -0.5e-6
        # if abs(e_klij[1])<0.5e-6:
        #     e_klij[1] = -0.5e-6
        # if abs(e_klij[2])<0.5e-7:
        #     e_klij[2] = -0.5e-7
        return e_klij



    def e_principal(self, ex, ey, gxy):

        r = 1 / 2 * np.sqrt((ex - ey) * (ex - ey) + gxy * gxy)
        m = (ex + ey) / 2
        e1 = m + r
        e3 = m - r

        # if e1 > ex:
        #     th = atan(gxy/(2*(e1-ex)))
        # else:
        #     # print("obacht")
        #     th = np.sign(gxy)*pi/2
        if gxy == 0:
            if ex > ey:
                th = np.pi / 2
            elif ex < ey:
                th = 0
            elif ex == ey:
                th = np.pi / 4
        elif abs(gxy) > 0:
            th = np.arctan(gxy / (2 * (e1 - ex)))

        return [e1, e3, th]



    def find_s(self, e,go, count = [0,0,0], rho_sublayer = False):
            """ ------------------------------- Calculate element stresses at all gauss points ------------------------------
                --------------------------------------------    INPUT: ------------------------------------------------------
                - e        --> Strains in form [k][l][i][j]
                --------------------------------------------- OUTPUT:--------------------------------------------------------
                - s        --> Matrix of element stresses [s_xi s_yi t_xyi] in form [k][l][i][j]
                - sx       --> Element normal stresses in x-Direction
                - sy       --> Element normal stresses in y-Direction
                - txy      --> Element shear stresses
            -----------------------------------------------------------------------------------------------------------------"""
            # 0 Initiate Time Measurement

            nel = self.nel
            if self.mat == 10: 
                nlk_max = int(max(self.nl))
            else: 
                nlk_max = self.nl
            s = np.ndarray((nel, nlk_max, go, go,3,5), dtype=np.complex128)
            for k in range(nel):
                if self.mat == 10:
                    nlk = int(self.nl[k][0])
                else: 
                    nlk = self.nl
                for l in range(nlk):
                    for i in range(go):
                        for j in range(go):
                            e_klij = e[k][l][i][j][:]
                            # MATK = self.MATK
                            # MAT = [MATK["Ec"][k],MATK["vc"][k],MATK["fcp"][k],MATK["fct"][k],MATK["ec0"][k],
                            #     0 * MATK["fct"][k] / MATK["Ec"][k],MATK["Esx"][k],MATK["Eshx"][k],MATK["fsyx"][k],
                            #     MATK["fsux"][k],MATK["Esy"][k],MATK["Eshy"][k],MATK["fsyy"][k],MATK["fsuy"][k],
                            #     MATK["tb0"][k],MATK["tb1"][k],MATK["Epx"][k],MATK["Epy"][k],MATK["tbp0"][k],
                            #     MATK["tbp1"][k],MATK["ebp1"][k],MATK["fpux"][k],MATK["fpuy"][k], 
                            #     MATK["Ec2"], MATK["vc2"]]                            
 
                            # GEOMK = self.GEOMK
                            # GEOM = [GEOMK["rhox"][k][l],GEOMK["rhoy"][k][l],GEOMK["dx"][k][l],GEOMK["dy"][k][l],
                            #         GEOMK["rhopx"][k][l],GEOMK["rhopy"][k][l],GEOMK["dpx"][k][l],GEOMK["dpy"][k][l], 
                            #         GEOMK["t"], GEOMK["t2"], l]

                            if self.mat !=3:
                                fcp, fct, ec0, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, tb0, tb1 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                Epx, Epy, tbp0, tbp1, ebp1, fpux, fpuy = 0, 0, 0, 0, 0, 0, 0
                                MAT = [self.E1, self.nu1, fcp, fct, ec0, 0 * fct / self.E1, 0, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, 
                                    tb0, tb1, Epx, Epy, tbp0, tbp1, ebp1, fpux, fpuy, self.E2, self.nu2]
                                GEOM = [0]*13
                            elif self.mat == 3:
                                MAT = [self.other[k,self.add+2][0], self.nu1, self.other[k,self.add+7][0], self.other[k,self.add+8][0], self.other[k,self.add+6][0], 
                                       self.other[k,self.add+5][0], self.mat_dict['Dmax'], self.mat_dict['Es'], self.mat_dict['Esh'], self.mat_dict['fsy'],
                                       self.mat_dict['fsu'], self.mat_dict['Es'], self.mat_dict['Esh'], self.mat_dict['fsy'],self.mat_dict['fsu'],
                                       self.other[k,self.add+3][0], self.other[k,self.add+4][0], 0, 0, 0, 
                                       0, 0, 0, 0, self.E2, self.nu2]
                                
                                if rho_sublayer:
                                    if l < 4 or l> nlk-5:
                                        GEOM = [self.other[k,0][0], self.other[k,self.add][0], self.mat_dict['D'], self.mat_dict['D'], 
                                                self.mat_dict['s'], self.mat_dict['s'],
                                                0, 0, 0, 0,
                                                self.t_1[k,0,0], 0, l]
                                    else: 
                                        GEOM = [0, 0, self.mat_dict['D'], self.mat_dict['D'], 
                                                self.mat_dict['s'], self.mat_dict['s'],
                                                0, 0, 0, 0,
                                                self.t_1[k,0,0], 0, l]
                                else: 
                                    GEOM = [self.other[k,0][0], self.other[k,self.add][0], self.mat_dict['D'], self.mat_dict['D'], 
                                                self.mat_dict['s'], self.mat_dict['s'],
                                                0, 0, 0, 0,
                                                self.t_1[k,0,0], 0, l]

                            if self.mat == 3:
                                # note: set s_prev to zero everywhere. Should not be required as cm_klij is never 4 (fixed cracks)
                                sig = stress(0, self.mat, l, k, i, j, MAT, GEOM, 1, 1, count=count)
                                sig.out(e_klij+np.array([0.0000000000000001j,0,0,0,0]))
                                s[k][l][i][j][0] = sig.to_array()

                                sig = stress(0, self.mat, l, k, i, j, MAT, GEOM, 1, 1, count=count)
                                sig.out(e_klij+np.array([0,0.0000000000000001j,0,0,0]))
                                s[k][l][i][j][1] = sig.to_array()

                                sig = stress(0, self.mat, l, k, i, j, MAT, GEOM, 1, 1, count=count)
                                sig.out(e_klij+np.array([0,0,0.0000000000000001j,0,0]))
                                s[k][l][i][j][2] = sig.to_array()

                            elif self.mat != 3:
                                sig = stress(0, self.mat, l, k, i, j, MAT, GEOM, 1, 1, count=count)
                                sig.out(e_klij)
                                s[k][l][i][j][0] = sig.to_array()

            return s


    def find_sh(self, s,go):
        num_elements = self.nel
        sh = np.zeros((num_elements, go, go, 8))
        for k in range(num_elements):
            cm_k = self.mat
            if cm_k == 10:
                nlk = int(self.nl[k][0])
                t_k1, t_k2 = self.t_1[k,0,0], self.t_2[k,0,0]
                t = np.zeros((1, nlk))
                t_k_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
                for l_ in range(nlk):
                    if l_%2 == 0:
                        t[0,l_] = t_k1
                    else: 
                        t[0,l_] = t_k2
                t_cum = np.cumsum(t)
            else: 
                nlk = self.nl
                t_k = self.t_1[k,0,0]


            for i in range(go):
                for j in range(go):
                    # if self.ELS[4][k] == 3 and i == 1 and j == 1:
                    #     sh[k][i][j][:] = -10**5*np.ones_like(sh[k][i][j][:])
                    # else:
                    sh_kij = np.zeros((8,1))
                    for l in range(nlk):
                        st = s[k][l][i][j][0]
                        # s_klij = np.array([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real])
                        s_klij = st.real
                        s_klij = np.ndarray.reshape(s_klij, 5, 1)
                        if cm_k == 10:
                                if l == 0:
                                    z = -t_k_tot/2 + 0.5*t[0,l]
                                else: 
                                    z = -t_k_tot/2 + t_cum[l-1] + 0.5*t[0,l]
                        else: 
                            z = -t_k / 2 + (2 * l + 1) * t_k / (2 * nlk)
                        S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                                    [0, 1, 0, 0, -z, 0, 0, 0],
                                    [0, 0, 1, 0, 0, -z, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
                        if cm_k == 10:
                            sh_kij = sh_kij + np.transpose(S)@s_klij*t[0,l]
                        else: 
                            sh_kij = sh_kij + np.transpose(S)@s_klij*t_k/nlk
                    sh[k][i][j][:] = sh_kij.reshape(8,)
        return sh



########### CALCULATION OF D ##################

    def get_et_vb(self, cm_klij,s_klij, k):
        """ ------------------------------------- Define Constitutive Matrix -----------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - cm_klij
                - 1: Linear Elasticity
                - 2: CMM-, CMM without tension stiffening (NOT IMPLEMENTED YET)
                - 3: CMM
                - 10: Glas [Vera]
            - k: Element number
            - l: Layer number
            - [ex_klij, ey_klij, gxy_klij]: Strain state
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -ET: Secant Constitutive Matrix
        -----------------------------------------------------------------------------------------------------------------"""
        s = s_klij
        if cm_klij == 1:
            E = self.E
            v = self.nu
            ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
        elif cm_klij == 3: 
            # ET = np.array([[s[0].sx.imag, s[1].sx.imag, s[2].sx.imag],
            #             [s[0].sy.imag, s[1].sy.imag, s[2].sy.imag],
            #             [s[0].txy.imag, s[1].txy.imag, s[2].txy.imag]])/0.0000000000000001
            ET = np.array([[s[0][0].imag, s[1][0].imag, s[2][0].imag],
                        [s[0][1].imag, s[1][1].imag, s[2][1].imag],
                        [s[0][2].imag, s[1][2].imag, s[2][2].imag]])/0.0000000000000001
        elif cm_klij == 10:
            E_1 = self.E1
            E_2 = self.E2
            v_1 = self.nu1
            v_2 = self.nu2
            ET_1 = E_1 / (1 - v_1**2) * np.array([[1, v_1, 0], [v_1, 1, 0], [0, 0, 0.5 * (1 - v_1)]])
            ET_2 = E_2 / (1-v_2**2) * np.array([[1, v_2, 0], [v_2, 1, 0], [0, 0, 0.5 * (1 - v_2)]])
            ET = {'ET_1': ET_1,
                    'ET_2': ET_2}
        return ET
    


    def dh_kij(self,s_kij, k, cm_k):
        Dmh = np.zeros((3, 3))
        Dbh = np.zeros((3, 3))
        Dmbh = np.zeros((3, 3))
        Dsh = np.zeros((2, 2))
        if cm_k == 1 or cm_k == 3:
            t_k = self.t_1[k,0,0]
            nlk = self.nl
            for l in range(nlk):
                z = -t_k / 2 + (2 * l + 1) * t_k / (2 * nlk)
                if cm_k == 1:
                    E = self.E
                elif cm_k == 3:
                    E = self.other[k,2][0]
                v = self.nu

                # Dp = self.get_et(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)
                Dp = self.get_et_vb(cm_k, s_kij[l,:], k)
                Ds = np.array([[5 / 6 * (E + E) / (4 * (1 + v)), 0], [0, 5 / 6 * (E + E) / (4 * (1 + v))]])
                Dmh_l=Dp
                Dmbh_l= -z*Dp
                Dbh_l = z*z*Dp
                Dsh_l = Ds
                Dmh = Dmh + Dmh_l * t_k / nlk
                Dbh = Dbh + Dbh_l * t_k / nlk
                Dmbh = Dmbh + Dmbh_l * t_k / nlk
                Dsh = Dsh + Dsh_l * t_k / nlk
        elif cm_k == 10:
            nlk = int(self.nl[k][0])
            t, E, v = np.zeros((1,nlk)), np.zeros((1,nlk)), np.zeros((1,nlk))
            Dp = np.zeros((nlk,3,3))
            t_k1, t_k2 = self.t_1[k,0,0], self.t_2[k,0,0]
            t_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
            E_1, E_2 = self.E1, self.E2
            v_1, v_2 = self.nu1, self.nu2
            for l in range(nlk):
                Dp_all = self.get_et_vb(cm_k, s_kij[l], k)
                if l%2 == 0:
                    t[0,l] = t_k1
                    E[0,l] = E_1
                    v[0,l] = v_1
                    Dp[l,:,:] = Dp_all['ET_1']
                else: 
                    t[0,l] = t_k2
                    E[0,l] = E_2
                    v[0,l] = v_2
                    Dp[l,:,:] = Dp_all['ET_2']
                t_cum = np.cumsum(t)
            for l in range(nlk):
                if l == 0:
                    z = -t_tot/2 + 0.5*t[0,l]
                else: 
                    z = -t_tot/2 + t_cum[l-1] + 0.5*t[0,l]
                Ds = np.array([[5 / 6 * (E[0,l] + E[0,l]) / (4 * (1 + v[0,l])), 0], [0, 5 / 6 * (E[0,l] + E[0,l]) / (4 * (1 + v[0,l]))]])
                Dmh_l=Dp[l,:,:]
                Dmbh_l= -z*Dp[l,:,:]
                Dbh_l = z*z*Dp[l,:,:]
                Dsh_l = Ds
                Dmh = Dmh + Dmh_l * t[0,l]
                Dbh = Dbh + Dbh_l * t[0,l]
                Dmbh = Dmbh + Dmbh_l * t[0,l]
                Dsh = Dsh + Dsh_l * t[0,l]
        return Dmh,Dmbh,Dbh,Dsh
    

    def get_dh(self, s_k):
        D_tot = np.zeros((s_k.shape[0], 8, 8))
        for k in range(s_k.shape[0]):
            Dmh,Dmbh,Dbh,Dsh = self.dh_kij(s_k[k,:,0,0,:], k, self.mat)
            De_1 = np.hstack([Dmh,Dmbh,np.zeros((3, 2))])
            De_2 = np.hstack([Dmbh,Dbh,np.zeros((3,2))])
            De_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), Dsh])
            De = np.vstack([De_1, De_2, De_3])
            D_tot[k, :, :] = De
        return D_tot
    


class Sampler_utils_vb:
    def __init__(self, E1, nu1, E2 = None, nu2 = None, mat_dict = None):
        self.E1 = E1
        self.nu1 = nu1
        self.E2 = E2
        self.nu2 = nu2
        self.mat_dict = mat_dict

    def D_an(self, eps:np.array, t: np.array, num_layers = 20, mat = 1, calc_meth = 'single', discrete = 'andreas', rho_sublayer = False):
        '''
        returns analytically calculated sig for given eps and t
        E, nu are assumed constant
        eps             (np.array)       [-] or [1/mm]                                  shape: m
        t               (np.array)       [mm]                                           shape: n
        num_layers      (int)            [-] amount of layers
        mat             (int)            material law to be used (1 = lin.el., 3 = reinforced concrete, 10 = glass / lin.el.)
        calc_meth       (str)            [parameter deprecated]
                                         'all': calculate every eps with every t --> get m x n data points
                                         'single': calculate single eps with single t --> get n data points
                                                --> m = n needs to hold for this method
                                                --> use 'single' to not over-sample.
        discrete        (str)           should be set to discrete = 'andreas', other methods are deprecated. 
        rho_sublayer    (bool)          if True: calculates stresses with rho only in 4 top and bottom layers (not in all)

        OUT:
        mat_analytical  (dict)          containing 'D_a' and 'sig_a': analytically calculated values for stiffness and stresses in [N,mm]
        '''

        if discrete == False: 
            raise RuntimeError('Method not in use, could lead to errors')
            D_an_1 = np.concatenate([t*D_p, np.zeros_like((t*D_p)), np.zeros((t.shape[0], 3, 2))], axis = 2)
            D_an_2 = np.concatenate([np.zeros_like((t*D_p)), (1/12)*(t**3)*D_p,  np.zeros((t.shape[0], 3, 2))], axis = 2)
            D_an_3 = np.concatenate([np.zeros((t.shape[0], 2, 3)), np.zeros((t.shape[0], 2, 3)), Dse_mat], axis = 2)
            D_analytical = np.concatenate([D_an_1, D_an_2, D_an_3], axis = 1)   # shape = (90,8,8)
            D_analytical_exp = D_analytical[:,np.newaxis, :, :]                 # shape = (90, 1, 8, 8)
            eps_exp = eps[:, np.newaxis, : ,:]                                   # shape = (90, 1, 8, 1)

            if calc_meth == 'all':
                sig_analytical = np.einsum('ijkl,mjkn->imkn', D_analytical_exp, eps_exp)        # shape = (90,90,8,1)
                sig_analytical = sig_analytical.reshape(-1, 8, 1)                               # shape = (8100,8,1)
            elif calc_meth == 'single': 
                sig_analytical = np.zeros((t.shape[0], 8, 1))
                # for i in range(t.shape[0]):
                #     sig_analytical[i,:,:] = np.matmul(D_analytical_exp[i,:,:,:], eps_exp[i, :, :, :])
                #     if i%1000 == 0:
                #         print('sig calculated for '+str(i)+'samples')
                sig_analytical = np.einsum('ijkl, ijkm->ijkm', D_analytical_exp, eps_exp)           # shape = (90,1,8,1)
                sig_analytical = sig_analytical.reshape(t.shape[0], 8, 1)                           # shape = (90,8,1)

        elif discrete == True:
            raise RuntimeError('Method not in use, could lead to errors')
            # 0 - Overall definitions
            nl = num_layers
            S_const = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
            S = np.tile(S_const, (t.shape[0], 1 ,1))                                                # shape = (90, 5, 8)
            G = (2*self.E) / (4 * (1 + self.nu))
            
            # 1 - Calculate e_layer
            e = np.zeros((t.shape[0], nl, 5, 1))
            for l in range(nl):
                z = -t/2+(2*l+1)*t/(2 * nl)                                                         # shape = (90, 1, 1)                     
                S[:,0,3] = -z[:,0, 0]                                                               # shape = (90, )
                S[:,1,4] = -z[:,0, 0]
                S[:,2,5] = -z[:,0, 0]
                # e_klij = np.matmul(S,eps)                                                         # shape eps = (90, 8, 1)
                e_klij = np.einsum('ijk,ikl->ijl', S, eps)                                          # shape = (90, 5, 1)
                e_klij[e_klij == 0] = 10**-13 
                e[:,l,:,:] = e_klij                                                                 # shape = (90, 20, 5, 1)
            
            # 2 - Calculate sig_layer
            s = np.zeros((t.shape[0], nl, 5, 1))
            for l in range(nl):
                sx_x = self.E/(1-self.nu**2)*e[:, l, 0, :]                                                    # shape = (90, 0, 0, 1)
                sx_y = self.E / (1 - self.nu ** 2) * self.nu * e[:, l, 1, :]
                sy_x = self.E/(1-self.nu**2)*self.nu*e[:, l, 0, :]
                sy_y = self.E / (1 - self.nu ** 2) * e[:, l, 1, :]
                txy_xy = self.E/(1-self.nu**2)*(1-self.nu)/2*e[:, l, 2, :]

                sx_xy = np.zeros_like(sx_x)
                sy_xy = np.zeros_like(sx_x)
                txy_x = np.zeros_like(sx_x)
                txy_y = np.zeros_like(sx_x)
                sx = sx_x + sx_y + sx_xy
                sy = sy_x + sy_y + sy_xy
                txy = txy_x + txy_y + txy_xy
                txz = 5/6*G*e[:, l, 3, :]
                tyz = 5/6*G*e[:, l, 4, :]
                s[:,l,:,:] = np.concatenate([sx, sy, txy, txz, tyz], axis = 1).reshape((-1,5,1))      # s shape = (90, 20, 5, 1)
            
            # 3 - Calculate sig
            sig_analytical = np.zeros((t.shape[0], 8, 1))
            sh_kij = np.zeros((t.shape[0], 8, 1))                                                     # shape = (90, 8, 1)
            for l in range(nl):
                z = -t/2+(2*l+1)*t/(2 * nl)
                S[:,0,3] = -z[:,0, 0]                           
                S[:,1,4] = -z[:,0, 0]
                S[:,2,5] = -z[:,0, 0]
                S_t = np.transpose(S, (0, 2, 1))
                sh_kij_new = np.einsum('ijk,ikl->ijl',S_t , s[:, l, :, :]) * t / nl
                sh_kij = sh_kij + sh_kij_new
            sig_analytical = sh_kij

            # 4 - Calculate D
            nl = num_layers
            Dmh = np.zeros((t.shape[0], 3, 3))
            Dbh = np.zeros((t.shape[0], 3, 3))
            Dmbh = np.zeros((t.shape[0], 3, 3))
            Dsh = np.zeros((t.shape[0], 2, 2))
            
            for l in range(nl):
                t_i = t[0,0,0]
                z = -t_i/2 +(2*l+1)*t_i/(2*nl)
                Dmh_l = D_p
                Dmbh_l = -z*D_p
                Dbh_l = z*z*D_p
                Dsh_l = Dse_mat
                Dmh = Dmh + Dmh_l * t_i / nl
                Dbh = Dbh + Dbh_l * t_i / nl
                Dmbh = Dmbh + Dmbh_l * t_i / nl
                Dsh = Dsh + Dsh_l * t_i / nl
            De_1 = np.concatenate([Dmh, Dmbh, np.zeros((t.shape[0], 3, 2))], axis = 2)
            De_2 = np.concatenate([Dbh, Dmbh, np.zeros((t.shape[0], 3, 2))], axis = 2)
            De_3 = np.concatenate([np.zeros((t.shape[0], 2, 3)), np.zeros((t.shape[0], 2, 3)), Dsh], axis = 2)
            D_analytical = np.concatenate([De_1, De_2, De_3], axis = 1)                  # shape = (90,8,8)
            D_analytical_exp = D_analytical[:,np.newaxis, :, :]

        elif discrete == 'andreas':
            if mat == 1:
                t1 = t.reshape(-1,1,1)
                t2 = np.zeros_like(t).reshape(-1,1,1)
                other = None
                mat_dict = None
            elif mat == 3:
                t1 = t[:,0].reshape(-1,1,1)
                t2 = np.zeros_like(t1).reshape(-1,1,1)
                other = t[:,1:].reshape(-1,t.shape[1]-1,1)
                mat_dict = self.mat_dict
            elif mat == 10: 
                t1 = t[:,0].reshape(-1,1,1)
                t2 = t[:,1].reshape(-1,1,1)
                other = None
                mat_dict = None
            
            if t.shape[0] < int(1e6+1):
                samplerutils = SamplerUtils(t1, t2, nl=num_layers, mat=mat, nel=t.shape[0], E1=self.E1, nu1=self.nu1, 
                                            E2=self.E2, nu2=self.nu2, other = other, mat_dict = mat_dict)
                eh = eps.reshape((-1, 1, 1, 8))                                                 # shape = (90, 1, 1, 8)
                e0 = np.zeros((t.shape[0], num_layers, 1, 1, 5), dtype=np.float32)              # shape = (90, 20, 1, 1, 5)
                [e,ex,ey,gxy,e1,e3,th] = samplerutils.find_e(e0,eh,1)
                print('Calculated layer strains e')

                s = samplerutils.find_s(e,1, count = [0,0,0], rho_sublayer = rho_sublayer)       # shape = (90, 20, 1, 1, 3)
                print('Calculated layer stresses s')
                sh = samplerutils.find_sh(s, 1)
                print('Calculated generalised stresses sh')
                sig_analytical = sh.reshape(-1, 8, 1)                                           # shape = (90, 8, 1)
                
                # 4 - Calculate D
                D_analytical = samplerutils.get_dh(s)                                           # shape = (90, 8, 8)
                print('Calculated stiffness matrix D')

            else:
                raise RuntimeWarning('Old implementation of batching, please dont use anymore. Please ensure a maximum batch size of 1e6') 
                # batch-wise calculation for large amounts of data.
                run = wandb.init(project = 'Sampling_RC_Data')
                sig_analytical = np.zeros((t.shape[0], 8, 1))
                D_analytical = np.zeros((t.shape[0], 8, 8))
                n_batches = 3
                batch_size = int(t.shape[0] / n_batches)
                print(f'Starting batchwise calculation with {n_batches} batches, with {batch_size}, {batch_size}, {t.shape[0]-2*batch_size} elements.')
                ######################################################################
                # Batch 1: n = batch_size
                ######################################################################
                
                samplerutils = SamplerUtils(t1, t2, nl=num_layers, mat=mat, nel=batch_size, E1=self.E1, nu1=self.nu1, 
                                            E2=self.E2, nu2=self.nu2, other = other, mat_dict = mat_dict)
                eh = eps[:batch_size, :].reshape((-1, 1, 1, 8))                                 # shape = (bs, 1, 1, 8)
                e0 = np.zeros((t.shape[0], num_layers, 1, 1, 5), dtype=np.float32)              # shape = (bs, 20, 1, 1, 5)
                [e,ex,ey,gxy,e1,e3,th] = samplerutils.find_e(e0,eh,1)
                print('Calculated layer strains e')

                s = samplerutils.find_s(e,1, count = [0,0,0], rho_sublayer = rho_sublayer)       # shape = (bs, 20, 1, 1, 3)
                print('Calculated layer stresses s')
                sh = samplerutils.find_sh(s, 1)
                print('Calculated generalised stresses sh')
                sig_analytical_1 = sh.reshape(-1, 8, 1)                                           # shape = (bs, 8, 1)
                
                # 4 - Calculate D
                D_analytical_1 = samplerutils.get_dh(s)                                           # shape = (bs, 8, 8)
                print('Calculated stiffness matrix D')
                
                sig_analytical[:batch_size, :] = sig_analytical_1
                D_analytical[:batch_size, :, :] = D_analytical_1

                print(f'Calculations for batch 1/{n_batches} with {batch_size} samples completed.')
                ######################################################################
                # Batch 2: n = batch_size
                ######################################################################

                samplerutils = SamplerUtils(t1, t2, nl=num_layers, mat=mat, nel=batch_size, E1=self.E1, nu1=self.nu1, 
                                            E2=self.E2, nu2=self.nu2, other = other, mat_dict = mat_dict)
                eh = eps[batch_size:2*batch_size, :].reshape((-1, 1, 1, 8))                                                 # shape = (bs, 1, 1, 8)
                e0 = np.zeros((t.shape[0], num_layers, 1, 1, 5), dtype=np.float32)              # shape = (bs, 20, 1, 1, 5)
                [e,ex,ey,gxy,e1,e3,th] = samplerutils.find_e(e0,eh,1)
                print('Calculated layer strains e')

                s = samplerutils.find_s(e,1, count = [0,0,0], rho_sublayer = rho_sublayer)       # shape = (bs, 20, 1, 1, 3)
                print('Calculated layer stresses s')
                sh = samplerutils.find_sh(s, 1)
                print('Calculated generalised stresses sh')
                sig_analytical_2 = sh.reshape(-1, 8, 1)                                           # shape = (bs, 8, 1)
                
                # 4 - Calculate D
                D_analytical_2 = samplerutils.get_dh(s)                                           # shape = (bs, 8, 8)
                print('Calculated stiffness matrix D')
                
                sig_analytical[batch_size:2*batch_size, :,:] = sig_analytical_2
                D_analytical[batch_size:2*batch_size, :, :] = D_analytical_2

                print(f'Calculations for batch 2/{n_batches} with {batch_size} samples completed.')
                ######################################################################
                # Batch 3: n = n_tot - batch_size
                ######################################################################

                samplerutils = SamplerUtils(t1, t2, nl=num_layers, mat=mat, nel=t.shape[0]-2*batch_size, E1=self.E1, nu1=self.nu1, 
                                            E2=self.E2, nu2=self.nu2, other = other, mat_dict = mat_dict)
                eh = eps[2*batch_size:,:].reshape((-1, 1, 1, 8))                                                 # shape = (bs, 1, 1, 8)
                e0 = np.zeros((t.shape[0], num_layers, 1, 1, 5), dtype=np.float32)              # shape = (bs, 20, 1, 1, 5)
                [e,ex,ey,gxy,e1,e3,th] = samplerutils.find_e(e0,eh,1)
                print('Calculated layer strains e')

                s = samplerutils.find_s(e,1, count = [0,0,0], rho_sublayer = rho_sublayer)       # shape = (bs, 20, 1, 1, 3)
                print('Calculated layer stresses s')
                sh = samplerutils.find_sh(s, 1)
                print('Calculated generalised stresses sh')
                sig_analytical_3 = sh.reshape(-1, 8, 1)                                           # shape = (bs, 8, 1)
                
                # 4 - Calculate D
                D_analytical_3 = samplerutils.get_dh(s)                                           # shape = (bs, 8, 8)
                print('Calculated stiffness matrix D')
                
                sig_analytical[2*batch_size:, :,:] = sig_analytical_3
                D_analytical[2*batch_size:, :, :] = D_analytical_3

                print(f'Calculations for batch 3/{n_batches} with {t.shape[0]-2*batch_size} samples completed.')
                wandb.finish()



        mat_analytical = {
            'D_a': D_analytical,
            'sig_a': sig_analytical
        }
        return mat_analytical
        

    def sample(self, min_, max_, t1, t2 = None, num_layer = None, num_samples = 90, rho=None, CC=None, twodim = False, uniform = False):
        """
        min_    (list)      minimum boundaries
        max_    (list)      maximum boundaries
        t1      (list)      thickness values
        t2      (list)      thickness values (for interlayer, in glass)
        num_layer  (int)    amount of layers
        num_samples(int)    amount of samples
        rho     (list)      values for reinforcement ratio
        CC      (list)      values for concrete class
        twodim  (bool)      if True: only samples 3 elements: eps_x, eps_y and eps_xy and does not vary t, rho, CC but sets them to one fixed value.
        uniform (bool)      if True: samples data according to uniform distribution insted of LHS
        
        """
        par_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y']
        if t2 != None:
            # glass sampling with LHS
            par_glass_add = ['t1', 't2', 'nl']
            par_names.extend(par_glass_add)

            # extract min, max and range from given parameter values
            min_t1, min_t2, max_t1, max_t2, min_num, max_num = min(t1), min(t2), max(t1), max(t2), min(num_layer), max(num_layer)
            round_to_t1, round_to_t2, round_to_num = (max_t1-min_t1)/(len(t1)-1), (max_t2-min_t2)/(len(t2)-1), (max_num-min_num)/(len(num_layer)-1), 
            min_glas = [min_t1, min_t2, min_num]
            max_glas = [max_t1, max_t2, max_num]
            round_to_glas = np.array([round_to_t1, round_to_t2, round_to_num])
            min_.extend(min_glas)
            max_.extend(max_glas)
        if rho != None and not twodim: 
            if len(rho) > 1: 
                raise UserWarning('Different rho_x and rho_y are not implemented for non-twodim cases.')
            # RC sampling with LHS
            par_rc_add = ['t', 'rho', 'CC']
            par_names.extend(par_rc_add)

            # extract min, max and range from given parameter values
            min_t, max_t, min_rho, max_rho, min_CC, max_CC = min(t1), max(t1), min(rho), max(rho), min(CC), max(CC)
            if len(CC) != 1:
                round_to_t, round_to_rho, round_to_CC = (max_t-min_t)/(len(t1)-1), (max_rho-min_rho)/(len(rho)-1), (max_CC-min_CC)/(len(CC)-1)
            else:
                round_to_t, round_to_rho, round_to_CC = (max_t-min_t)/(len(t1)-1), (max_rho-min_rho)/(len(rho)-1), 1
            min_rc = [min_t, min_rho, min_CC]
            max_rc = [max_t, max_rho, max_CC]
            round_to_rc = np.array([round_to_t, round_to_rho, round_to_CC])
            min_.extend(min_rc)
            max_.extend(max_rc)

        if twodim: 
            par_names = ['eps_x', 'eps_y', 'eps_xy']

        samples = num_samples
        criterion = 'c'
        if uniform:
            uniform_sampler = samplers(par_names, min_, max_, samples, criterion)
            data = uniform_sampler.uniform()
        else:
            lhs_sampler = samplers(par_names, min_, max_, samples, criterion) 
            data = lhs_sampler.lhs()

        if t2 == None and rho == None:
            # for sampling lin.el. data 
            t1_samples = np.random.choice(t1, size=samples)
            data['t1'] = t1_samples
        if twodim: 
            # for sampling twodimensional data with just ONE geometrical set of parameters.
            data['chi_x'] = np.zeros((num_samples,1))
            data['chi_y'] = np.zeros((num_samples,1))
            data['chi_xy'] = np.zeros((num_samples,1))
            data['gam_x'] = np.zeros((num_samples,1))
            data['gam_y'] = np.zeros((num_samples,1))
            data['t1'] = t1[0]*np.ones((num_samples,1))
            data['rho_x'] = rho[0]*np.ones((num_samples,1))
            data['rho_y'] = rho[1]*np.ones((num_samples,1))
            data['CC'] = CC[0]*np.ones((num_samples,1))

        np_data = data.to_numpy(dtype=float)
        np_data.reshape(-1, np_data.shape[1],1)
        np_data_clean = np_data

        if t2 != None: 
            # round parameters for glass such that they correspond to given ranges
            np_data_clean[:,8] = np.multiply(np.round(np_data[:,8]/round_to_glas[0]),round_to_glas[0])
            np_data_clean[:,9] = np.multiply(np.round(np_data[:,9]/round_to_glas[1]),round_to_glas[1])

            # just for the amount of layers:
            targets = np.array(num_layer)
            np_data_clean[:,10] = np.array([targets[np.abs(targets - value).argmin()] for value in np_data[:,10].astype(int)]).astype(int)
        
        if rho != None and not twodim:
            # round parameters for RC such that they correspond to given ranges
            np_data_clean[:,8] = np.multiply(np.round(np_data[:,8]/round_to_rc[0]),round_to_rc[0])
            np_data_clean[:,9] = np.multiply(np.round(np_data[:,9]/round_to_rc[1]),round_to_rc[1])
            # if len(CC) !=1:
            np_data_clean[:,10] = np.multiply(np.round(np_data[:,10]/round_to_rc[2]),round_to_rc[2])
            # else: 
            #     np_data_clean[:,10] = np_data[:,10]

        return np_data_clean

    def extend_material_parameters(self, t_sampled):
        geom_size = t_sampled.shape[1]
        t_extended = np.zeros((t_sampled.shape[0], geom_size+7))
        t_extended[:,0:geom_size] = np.array(t_sampled)
        for i in range(t_sampled.shape[0]): 
            index = int(np.where(np.array(self.mat_dict['CC']) == np.array(t_sampled)[i,geom_size-1])[0])
            t_extended[i,geom_size] = self.mat_dict['Ec'][index]
            t_extended[i,geom_size+1] = self.mat_dict['tb0'][index]
            t_extended[i,geom_size+2] = self.mat_dict['tb1'][index]
            t_extended[i,geom_size+3] = self.mat_dict['ect'][index]
            t_extended[i,geom_size+4] = self.mat_dict['ec0'][index]
            t_extended[i,geom_size+5] = self.mat_dict['fcp'][index]
            t_extended[i,geom_size+6] = self.mat_dict['fct'][index]
        return t_extended

class samplers:
    def __init__(self, parnames, min, max, samples, criterion):
        self.parnames = parnames
        self.min = min
        self.max = max
        self.samples = samples
        self.criterion = criterion

    def lhs(self):
        """
        Returns LHS samples.

        :param parnames: List of parameter names
        :type parnames: list(str)
        :param bounds: List of lower/upper bounds,
                        must be of the same length as par_names
        :type bounds: list(tuple(float, float))
        :param int samples: Number of samples
        :param str criterion: A string that tells lhs how to sample the
                                points. See docs for pyDOE.lhs().
        :return: DataFrame
        """
        
        bounds = np.vstack((self.min, self.max))
        bounds = bounds.T
        

        lhs = doe.lhs(len(self.parnames), samples=self.samples, criterion=self.criterion)
        par_vals = {}
        for par, i in zip(self.parnames, range(len(self.parnames))):
            par_min = bounds[i][0]
            par_max = bounds[i][1]
            par_vals[par] = np.array(lhs[:, i]) * (par_max - par_min) + par_min

        # Convert dict(str: np.ndarray) to pd.DataFrame
        par_df = pd.DataFrame(columns=self.parnames, index=np.arange(int(self.samples)))
        for i in range(self.samples):
            for p in self.parnames:
                par_df.loc[i, p] = par_vals[p][i]

        return par_df
    

    def uniform(self):
        n_i = int(np.round((self.samples)**(1/3),0))
        
        x = np.linspace(self.min[0], self.max[0], n_i)
        y = np.linspace(self.min[1], self.max[1], n_i)
        z = np.linspace(self.min[2], self.max[2], n_i)

        X,Y,Z = np.meshgrid(x,y,z,indexing = 'ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        par_df = pd.DataFrame(points, columns=self.parnames)
        
        return par_df