global u
"""-------------------------------------- Import General Settings and Tools------------------------------------------"""
import os
from math import *
import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve
import sys
import pickle



from dict_CC import dict_CC
from Standard import e_principal,s_c3,s_sc,f_cs
from Stresses_mixreinf import stress
from constitutive_laws import ConstitutiveLaws
from deployment_prediction import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



"""---------------------------------------------- Time Measurement---------------------------------------------------"""
class gettime:
    def _updatetime(self,update=True,delta_t=0):
        if update:
            self.time_spent += delta_t
            # print(self.time_spent)
        else:
            self.time_spent = 0
time_stress = gettime()
time_stress._updatetime(update=False)
time_strain = gettime()
time_strain._updatetime(update=False)
time_K = gettime()
time_K._updatetime(update=False)
time_B = gettime()
time_B._updatetime(update=False)
time_Kinv = gettime()
time_Kinv._updatetime(update=False)
time_sh = gettime()
time_sh._updatetime(update=False)
time_eh = gettime()
time_eh._updatetime(update=False)

"""----------------------------------------- Save old Stiffness Matrix ----------------------------------------------"""
class ki:
    def _updatek(self,update=False,knew=0):
        if update:
            self.itstep += 1
            self.k = knew
        else:
            self.itstep = 0
            self.k = knew
kold = ki()
kold._updatek()


""" ----------------------------------------- Import Input and Meshing ----------------------------------------------"""
# from Mesh_gmsh import order,gauss_order,BC,ELS,GEOMK,MATK,COORD,copln,it_type
# from Mesh_gmsh_vb import input_definition

class fem_func():
    def __init__(self, MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, model_path):
        self.MATK = MATK
        self.NODESG = NODESG
        self.ELS = ELS
        self.COORD = COORD
        self.GEOMA = GEOMA
        self.GEOMK = GEOMK
        self.MASK = MASK
        self.na = na
        self.BC = BC
        self.gauss_order = gauss_order
        self.it_type = it_type
        self.Load_el = Load_el
        self.Load_n = Load_n
        self.copln = copln
        self.ELEMENTS = ELS[0]
        self.model_path = model_path

    # MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln = input_definition(mat)

    # Load_el,Load_n

    def check_range_NN(self, vec:np.array, id:str, predict = 'sig', cmk=3, return_key=False):
        '''
        Checks whether the given vector vec lies within range of training data. If not, prints error message (does not stop programme)
        vec         (np.array)      vector to be checked for range
                                    expected shapes: 'sig' (1,8), 'eps-t' (1,8), 'D' (1,8,8)
        id          (str)           can be 'sig', 'eps-t' or 'D'
        return_key  (bool)          If True, return the model key ('sig_I'/'sig_II'/'sig_III'/'D')
                                    instead of self.model_path['model'][key]. Useful when the
                                    caller holds a `loaded_models` dict keyed identically and
                                    wants to look up the preloaded model directly.
        '''
        if id == 'eps-t' and predict == 'sig':
            # define strain ranges of models (acc. to 241208_DetailedDocumentation)
            if cmk == 1:
                raise Warning('Please check the boundaries. They are currently set for steel')
                # steel
                min_I = np.array([-6e-6, -6e-6, -6e-6, -20e-5, -20e-5, -20e-5, -4e-5, -4e-5])
                max_I = np.array([ 6e-6,  6e-6,  6e-6,  20e-5,  20e-5,  20e-5,  4e-5,  4e-5])
                min_II = np.array([-2e-5, -2e-5, -2e-5, -0.15e-3, -0.15e-3, -0.15e-3, -4e-4, -4e-4])
                max_II = np.array([ 2e-5,  2e-5,  2e-5,  0.15e-3,  0.15e-3,  0.15e-3,  4e-4,  4e-4])
                min_III = np.array([-1.12e-3, -1.12e-3, -1.12e-3, -0.15e-3, -0.15e-3, -0.15e-3, -1.67e-3, -1.67e-3])
                max_III = np.array([ 1.12e-3,  1.12e-3,  1.12e-3,  0.15e-3,  0.15e-3,  0.15e-3,  1.67e-3,  1.67e-3])
                # RC concrete (lin.el.)
                # min_I = np.array([-10e-6, -10e-6, -10e-6, -5e-6, -5e-6, -5e-5, -30e-6, -30e-6])
                # max_I = np.array([ 5e-6,   5e-6,    5e-6,  5e-6,  5e-6,  5e-6,  30e-6,  30e-6])
                # min_II = np.array([-0.1e-3,   -0.1e-3, -0.1e-3, -0.005e-3, -0.005e-3, -0.01e-3, -0.1e-3, -0.1e-3])
                # max_II = np.array([0.005e-3, 0.005e-3,  0.1e-3,  0.005e-3,  0.005e-3,  0.01e-3,  0.1e-3,  0.1e-3])
                # min_III = np.array([-0.37e-3, -0.37e-3, -0.47e-3, -0.005e-3, -0.005e-3, -0.01e-3, -0.47e-3, -0.47e-3])
                # max_III = np.array([ 0.09e-3,  0.09e-3,  0.47e-3,  0.005e-3,  0.005e-3,  0.01e-3,  0.47e-3,  0.47e-3])
            if cmk == 3: 
                # RC concrete (nonlin)
                min_I = np.array([-10e-6, -10e-6, -10e-6, -5e-6, -5e-6, -5e-5, -30e-6, -30e-6])
                max_I = np.array([ 5e-6,   5e-6,    5e-6,  5e-6,  5e-6,  5e-6,  30e-6,  30e-6])
                min_II = np.array([-0.1e-3,   -0.1e-3, -0.1e-3, -0.005e-3, -0.005e-3, -0.01e-3, -0.1e-3, -0.1e-3])
                max_II = np.array([0.005e-3, 0.005e-3,  0.1e-3,  0.005e-3,  0.005e-3,  0.01e-3,  0.1e-3,  0.1e-3])
                min_III = np.array([-3e-3, -3e-3, -4e-3, -0.03e-3, -0.03e-3, -0.04e-3, -5e-3, -5e-3])
                max_III = np.array([ 5e-3,  5e-3,  4e-3,  0.05e-3,  0.05e-3,  0.04e-3,  5e-3,  5e-3])



            # Choose the correct model depending on the strain range:
            v = vec[:,0:8]
            mask_I = (v >= min_I) & (v <= max_I)
            mask_II = (v >= min_II) & (v <= max_II)
            mask_III = (v >= min_III) & (v <= max_III)
            
            if mask_I.all():
                key = 'sig_I'
                # print('Chosen model is model I.')
                id_choice = '_I'
            elif mask_II.all():
                key = 'sig_II'
                # print('Chosen model is model II.')
                id_choice = '_II'
            elif mask_III.all():
                key = 'sig_III'
                # print('Chosen model is model III.')
                id_choice = '_III'
            else:
                key = 'sig_III'
                # print('Some values are outside the strain boundaries of the sampled data. Chosen model is still model III.')
                id_choice = '_III'
            chosen_model_path = self.model_path['model'][key]

        elif id == 'eps-t' and predict == 'D':
            key = 'D'
            chosen_model_path = self.model_path['model'][key]


        else:
            key = None
            chosen_model_path = None

        if return_key:
            return key
        return chosen_model_path
    

    def check_small_eps(self, D, input, Delta_min = 1e-7):
        '''
        checks if small epsilon are in dataset. If so, sets stiffness to lin.el. stiffness.
        D               (np.array)          stiffness matrix, expected shape: (k,go,go,8,8)
        input           (np.array)          input to the ML model, expected shape: (k,go,go,8)  
        Delta_min       (float)             minimum value of epsilon / chi. 
                                            Below this, it is counted as small and D will be changed to lin.el. value
        '''
        numel = D.shape[0]
        go = D.shape[1]
        eps_corr = np.zeros((numel,go,go,6,6))
        chi_corr = np.zeros((numel,go,go,6,6))
        D_new = D

        for i in range(3):
            eps_corr[:,:,:,i,0:3] = input[:,:,:,0:3]
            eps_corr[:,:,:,i+3,0:3] = input[:,:,:,0:3]
            eps_corr[:,:,:,i,3:6] = input[:,:,:,0:3]
            eps_corr[:,:,:,i+3,3:6] = (Delta_min/10)*np.ones((numel,go,go,3))

            chi_corr[:,:,:,i,0:3] = (Delta_min/10)*np.ones((numel,go,go,3))
            chi_corr[:,:,:,i+3,0:3] = input[:,:,:,3:6]
            chi_corr[:,:,:,i,3:6] = input[:,:,:,3:6]
            chi_corr[:,:,:,i+3,3:6] = input[:,:,:,3:6]

        for i in range(6):
            for j in range(6):
                if (eps_corr[:,:,:,i,j].all() and chi_corr[:,:,:,i,j].all()) < Delta_min:
                    # eps and sig are set to zero for the call of dh_kij because they are not required when calculating lin.el.
                    nlk = self.GEOMK["nlk"][0]
                    Dm_adm, Dmb_adm, Db_adm, _ = self.dh_kij(e_kij=np.zeros((nlk,8)), s_kij=np.zeros((nlk,8)), k=0, i=None, j=None, cm_k = 1)
                    if i <3 and j <3: 
                        D_adm = Dm_adm[i, j]
                    elif i >= 3 and j >= 3:
                        D_adm = Db_adm[i - 3, j - 3]
                    elif i >= 3 and j < 3:
                        D_adm = Dmb_adm[i - 3, j]
                    else:  # i < 3 and j >= 3
                        D_adm = Dmb_adm[i, j - 3]

                    if D[:, :, :, i, j].all() > D_adm:
                        D_new[:, :, :, i, j] = D_adm * np.ones((numel, go, go))
                        print(f'Adjusted D_{i}{j} because eps and chi were smaller than Delta_min = {Delta_min}.')
                    else: 
                        # leave D_ij as is if it is smaller than D,ij,adm,max
                        print(f'Did not adjust D_{i}{j} as it is small enough.')
                else:
                    print(f'eps and chi for {i},{j} are large enough.')
                    # leave D_ij as it is if eps and chi are larger than threshold.
        
        # D_s is not checked, as the predictions for D_s were already ok without adjustments.


        return D_new

    
    def filter_small_stiffness(self, D_NN, k, scenario):
        '''
        Checks shear stiffness and removes too small values of shear stiffness to be replaced with G/10
        
        :param D_NN: (3x3) matrix (D_m)
        :param k: Element number
        :param scenario: Boundary conditions
        '''
        if D_NN.shape != (3,3):
            raise UserWarning('If D_NN does not have shape (3,3), this function will lead to erroneous results. Please adjust function or redo calculation.')

        if scenario not in [8, 9, 109, 110, 111, 112, 201, 202]:
            # only apply shear stiffness reduction for base cases and combination cases which don't include any shear strains.
            pass
        else:
            ff = self.MATK["Ec"][k]/10
            D_p_min = np.array([[ff, 0, 0], [0, ff, 0], [0, 0, ff/2]])
            D_min = self.GEOMK["t"][k]*D_p_min

            if D_NN[2,2] < D_min[2,2]:
                D_NN[2,2] = D_min[2,2]

        return D_NN



    """------------------------------------------------ Integration------------------------------------------------------"""

    def gauss_points(self,nn,go):
        """ ------------------------------ Gauss Points for Quadrilaterals and Triangles -----------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - nn: Number of nodes
            - go: Gauss order
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - xi: Gauss points in one dimension
            - w: Gauss weights in one dimension
        -----------------------------------------------------------------------------------------------------------------"""
        # 0 Gauss Points for Quadrilaterals
        if nn == 4:
            if go == 1:
                xi = np.array([0])
                w = np.array([2])
            elif go == 2:
                xi = np.array([-sqrt(1/3),sqrt(1/3)])
                w = np.array([1,1])
            elif go == 3:
                xi = np.array([-0.774597, 0, 0.774597])
                w = np.array([5/9,8/9,5/9])
            elif go == 4:
                xi = np.array([-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053])
                w = np.array([0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538])
            elif go == 5:
                xi = np.array([-0.9061798459386640,-0.5384693101056831,0,0.5384693101056831,0.9061798459386640])
                w = np.array([0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891])
            elif go == 6:
                xi1 = -0.9324695142031521
                xi2 = -0.6612093864662645
                xi3 = -0.2386191860831969
                xi4 = 0.2386191860831969
                xi5 = 0.661209386466264
                xi6 = 0.9324695142031521
                a1 = 0.1713244923791704
                a2 = 0.3607615730481387
                a3 = 0.4679139345726913
                a4 = a3
                a5 = a2
                a6 = a1
                xi = np.array([xi1,xi2,xi3,xi4,xi5,xi6])
                w = np.array([a1,a2,a3,a4,a5,a6])
            elif go == 30:
                w = [0.102852653,0.102852653,0.10176239,0.10176239,0.099593421,0.099593421,0.096368737,0.096368737,0.092122522,0.092122522,
                    0.086899787,0.086899787,0.080755895,0.080755895,0.073755975,0.073755975,0.06597423,0.06597423,0.057493156,0.057493156,
                    0.048402673,0.048402673,0.038799193,0.038799193,0.028784708,0.028784708,0.018466468,0.018466468,0.007968192,0.007968192]
                xi = [-0.051471843,0.051471843,-0.153869914,0.153869914,-0.254636926,0.254636926,-0.352704726,0.352704726,-0.44703377,0.44703377,
                    -0.536624148,0.536624148,-0.620526183,0.620526183,-0.697850495,0.697850495,-0.767777432,0.767777432,-0.829565762,0.829565762,
                    -0.882560536,0.882560536,-0.926200047,0.926200047,-0.960021865,0.960021865,-0.983668123,0.983668123,-0.996893484,0.996893484]
            elif go == 40:
                w = [0.077505948,
                    0.077505948,
                    0.077039818,
                    0.077039818,
                    0.076110362,
                    0.076110362,
                    0.074723169,
                    0.074723169,
                    0.072886582,
                    0.072886582,
                    0.070611647,
                    0.070611647,
                    0.067912046,
                    0.067912046,
                    0.064804013,
                    0.064804013,
                    0.061306242,
                    0.061306242,
                    0.057439769,
                    0.057439769,
                    0.053227847,
                    0.053227847,
                    0.048695808,
                    0.048695808,
                    0.043870908,
                    0.043870908,
                    0.038782168,
                    0.038782168,
                    0.033460195,
                    0.033460195,
                    0.027937007,
                    0.027937007,
                    0.022245849,
                    0.022245849,
                    0.016421058,
                    0.016421058,
                    0.010498285,
                    0.010498285,
                    0.004521277,
                    0.004521277]
                xi = [-0.038772418,
                0.038772418,
                -0.116084071,
                0.116084071,
                -0.192697581,
                0.192697581,
                -0.268152185,
                0.268152185,
                -0.341994091,
                0.341994091,
                -0.413779204,
                0.413779204,
                -0.483075802,
                0.483075802,
                -0.549467125,
                0.549467125,
                -0.61255389,
                0.61255389,
                -0.671956685,
                0.671956685,
                -0.727318255,
                0.727318255,
                -0.778305651,
                0.778305651,
                -0.824612231,
                0.824612231,
                -0.865959503,
                0.865959503,
                -0.902098807,
                0.902098807,
                -0.932812808,
                0.932812808,
                -0.957916819,
                0.957916819,
                -0.97725995,
                0.97725995,
                -0.990726239,
                0.990726239,
                -0.99823771,
                0.99823771]
            elif go == 50:
                w = [0.062176617,
                    0.062176617,
                    0.061936067,
                    0.061936067,
                    0.0614559,
                    0.0614559,
                    0.060737971,
                    0.060737971,
                    0.059785059,
                    0.059785059,
                    0.05860085,
                    0.05860085,
                    0.057189926,
                    0.057189926,
                    0.055557745,
                    0.055557745,
                    0.053710622,
                    0.053710622,
                    0.051655703,
                    0.051655703,
                    0.049400938,
                    0.049400938,
                    0.046955051,
                    0.046955051,
                    0.044327504,
                    0.044327504,
                    0.041528463,
                    0.041528463,
                    0.038568757,
                    0.038568757,
                    0.035459836,
                    0.035459836,
                    0.032213728,
                    0.032213728,
                    0.028842994,
                    0.028842994,
                    0.025360674,
                    0.025360674,
                    0.021780243,
                    0.021780243,
                    0.018115561,
                    0.018115561,
                    0.014380823,
                    0.014380823,
                    0.010590548,
                    0.010590548,
                    0.006759799,
                    0.006759799,
                    0.002908623,
                    0.002908623]
                xi = [-0.031098338,
                    0.031098338,
                    -0.093174702,
                    0.093174702,
                    -0.15489059,
                    0.15489059,
                    -0.216007237,
                    0.216007237,
                    -0.276288194,
                    0.276288194,
                    -0.335500245,
                    0.335500245,
                    -0.393414312,
                    0.393414312,
                    -0.449806335,
                    0.449806335,
                    -0.504458145,
                    0.504458145,
                    -0.557158305,
                    0.557158305,
                    -0.607702927,
                    0.607702927,
                    -0.655896466,
                    0.655896466,
                    -0.701552469,
                    0.701552469,
                    -0.744494302,
                    0.744494302,
                    -0.784555833,
                    0.784555833,
                    -0.821582071,
                    0.821582071,
                    -0.855429769,
                    0.855429769,
                    -0.88596798,
                    0.88596798,
                    -0.913078557,
                    0.913078557,
                    -0.936656619,
                    0.936656619,
                    -0.956610955,
                    0.956610955,
                    -0.972864385,
                    0.972864385,
                    -0.985354084,
                    0.985354084,
                    -0.994031969,
                    0.994031969,
                    -0.998866404,
                    0.998866404]

        # 1 Gauss Points for Triangles
        elif nn == 3:
            if go == 1:
                xi = np.array([1/3])
                w = np.array([1/sqrt(2)])
            elif go == 2:
                xi = np.array([1/6,2/3])
                w = np.array([1/sqrt(6),1/sqrt(6)])
        return xi,w


    """----------------------------------------------- Material Laws-----------------------------------------------------"""


    def get_et(self, cm_klij,e_klij,s_klij, k, l,i,j):
        """ ------------------------------------- Define Constitutive Matrix -----------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - cm_klij
                - 1: Linear Elasticity
                - 2: CMM-, CMM without tension stiffening (NOT IMPLEMENTED YET)
                - 3: CMM
            - k: Element number
            - l: Layer number
            - [ex_klij, ey_klij, gxy_klij]: Strain state
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -ET: Secant Constitutive Matrix
        -----------------------------------------------------------------------------------------------------------------"""
        s = s_klij
        if self.it_type == 1:
            if cm_klij == 1:
                E = self.MATK["Ec"][k]
                v = self.MATK["vc"][k]
                ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            else:

                ET = np.array([[s[0].sx.imag, s[1].sx.imag, s[2].sx.imag],
                            [s[0].sy.imag, s[1].sy.imag, s[2].sy.imag],
                            [s[0].txy.imag, s[1].txy.imag, s[2].txy.imag]])/0.0000000000000001
        elif self.it_type == 2:
            if cm_klij == 1:
                E = self.MATK["Ec"][k]
                v = self.MATK["vc"][k]
                ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            else:
                # Concrete Secant Stiffness Matrix
                if s[0].e1 < 0:
                    E1 = abs(s[0].sc1 / s[0].e1)
                else:
                    E1 = 10
                if s[0].e3 < 0:
                    E3 = abs(s[0].sc3 / s[0].e3)
                else:
                    E3 = 10
                if abs(s[0].e3 - s[0].e1) > 0:
                    G = max(abs(0.5 * (s[0].sc3 - s[0].sc1) / (s[0].e3 - s[0].e1)), 5)
                else:
                    G = 5
                Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
                s[0].t_mat()
                Ec = np.linalg.inv(s[0].Tsigma) @ Ec13 @ s[0].Tepsilon

                # Steel and CFRP Stiffness Matrix
                if abs(s[0].ex) > 0:
                    Esx = s[0].ssx / s[0].ex
                    Epx = s[0].spx / s[0].ex
                else:
                    Esx = 0
                    Epx = 0
                if abs(s[0].ey) > 0:
                    Esy = s[0].ssy / s[0].ey
                    Epy = s[0].spy / s[0].ey
                else:
                    Esy = 0
                    Epy = 0
                Ds = np.array([[s[0].rhox * Esx, 0, 0], [0, s[0].rhoy * Esy, 0], [0, 0, 0]])
                Dp = np.array([[s[0].rhopx * Epx, 0, 0], [0, s[0].rhopy * Epy, 0], [0, 0, 0]])

                # Assemble
                ET = Ec + Ds + Dp
        return ET
    
    def get_et_vb(self, cm_klij,e_klij,s_klij, k, l,i,j):
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
        if self.it_type == 1:
            if cm_klij == 1 and self.MATK["cm"][k] == 3:
                # E = self.MATK["Ec"][k]/3
                E = self.MATK["Ec"][k]
                # v = 0.2
                v = 0
                ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            elif cm_klij == 1:
                E = self.MATK["Ec"][k]
                v = self.MATK["vc"][k]
                ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            elif cm_klij == 3: 
                ET = np.array([[s[0].sx.imag, s[1].sx.imag, s[2].sx.imag],
                            [s[0].sy.imag, s[1].sy.imag, s[2].sy.imag],
                            [s[0].txy.imag, s[1].txy.imag, s[2].txy.imag]])/0.0000000000000001
            elif cm_klij == 10:
                E_1 = self.MATK["Ec"][k]
                E_2 = self.MATK["Ec2"][k]
                v_1 = self.MATK["vc"][k]
                v_2 = self.MATK["vc2"][k]
                ET_1 = E_1 / (1 - v_1**2) * np.array([[1, v_1, 0], [v_1, 1, 0], [0, 0, 0.5 * (1 - v_1)]])
                ET_2 = E_2 / (1-v_2**2) * np.array([[1, v_2, 0], [v_2, 1, 0], [0, 0, 0.5 * (1 - v_2)]])
                ET = {'ET_1': ET_1,
                      'ET_2': ET_2}
        elif self.it_type == 2:
            if cm_klij == 1:
                E = self.MATK["Ec"][k]
                v = self.MATK["vc"][k]
                ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            elif cm_klij == 3:
                # Concrete Secant Stiffness Matrix
                if s[0].e1 < 0:
                    E1 = abs(s[0].sc1 / s[0].e1)
                else:
                    E1 = 10
                if s[0].e3 < 0:
                    E3 = abs(s[0].sc3 / s[0].e3)
                else:
                    E3 = 10
                if abs(s[0].e3 - s[0].e1) > 0:
                    G = max(abs(0.5 * (s[0].sc3 - s[0].sc1) / (s[0].e3 - s[0].e1)), 5)
                else:
                    G = 5
                Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
                s[0].t_mat()
                Ec = np.linalg.inv(s[0].Tsigma) @ Ec13 @ s[0].Tepsilon

                # Steel and CFRP Stiffness Matrix
                if abs(s[0].ex) > 0:
                    Esx = s[0].ssx / s[0].ex
                    Epx = s[0].spx / s[0].ex
                else:
                    Esx = 0
                    Epx = 0
                if abs(s[0].ey) > 0:
                    Esy = s[0].ssy / s[0].ey
                    Epy = s[0].spy / s[0].ey
                else:
                    Esy = 0
                    Epy = 0
                Ds = np.array([[s[0].rhox * Esx, 0, 0], [0, s[0].rhoy * Esy, 0], [0, 0, 0]])
                Dp = np.array([[s[0].rhopx * Epx, 0, 0], [0, s[0].rhopy * Epy, 0], [0, 0, 0]])

                # Assemble
                ET = Ec + Ds + Dp
            elif cm_klij == 10:
                E_1 = self.MATK["Ec"][k]
                E_2 = self.MATK["Ec2"][k]
                v_1 = self.MATK["vc"][k]
                v_2 = self.MATK["vc2"][k]
                ET_1 = E_1 / (1 - v_1**2) * np.array([[1, v_1, 0], [v_1, 1, 0], [0, 0, 0.5 * (1 - v_1)]])
                ET_2 = E_2 / (1-v_2**2) * np.array([[1, v_2, 0], [v_2, 1, 0], [0, 0, 0.5 * (1 - v_2)]])
                ET = {'ET_1': ET_1,
                      'ET_2': ET_2}
        return ET


    """------------------------------------------------ B_Matrices-------------------------------------------------------"""

    def jacobi(self, k, i, j, go):
        """ -------------------------------------- Get Jacobian of Element  ------------------------------------------------
                ----------------------------------------------- INPUT: --------------------------------------------------
                - k: element number
                - i,j: Gauss point numbers in xi - and eta space
                - go: gauss order
                ---------------------------------------------- OUTPUT: --------------------------------------------------
                - xi,eta: Gauss points
                - J: Jacobian
                - J_inv: Inverse of jacobian
                - J_det: Determinant of jacobian
        -----------------------------------------------------------------------------------------------------------------"""
        # 1 Values of Importance

        # 1.1 Nodes of element k
        e_k = self.ELEMENTS[k, :]
        e_k = e_k[e_k<10**5]

        # 1.2 Area of element k
        a_k = self.GEOMK["ak"][k]

        # 1.3 Local coordinates of nodes of element k
        NODESL = self.COORD["n"][2][a_k]
        v = np.array(NODESL[e_k])

        # -----------------------------------------------------------------------------------------------------------------#
        # 2 Jacobian for quadrilaterals
        # -----------------------------------------------------------------------------------------------------------------#
        order = 1
        if len(e_k) == 4:
            if order == 1:

                # 2.1 Gauss points and weights
                gp, w = self.gauss_points(4,go)
                xi = gp[j]
                eta = gp[i]

                # 2.2 Derivatives of Shape Functions in xi-eta-space
                N1xi = float(- (1 - eta) / 4)
                N2xi = float((1 - eta) / 4)
                N3xi = float((1 + eta) / 4)
                N4xi = float(- (1 + eta) / 4)
                N1eta = float(- (1 - xi) / 4)
                N2eta = float(- (1 + xi) / 4)
                N3eta = float((1 + xi) / 4)
                N4eta = float((1 - xi) / 4)

                # 2.3 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

                # 2.4 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)
        # -----------------------------------------------------------------------------------------------------------------#
        # 3 Jacobian of triangles
        # -----------------------------------------------------------------------------------------------------------------#
        else:
            if order == 1:

                # 3.1 Coordinates of element nodes: delete entry 100'001
                v = np.array(NODESL[self.ELEMENTS[k][0:3]])

                # 3.2 Gauss Points
                gp, w = self.gauss_points(3, go)
                xi = gp[j]
                eta = gp[i]

                # 3.3 Derivatives Shape Functions in xi-eta space
                N1xi = -1
                N2xi = 1
                N3xi = 0
                N1eta = -1
                N2eta = 0
                N3eta = 1

                # 3.4 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

                # 3.5 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

        return xi,eta,J,J_inv,J_det


    def b_kij(self, k, i, j, go,rot=True):
        """ ---------------------------------------- Create B-Matrices  -------------------------------------------------
            ----------------------------------------------- INPUT: ------------------------------------------------------
            - k: element number
            - i,j: Gauss point numbers in xi - and eta space
            - go: gauss order
            - rot: True, if B-Matrices shall be rotated in global coordinate system
            ---------------------------------------------- OUTPUT: ------------------------------------------------------
            - B_m: Membrane strain matrix
            - B_b: Bending strain matrix
            - B_s: Transverse shear strain matrix.
                - For quadrilaterals: locking-free element with prescribed transverse shear strain profile
                - For triangles: locking-free element with reduced integration of shear terms
            - J_det: Determinant of Jacobian
        -----------------------------------------------------------------------------------------------------------------"""

        # 0 Initiate Time Measurement
        start = time.time()

        # 1 Values of Importance

        # 1.1 Nodes of element k
        e_k = self.ELEMENTS[k, :]
        e_k = e_k[e_k<10**5]

        # 1.2 Area of element k
        a_k = self.GEOMK["ak"][k]

        # 1.3 Local coordinates of nodes of element k
        NODESL = self.COORD["n"][2][a_k]
        v = np.array(NODESL[e_k])

        # -----------------------------------------------------------------------------------------------------------------#
        # 2 Integration of quadrilaterals
        # -----------------------------------------------------------------------------------------------------------------#
        order = 1
        if len(e_k) == 4:
            if order == 1:
                # print(go)
                # 2.1 Integration of Bending and Membrane Terms

                # 2.1.1 Gauss points and weights
                gp, w = self.gauss_points(4,go)
                xi = gp[j]
                eta = gp[i]

                # 2.1.2 Shape Functions in xi-eta-space
                N1xe = 1/4*(1-xi)*(1-eta)/1
                N2xe = 1/4*(1+xi)*(1-eta)/1
                N3xe = 1/4*(1+xi)*(1+eta)/1
                N4xe = 1/4*(1-xi)*(1+eta)/1

                # 2.1.3 Derivatives of Shape Functions in xi-eta-space
                N1xi = float(- (1 - eta) / 4)
                N2xi = float((1 - eta) / 4)
                N3xi = float((1 + eta) / 4)
                N4xi = float(- (1 + eta) / 4)
                N1eta = float(- (1 - xi) / 4)
                N2eta = float(- (1 + xi) / 4)
                N3eta = float((1 + xi) / 4)
                N4eta = float((1 - xi) / 4)
                N1xi_eta = np.array([[N1xi], [N1eta]])
                N2xi_eta = np.array([[N2xi], [N2eta]])
                N3xi_eta = np.array([[N3xi], [N3eta]])
                N4xi_eta = np.array([[N4xi], [N4eta]])

                # 2.1.4 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

                # 2.1.5 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

                # 2.1.6 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(J_inv, N1xi_eta)
                N2x_y = np.matmul(J_inv, N2xi_eta)
                N3x_y = np.matmul(J_inv, N3xi_eta)
                N4x_y = np.matmul(J_inv, N4xi_eta)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])
                N4x = float(N4x_y[0])
                N4y = float(N4x_y[1])

                # 2.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
                #       without prescribed shear transverse shear strain (with locking)
                #       - Rotated global coordinates with Tk
                Tk = self.rotLG(k)[0]
                B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0,0, 0, N2x, -N2xe, 0, 0,0, 0, N3x, -N3xe, 0, 0,0, 0, N4x, -N4xe, 0,0],
                                [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])
                B_b = np.array([[0,0,0, N1x,0,0,0, 0, 0, N2x, 0,0,0,0,0, N3x, 0,0, 0,0,0, N4x, 0,0],
                                [0,0,0,0, N1y, 0,0, 0,0,0, N2y, 0,0, 0,0,0, N3y, 0,0, 0,0,0, N4y,0],
                                [0,0,0,N1y, N1x,0,0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0]])
                B_m = np.array([[N1x,0,0,0, 0, 0, N2x,0, 0,0,0,0, N3x,0, 0, 0,0,0, N4x,0, 0,0,0,0],
                                [0, N1y,0, 0, 0,0,0, N2y,0, 0, 0,0,0, N3y,0, 0, 0,0,0, N4y,0,0, 0,0],
                                [N1y, N1x,0, 0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0,0,0,0]])

                # 2.2 Integration of Shear Terms: Assumed Transverse Shear Element
                Js = np.zeros((4,2,2))
                for ijs in range(4):

                    # 2.2.1 Integration points in edge centers
                    xis = np.array([0, 1, 0, -1])[ijs]
                    etas = np.array([-1, 0, 1, 0])[ijs]

                    # 2.2.2 Shape Functions in xi-etas-space
                    N1s = 1 / 4 * (1 - xis) * (1 - etas)/1
                    N2s = 1 / 4 * (1 + xis) * (1 - etas)/1
                    N3s = 1 / 4 * (1 + xis) * (1 + etas)/1
                    N4s = 1 / 4 * (1 - xis) * (1 + etas)/1

                    # 2.2.3 Derivatives of Shape Functions in xis-etas-space
                    N1xis = float(- (1 - etas) / 4)
                    N2xis = float((1 - etas) / 4)
                    N3xis = float((1 + etas) / 4)
                    N4xis = float(- (1 + etas) / 4)
                    N1etas = float(- (1 - xis) / 4)
                    N2etas = float(- (1 + xis) / 4)
                    N3etas = float((1 + xis) / 4)
                    N4etas = float((1 - xis) / 4)
                    N1xis_etas = np.array([[N1xis], [N1etas]])
                    N2xis_etas = np.array([[N2xis], [N2etas]])
                    N3xis_etas = np.array([[N3xis], [N3etas]])
                    N4xis_etas = np.array([[N4xis], [N4etas]])

                    # 2.1.4 Gradient Matrix
                    Grad_Mats = np.array([[N1xis, N2xis, N3xis, N4xis], [N1etas, N2etas, N3etas, N4etas]])

                    # 2.1.5 Jacobian
                    Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                    Js_inv = np.linalg.inv(Js[ijs][:][:])
                    Js_det = np.linalg.det(Js[ijs][:][:])

                    # 2.2.4 Derivatives of Shape Functions in x-y-Space
                    N1x_y = np.matmul(Js_inv, N1xis_etas)
                    N2x_y = np.matmul(Js_inv, N2xis_etas)
                    N3x_y = np.matmul(Js_inv, N3xis_etas)
                    N4x_y = np.matmul(Js_inv, N4xis_etas)
                    N1x = float(N1x_y[0])
                    N1y = float(N1x_y[1])
                    N2x = float(N2x_y[0])
                    N2y = float(N2x_y[1])
                    N3x = float(N3x_y[0])
                    N3y = float(N3x_y[1])
                    N4x = float(N4x_y[0])
                    N4y = float(N4x_y[1])

                    # 2.2.5 Bijs-Matrix
                    B_ijs = np.array([[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0, 0, 0, N4x, -N4s, 0, 0],
                                    [0, 0, N1y, 0, -N1s, 0,0, 0, N2y, 0, -N2s, 0,0, 0, N3y, 0, -N3s, 0,0, 0, N4y, 0, -N4s, 0]
                                    ])

                    # 2.2.6 Assembly to B-bar Matrix
                    if ijs == 0:
                        B_bar = B_ijs
                    else:
                        B_bar = np.append(B_bar,B_ijs,axis=0)

                # 2.2.7 Auxiliary Matrices for prescribed transverse shear strain field

                Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0, 0, 0, 0, 0],
                                [Js[0][1, 0], Js[0][1, 1], 0, 0, 0, 0, 0, 0],
                                [0, 0, Js[1][0, 0], Js[1][0, 1], 0, 0, 0, 0],
                                [0, 0, Js[1][1, 0], Js[1][1, 1], 0, 0, 0, 0],
                                [0, 0, 0, 0, Js[2][0, 0], Js[2][0, 1], 0, 0],
                                [0, 0, 0, 0, Js[2][1, 0], Js[2][1, 1], 0, 0],
                                [0, 0, 0, 0, 0, 0, Js[3][0, 0], Js[3][0, 1]],
                                [0, 0, 0, 0, 0, 0, Js[3][1, 0], Js[3][1, 1]],
                                ])
                Tmat = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1]])
                Pmat = np.array([[1, -1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]])
                Amat = np.array([[1, gp[i], 0, 0], [0, 0, 1, gp[j]]])

                # 2.2.8 B_s-Matrix
                B_sats = np.matmul(J_inv,np.matmul(Amat,np.matmul(np.linalg.inv(Pmat),np.matmul(Tmat,np.matmul(Cmat,B_bar)))))

                # 2.3 Integration of Shear Terms: reduced integration
                # 2.3.1 Gauss points and weights
                # gp, w = gauss_points(4, 1)
                # xi = gp[0]
                # eta = gp[0]
                #
                # # 2.3.2 Shape Functions in xi-eta-space
                # N1xe = 1 / 4 * (1 - xi) * (1 - eta)
                # N2xe = 1 / 4 * (1 + xi) * (1 - eta)
                # N3xe = 1 / 4 * (1 + xi) * (1 + eta)
                # N4xe = 1 / 4 * (1 - xi) * (1 + eta)
                #
                # # 2.3.3 Derivatives of Shape Functions in xi-eta-space
                # N1xi = float(- (1 - eta) / 4)
                # N2xi = float((1 - eta) / 4)
                # N3xi = float((1 + eta) / 4)
                # N4xi = float(- (1 + eta) / 4)
                # N1eta = float(- (1 - xi) / 4)
                # N2eta = float(- (1 + xi) / 4)
                # N3eta = float((1 + xi) / 4)
                # N4eta = float((1 - xi) / 4)
                # N1xi_eta = np.array([[N1xi], [N1eta]])
                # N2xi_eta = np.array([[N2xi], [N2eta]])
                # N3xi_eta = np.array([[N3xi], [N3eta]])
                # N4xi_eta = np.array([[N4xi], [N4eta]])
                #
                # # 2.3.4 Gradient Matrix
                # Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])
                #
                # # 2.3.5 Jacobian
                # J = np.matmul(Grad_Mat, v)
                # J_inv = np.linalg.inv(J)
                # J_det = np.linalg.det(J)
                #
                # # 2.3.6 Derivatives of Shape Functions in x-y-Space
                # N1x_y = np.matmul(J_inv, N1xi_eta)
                # N2x_y = np.matmul(J_inv, N2xi_eta)
                # N3x_y = np.matmul(J_inv, N3xi_eta)
                # N4x_y = np.matmul(J_inv, N4xi_eta)
                # N1x = float(N1x_y[0])
                # N1y = float(N1x_y[1])
                # N2x = float(N2x_y[0])
                # N2y = float(N2x_y[1])
                # N3x = float(N3x_y[0])
                # N3y = float(N3x_y[1])
                # N4x = float(N4x_y[0])
                # N4y = float(N4x_y[1])
                #
                # # 2.1.7 Shear Strain matrix with reduced integration
                # #       - Rotated global coordinates with Tk
                # Tk = rotLG(k)[0]
                # B_sred = np.array(
                #     [[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0, 0, 0, N4x, -N4xe, 0, 0],
                #      [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])

                # Return wished for Shear matrix
                # B_s = B_sf
                B_s = B_sats
                # B_s = B_sred
        # -----------------------------------------------------------------------------------------------------------------#
        # 3 Integration of triangles
        # -----------------------------------------------------------------------------------------------------------------#
        else:
            if order == 1:
                # 3.1 Integration of Bending and Membrane Terms

                # 3.1.1 Coordinates of element nodes: delete entry 100'001
                v = np.array(NODESL[self.ELEMENTS[k][0:3]])

                # 3.1.2 Gauss Points
                gp, w = self.gauss_points(3, go)
                xi = gp[j]
                eta = gp[i]

                # 3.1.3 Shape Functions in xi-eta space
                N1xe = 1 - xi - eta
                N2xe = xi
                N3xe = eta

                # 3.1.4 Derivatives Shape Functions in xi-eta space
                N1xi = -1
                N2xi = 1
                N3xi = 0
                N1eta = -1
                N2eta = 0
                N3eta = 1

                N1xi_eta = np.array([[N1xi], [N1eta]])
                N2xi_eta = np.array([[N2xi], [N2eta]])
                N3xi_eta = np.array([[N3xi], [N3eta]])

                # 3.1.5 Gradient Matrix
                Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

                # 3.1.6 Jacobian
                J = np.matmul(Grad_Mat, v)
                J_inv = np.linalg.inv(J)
                J_det = np.linalg.det(J)

                # 3.1.7 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(J_inv, N1xi_eta)
                N2x_y = np.matmul(J_inv, N2xi_eta)
                N3x_y = np.matmul(J_inv, N3xi_eta)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])

                # 3.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
                #       without prescribed shear transverse shear strain (with locking)
                #       - Rotated global coordinates with Tk
                B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
                                [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])
                B_b = np.array([[0, 0, 0, N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0],
                                [0, 0, 0, 0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0],
                                [0, 0, 0, N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0]])
                B_m = np.array([[N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0, 0, 0, 0],
                                [0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0, 0, 0, 0],
                                [N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0, 0, 0, 0]])

                # 3.2 Integration of Shear Terms: Assumed Transverse Shear Element: constant! only dep. on a1 and a2
                Js = np.zeros((2, 2, 2))
                xisall = np.array([0.5,0])
                etasall = np.array([0, 0.5])
                for ijs in range(2):

                    # 3.2.1 Integration points in edge centers
                    xis = xisall[ijs]
                    etas = etasall[ijs]

                    # 3.2.2 Shape Functions in xi-etas-space
                    N1s = 1 - xis - etas
                    N2s = xis
                    N3s = etas

                    # 3.2.3 Derivatives of Shape Functions in xis-etas-space
                    N1xis = -1
                    N2xis = 1
                    N3xis = 0
                    N1etas = -1
                    N2etas = 0
                    N3etas = 1

                    N1xis_etas = np.array([[N1xis], [N1etas]])
                    N2xis_etas = np.array([[N2xis], [N2etas]])
                    N3xis_etas = np.array([[N3xis], [N3etas]])

                    # 3.1.4 Gradient Matrix
                    Grad_Mats = np.array([[N1xis, N2xis, N3xis], [N1etas, N2etas, N3etas]])

                    # 3.1.5 Jacobian
                    Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                    Js_inv = np.linalg.inv(Js[ijs][:][:])
                    Js_det = np.linalg.det(Js[ijs][:][:])

                    # 3.2.4 Derivatives of Shape Functions in x-y-Space
                    N1x_y = np.matmul(Js_inv, N1xis_etas)
                    N2x_y = np.matmul(Js_inv, N2xis_etas)
                    N3x_y = np.matmul(Js_inv, N3xis_etas)
                    N1x = float(N1x_y[0])
                    N1y = float(N1x_y[1])
                    N2x = float(N2x_y[0])
                    N2y = float(N2x_y[1])
                    N3x = float(N3x_y[0])
                    N3y = float(N3x_y[1])


                    # 3.2.5 Bijs-Matrix
                    B_ijs = np.array(
                        [[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0],
                        [0, 0, N1y, 0, -N1s, 0, 0, 0, N2y, 0, -N2s, 0, 0, 0, N3y, 0, -N3s, 0]
                        ])

                    # 3.2.6 Assembly to B-bar Matrix
                    if ijs == 0:
                        B_bar = B_ijs
                    else:
                        B_bar = np.append(B_bar, B_ijs, axis=0)

                # 3.2.7 Auxiliary Matrices for prescribed transverse shear strain field

                Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0],
                                [Js[0][1, 0], Js[0][1, 1], 0, 0],
                                [0, 0, Js[1][0, 0], Js[1][0, 1]],
                                [0, 0, Js[1][1, 0], Js[1][1, 1]]])
                Tmat = np.array([[1, 0, 0,0], [0,0, 0, 1]])
                Pmat = np.array([[1, 0,], [0,1]])
                Amat = np.array([[1, 0], [0, 1]])

                # 3.2.8 B_s-Matrix
                Tk = self.rotLG(k)[0]
                Tk = Tk[0:18,0:18]
                B_s = np.matmul(J_inv,
                                np.matmul(Amat, np.matmul(np.linalg.inv(Pmat), np.matmul(Tmat, np.matmul(Cmat, B_bar)))))

                # 3.2 Integration of Shear Terms: Reduced Integration
                # 3.2.1 Gauss Points
                # xi = 1/3
                # eta = 1/3
                #
                # # 3.2.2 Shape Functions in xi-eta space
                # N1xe = 1 - xi - eta
                # N2xe = xi
                # N3xe = eta
                #
                # # 3.2.3 Derivatives Shape Functions in xi-eta space
                # N1xi = -1
                # N2xi = 1
                # N3xi = 0
                # N1eta = -1
                # N2eta = 0
                # N3eta = 1
                # N1xi_eta = np.array([[N1xi], [N1eta]])
                # N2xi_eta = np.array([[N2xi], [N2eta]])
                # N3xi_eta = np.array([[N3xi], [N3eta]])
                #
                # # 3.2.4 Gradient Matrix
                # Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])
                #
                # # 3.2.5 Jacobian
                # J = np.matmul(Grad_Mat, v)
                # J_inv = np.linalg.inv(J)
                # J_det = np.linalg.det(J)
                #
                # # 3.2.6 Derivatives of Shape Functions in x-y-Space
                # N1x_y = np.matmul(J_inv, N1xi_eta)
                # N2x_y = np.matmul(J_inv, N2xi_eta)
                # N3x_y = np.matmul(J_inv, N3xi_eta)
                # N1x = float(N1x_y[0])
                # N1y = float(N1x_y[1])
                # N2x = float(N2x_y[0])
                # N2y = float(N2x_y[1])
                # N3x = float(N3x_y[0])
                # N3y = float(N3x_y[1])
                #
                # # 3.2.7 Strain matrix in transverse shear with reduced integration
                # Tk = rotLG(k)[0]
                # Tk = Tk[0:18,0:18]
                # B_s = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
                #                  [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])


        end = time.time()
        time_B._updatetime(delta_t=end - start)
        # 4 Return B-Matrices in Global or Local Coordinate System
        if rot:
            return B_m@Tk,B_b@Tk,B_s@Tk,J_det
        else:
            return B_m,B_b,B_s,J_det


    def find_b(self,go):
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        Bmr = {}
        Bbr = {}
        Bsr = {}
        Bmnr = {}
        Bbnr = {}
        Bsnr = {}
        J = {}
        for k in range(0, nk):
            Bmr[k] = {}
            Bbr[k] = {}
            Bsr[k] = {}
            Bmnr[k] = {}
            Bbnr[k] = {}
            Bsnr[k] = {}
            J[k] = {}
            gp, w = self.gauss_points(self.ELS[4][k], go)
            n_k = self.find_nodes(k)
            n_k = n_k[n_k < 100000]
            for i in range(len(gp)):
                Bmr[k][i] = {}
                Bbr[k][i] = {}
                Bsr[k][i] = {}
                Bmnr[k][i] = {}
                Bbnr[k][i] = {}
                Bsnr[k][i] = {}
                J[k][i] = {}
                for j in range(len(gp)):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        continue
                    [Bmr[k][i][j], Bbr[k][i][j], Bsr[k][i][j], J[k][i][j]] = self.b_kij(k, i, j, go, rot=True)
                    [Bmnr[k][i][j], Bbnr[k][i][j], Bsnr[k][i][j], J[k][i][j]] = self.b_kij(k, i, j, go, rot=False)
        B = {}
        B["Bm"] = {}
        B["Bm"]["r"] = Bmr
        B["Bm"]["nr"] = Bmnr
        B["Bb"] = {}
        B["Bb"]["r"] = Bbr
        B["Bb"]["nr"] = Bbnr
        B["Bs"] = {}
        B["Bs"]["r"] = Bsr
        B["Bs"]["nr"] = Bsnr
        B["Jdet"] = J
        return B

    """---------------------------------------- Stiffness Matrix Calculation---------------------------------------------"""

    def rotLG(self,k):

        Tk1 = self.GEOMK["Tk1"][k]
        Tk2 = self.GEOMK["Tk2"][k]
        Tk3 = self.GEOMK["Tk3"][k]

        temp1 = np.append(Tk1, np.zeros((3, 3)), axis=1)
        temp2 = np.append(np.zeros((3, 3)), Tk2, axis=1)
        temp3 = np.append(temp1, temp2, axis=0)
        temp4 = np.append(temp3, np.zeros((6, 18)), axis=1)
        temp5 = np.append(np.append(np.zeros((6, 6)), temp3, axis=1), np.zeros((6, 12)), axis=1)
        temp6 = np.append(np.append(np.zeros((6, 12)), temp3, axis=1), np.zeros((6, 6)), axis=1)
        temp7 = np.append(np.zeros((6, 18)), temp3, axis=1)
        Tk = np.append(np.append(temp4, temp5, axis=0), np.append(temp6, temp7, axis=0), axis=0)

        temp1 = np.append(np.zeros((3, 3)), np.zeros((3, 3)), axis=1)
        temp2 = np.append(np.zeros((3, 3)),Tk3, axis=1)
        temp3 = np.append(temp1,temp2, axis=0)
        temp4 = np.append(temp3,np.zeros((6,6)),axis = 1)
        temp5 = np.append(np.zeros((6,6)),temp3,axis = 1)
        temp6 = np.append(temp4,temp5,axis=0)
        temp7 = np.append(temp6,np.zeros((12,12)),axis=1)
        temp8 = np.append(np.zeros((12, 12)),temp6, axis=1)
        Tkr = np.append(temp7,temp8,axis=0)

        return Tk,Tkr
        # return np.identity(24),np.identity(24)


    def dh_kij(self,e_kij,s_kij, k, i, j, cm_k):
        Dmh = np.zeros((3, 3))
        Dbh = np.zeros((3, 3))
        Dmbh = np.zeros((3, 3))
        Dsh = np.zeros((2, 2))
        nlk = self.GEOMK["nlk"][k]
        if cm_k == 1 or cm_k == 3:
            t_k = self.GEOMK["t"][k]
            for l in range(nlk):
                z = -t_k / 2 + (2 * l + 1) * t_k / (2 * nlk)
                E = self.MATK["Ec"][k]
                v = self.MATK["vc"][k]

                # Dp = self.get_et(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)
                Dp = self.get_et_vb(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)
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
            t, E, v = np.zeros((1,nlk)), np.zeros((1,nlk)), np.zeros((1,nlk))
            Dp = np.zeros((nlk,3,3))
            t_k1, t_k2 = self.GEOMK["t"][k], self.GEOMK["t2"][k]
            t_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
            E_1, E_2 = self.MATK["Ec"][k], self.MATK["Ec2"][k]
            v_1, v_2 = self.MATK["vc"][k], self.MATK["vc2"][k]
            for l in range(nlk):
                Dp_all = self.get_et_vb(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)                # yields a dict with ET1 and ET2
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


    def dh_klij_vec(self, s, cmk):
        """
        Vectorised per-(k, l, i, j) layer tangent matrices. Vec-form analogue of the inner
        per-layer block of dh_kij (the loop body that calls get_et_vb). Handles cm == 1 and
        cm == 3 per element (mirroring the cm branching in find_dh_vec / get_et_vec).

        Args:
            s    (np.arr): layer stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
                           Axis -2 = strain perturbation index (0..2 for ex, ey, gxy),
                           axis -1 = stress component (sx, sy, txy, txz, tyz).
                           For elements with cm == 1, s.imag is ignored (the tangent is
                           analytical), so s may be a stub array there.
            cmk  (array-like): per-element cm identifier, shape (nel,). Only values in {1, 3}
                           are supported. cm == 10 (alternating layer materials) needs a
                           separate branch -- not implemented.

        Returns:
            Dp   (np.arr): in-plane tangent ET[a, b] = d s_in[a] / d e_in[b],
                           shape (nel, nlk, go, go, 3, 3)
            Ds   (np.arr): out-of-plane shear tangent (5/6 * G * I), per (k, l) -- no go axes,
                           shape (nel, nlk, 2, 2)

        Per-element constitutive split:
            cm == 1: analytical plane-stress tangent, ET = E/(1-v^2) * [[1, v, 0], [v, 1, 0],
                     [0, 0, (1-v)/2]] -- constant across layers and Gauss points.
            cm == 3: complex-step tangent from s.imag.

        Convention note: s_flat[..., p, c] from find_s_vec is the c-th stress component when
        the p-th strain was perturbed; its imag part is d(s_c) / d(e_p). To match the standard
        ET[a, b] = d(s_a) / d(e_b) convention used by k_k / get_et_vb, we swap the perturbation
        and component axes before scaling by 1/eps.
        """
        nel = s.shape[0]
        nlk = s.shape[1]

        cm_arr = np.asarray(cmk)
        valid_cm = np.isin(cm_arr, [1, 3])
        if not valid_cm.all():
            invalid = np.unique(cm_arr[~valid_cm])
            raise NotImplementedError(
                f"dh_klij_vec: only cm in {{1, 3}} supported; got cm values {invalid.tolist()}."
            )

        Ec = np.asarray(self.MATK["Ec"], dtype=float)                            # (nel,)
        vc = np.asarray(self.MATK["vc"], dtype=float)                            # (nel,)

        # cm == 3 path: complex-step tangent, ET[..., a, b] = s[..., b, a].imag / EPS
        Dp_cm3 = s[..., :3, :3].imag.swapaxes(-2, -1) / 1e-16                    # (nel, nlk, go, go, 3, 3)

        # cm == 1 path: analytical plane-stress tangent per element, constant across (l, i, j).
        Dp_cm1 = np.zeros((nel, 3, 3))
        Dp_cm1[:, 0, 0] = 1.0
        Dp_cm1[:, 0, 1] = vc
        Dp_cm1[:, 1, 0] = vc
        Dp_cm1[:, 1, 1] = 1.0
        Dp_cm1[:, 2, 2] = 0.5 * (1.0 - vc)
        Dp_cm1 = (Ec / (1.0 - vc**2))[:, None, None] * Dp_cm1                    # (nel, 3, 3)

        # Combine per element via mask (broadcasts the cm==1 matrix over l, i, j).
        mask_cm1 = (cm_arr == 1)
        Dp = np.where(mask_cm1[:, None, None, None, None, None],
                      Dp_cm1[:, None, None, None, :, :],
                      Dp_cm3)                                                    # (nel, nlk, go, go, 3, 3)

        # Out-of-plane shear tangent: 5/6 * G * I, G = Ec / (2 * (1 + vc)), per element.
        # Same formula for cm == 1 and cm == 3 (k_k's Ds doesn't depend on cm).
        G  = Ec / (2.0 * (1.0 + vc))                                             # (nel,)
        I2 = np.eye(2)
        Ds = (5.0 / 6.0) * G[:, None, None, None] * np.broadcast_to(I2, (nel, nlk, 2, 2))
        Ds = np.ascontiguousarray(Ds)                                            # (nel, nlk, 2, 2)

        return Dp, Ds


    def find_dh_vec(self, s, mat_dict, cm_klij = 3, go = 1) -> np.array:
        raise Warning('Not in use')
        """
        Vectorised version of Andreas' function find dh for go = 1
        Args:
            s       (np.arr): layer stresses (n_tot, 20,3)
            mat_dict  (dict): material parameters
        Returns:
            dh       (np.arr): stiffness matrix entries (n_tot, 6, 6)

        """
        t0 = time.perf_counter()
        t = self.constants['t']
        nl = self.constants['n_layer']
        l = np.arange(nl)                               # shape (nl,)
        z = -t / 2 + (2 * l + 1) * t / (2 * nl)         # shape (nl,)

        # Dmh = np.zeros((s.shape[0],3,3))
        # Dbh = np.zeros((s.shape[0],3,3))
        # Dmbh = np.zeros((s.shape[0],3,3))

        Dp = self.get_et_vec(s, mat_dict, cm_klij = cm_klij)       # shape (n_tot, 20, 3, 3)

        z_ = z.reshape(1, -1, 1, 1)

        Dmh     = np.sum((Dp)       , axis = 1)*t/nl               # shape (n_tot, 3, 3)
        Dmbh    = np.sum((-z_*Dp)   , axis = 1)*t/nl                # shape (n_tot, 3, 3)
        Dbh     = np.sum((z_**2*Dp) , axis = 1)*t/nl                # shape (n_tot, 3, 3)

        De_1 = np.concatenate([Dmh, Dmbh], axis=2)          # (n_tot, 3, 6)
        De_2 = np.concatenate([Dmbh, Dbh], axis=2)          # (n_tot, 3, 6)
        De   = np.concatenate([De_1, De_2], axis=1)         # (n_tot, 6, 6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated stiffness matrix D in {t1/60:.2f} min.')
        return De

    def get_et_vec(self, s, mat_dict, cm_klij = 3) -> np.array:
        raise Warning('Not in use')
        """
        Calculates per-layer stiffness matrix
        Args:
            s       (np.arr): layer stresses (n_tot, 20,3,3)
            mat_dict  (dict): material parameters
            cmklij     (int): if 1: linear elastic, if 3: concrete, nonlinear
        Returns:
            dp       (np.arr): stiffness matrix entries (n_tot, 20, 3, 3)
        """

        n_tot = s.shape[0]
        nl = self.constants['n_layer']
           
        if cm_klij == 1:
            E = mat_dict['Ec']
            v = self.constants['nu']
            ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
            dp = np.broadcast_to(ET[np.newaxis, np.newaxis,:,:], (n_tot, nl, 3,3))
        
        elif cm_klij == 3: 
            ET = s.imag / 1e-16
            dp = ET


            # for debugging:
            # fig1, ax1 = plt.subplots(figsize=(10, 8))
            # im1 = ax1.imshow(ET.reshape((s.shape[0],180)), aspect='auto', cmap='viridis', interpolation='nearest')
            # plt.colorbar(im1, ax=ax1)
            # plt.show()
            
        return dp


    def k_k(self,Bm_k,Bb_k,Bs_k,Jdet_k,e_k,s_k, k, cm_k, go, perm = None, perm1=None, random_factor1 = None):
        ne_k = self.ELEMENTS[k, :]
        ne_k = ne_k[ne_k<10**5]
        a_k = self.GEOMK["ak"][k]
        NODESL = self.COORD["n"][2][a_k]
        if len(ne_k) == 4:
            v = np.array(NODESL[self.ELEMENTS[k]])
            Tkr = self.rotLG(k)[1]
        else:
            v = np.array(NODESL[self.ELEMENTS[k][0:3]])
            Tkr = self.rotLG(k)[1]
            Tkr = Tkr[0:18,0:18]

        """-------------------- Integration of Membrane,Bending,Shear and Coupling Stiffness Matrix ---------------------"""
        gp, w = self.gauss_points(self.ELS[4][k],go)
        Kbe = np.zeros((len(v) * 6, len(v) * 6))
        Kme = np.zeros((len(v) * 6, len(v) * 6))
        Kmbe= np.zeros((len(v) * 6, len(v) * 6))
        Kse = np.zeros((len(v) * 6, len(v) * 6))
        De_ = np.zeros((go, go, 8, 8))
        A_k = 0
        for i in range(len(gp)):
            for j in range(len(gp)):
                if self.ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                Bm = Bm_k[i][j]
                Bb = Bb_k[i][j]
                Bs = Bs_k[i][j]
                Jdet = Jdet_k[i][j]
                # D-Matrices
                Dmh,Dmbh,Dbh,Dsh = self.dh_kij(e_k[:,i,j,:],s_k[:,i,j], k, i, j, cm_k)
                De_1 = np.hstack([Dmh,Dmbh,np.zeros((3, 2))])
                De_2 = np.hstack([Dmbh,Dbh,np.zeros((3,2))])
                De_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), Dsh])
                De = np.vstack([De_1, De_2, De_3])
                if perm is not None:
                    # random_factor = np.random.uniform(perm[0], perm[1])
                    # Dmh_, Dmbh_, Dbh_, Dsh_ =  random_factor*Dmh,random_factor*Dmbh,random_factor*Dbh,random_factor*Dsh
                    # De = random_factor*De
                    random_matrix = np.random.uniform(perm[0], perm[1], (3,3))
                    random_matrix_s = np.random.uniform(perm[0], perm[1], (2,2))
                    Dmh_, Dmbh_, Dbh_, Dsh_ =  np.multiply(random_matrix,Dmh),np.multiply(random_matrix,Dmbh),np.multiply(random_matrix,Dbh),np.multiply(random_matrix_s, Dsh)
                    De_1 = np.hstack([Dmh_,Dmbh_,np.zeros((3, 2))])
                    De_2 = np.hstack([Dmbh_,Dbh_,np.zeros((3,2))])
                    De_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), Dsh_])
                    De = np.vstack([De_1, De_2, De_3])
                    if (Dmh - Dmh_ == 0).any():
                        print('The random factor did not permute the D-values; Dmh = Dmh_')
                elif perm1 is not None: 
                    # only permutes D_00 with random factor
                    # random_factor = np.random.uniform(perm1[0], perm1[1])
                    random_matrix = np.ones((3,3))
                    random_matrix[0,0] = random_factor1
                    Dmh_, Dmbh_, Dbh_, Dsh_ =  np.multiply(random_matrix,Dmh),Dmbh,Dbh,Dsh
                    De_1 = np.hstack([Dmh_,Dmbh_,np.zeros((3, 2))])
                    De_2 = np.hstack([Dmbh_,Dbh_,np.zeros((3,2))])
                    De_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), Dsh_])
                    De = np.vstack([De_1, De_2, De_3])
                    if (Dmh[0,0] - Dmh_[0,0] == 0):
                        print('The random factor did not permute the D-values; Dmh = Dmh_')
                else: 
                    Dmh_, Dmbh_, Dbh_, Dsh_ = Dmh,Dmbh,Dbh,Dsh
                Kbe = np.add(Kbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bb), np.matmul(Dbh_, Bb)))
                Kme = np.add(Kme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm), np.matmul(Dmh_, Bm)))
                Kmbe= np.add(Kmbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dmbh_, Bb)))
                Kse = np.add(Kse, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bs), np.matmul(Dsh_, Bs)))
                A_k+=Jdet*w[i]*w[j]
                De_[i,j,:,:] = De
                # print('Element number', k)
                # print('Generalised Strains', e_k[:,:,:,:])
                # print('e_k.shape', e_k.shape)
                # print('D (analytical)', De)

        """--------------------------------- Stiffness Term for rotational DOF ------------------------------------------
        to avoid coplanar signularity effects -- see book chapter 8.9.3 ----------------------------------------------"""
        n_k = self.ELEMENTS[k]
        if n_k.any() in self.copln:
            iscoplk = 1
        else:
            iscoplk = 0
        """--------------------------------- Assembly of entire Stiffness Matrix ----------------------------------------"""
        if cm_k == 1 or cm_k == 3:
            Ke = Kme+Kbe+Kse+Kmbe+np.transpose(Kmbe)+iscoplk*A_k*self.GEOMK["t"][k]*33600*Tkr*10**-8
        elif cm_k == 10:
            nlk = self.GEOMK["nlk"][k]
            t_k1, t_k2 = self.GEOMK["t"][k], self.GEOMK["t2"][k]
            t_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
            Ke = Kme+Kbe+Kse+Kmbe+np.transpose(Kmbe)+iscoplk*A_k*t_tot*33600*Tkr*10**-8
        if Jdet < 0:
            print(self.ELS[0][k])
            print(self.ELS[0][k+1])
            print(Jdet)
        return Ke, De_


    def k_k_vec(self, B, e, s, cmk, go):
        """
        Vectorised per-element stiffness for all elements at once. Combines layer integration
        (Dmh, Dmbh, Dbh, Dsh) and Gauss-point integration of B^T D B into batched einsums,
        grouped by element type (4-node quads, 3-node tris).

        Args:
            B    (dict): B-matrix container (uses B["Bm"]["r"], B["Bb"]["r"], B["Bs"]["r"], B["Jdet"]).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5). Currently unused
                           in the cm == 3 path (the tangent comes from s.imag), accepted for
                           signature parity with k_k.
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
            cmk  (array-like): cm per element, shape (nel,).
            go   (int):  Gauss order.

        Returns:
            Ke_dict (dict): {k: Ke[k]} per element. quads -> (24, 24), tris -> (18, 18).
            De      (np.arr): 8x8 tangent per (k, i, j), shape (nel, go, go, 8, 8).

        Drops perm / perm1 (random tangent permutations) and the drilling / coplanar correction
        term from k_k -- add back if a specific mesh needs them. Supports cm in {1, 3} per
        element via dh_klij_vec (mixed cm in the same mesh is fine). cm == 10 (alternating
        layer materials) is not implemented.
        """
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        cm_arr   = np.asarray(cmk)
        valid_cm = np.isin(cm_arr, [1, 3])
        if not valid_cm.all():
            invalid = np.unique(cm_arr[~valid_cm])
            raise NotImplementedError(
                f"k_k_vec: only cm in {{1, 3}} supported; got cm values {invalid.tolist()}."
            )

        # 1 Per-layer tangents -----------------------------------------------------------------
        Dp_klij, Ds_kl = self.dh_klij_vec(s, cmk)                           # (nel, nlk, go, go, 3, 3), (nel, nlk, 2, 2)

        # 2 Layer offsets and thicknesses (uniform layer layout, cm == 3) ---------------------
        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)                # (nel,)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)                  # (nel,)
        l_idx   = np.arange(nlk)                                            # (nlk,)
        z  = (-t_arr[:, None] / 2.0
              + (2 * l_idx[None, :] + 1) * t_arr[:, None] / (2.0 * nlk_arr[:, None]))   # (nel, nlk)
        dz = np.broadcast_to(t_arr[:, None] / nlk_arr[:, None], (nel, nlk)).copy()      # (nel, nlk)
        layer_mask = l_idx[None, :] < nlk_arr[:, None]
        dz = dz * layer_mask                                                # zero past nlk_k

        # 3 Layer integration -> per (k, i, j) 3x3 / 2x2 blocks --------------------------------
        z_exp  = z [:, :, None, None, None, None]
        dz_exp = dz[:, :, None, None, None, None]
        Dmh  = np.sum(           Dp_klij * dz_exp, axis=1)                  # (nel, go, go, 3, 3)
        Dmbh = np.sum(-z_exp   * Dp_klij * dz_exp, axis=1)                  # (nel, go, go, 3, 3)
        Dbh  = np.sum( z_exp**2 * Dp_klij * dz_exp, axis=1)                 # (nel, go, go, 3, 3)
        Dsh_kl   = np.sum(Ds_kl * dz[:, :, None, None], axis=1)             # (nel, 2, 2)
        Dsh_full = np.broadcast_to(Dsh_kl[:, None, None, :, :],
                                   (nel, go, go, 2, 2))                     # (nel, go, go, 2, 2)

        # 4 Assemble 8x8 De block per (k, i, j) ------------------------------------------------
        # Block layout matches non-vec k_k: [[Dmh, Dmbh, 0], [Dmbh, Dbh, 0], [0, 0, Dsh]].
        # Note: the off-diagonal block is Dmbh (not Dmbh^T) on BOTH sides -- the symmetrisation
        # for the assembled K comes from `Kmbe + Kmbe^T` in step 5, not from De itself.
        De = np.zeros((nel, go, go, 8, 8))
        De[..., 0:3, 0:3] = Dmh
        De[..., 0:3, 3:6] = Dmbh
        De[..., 3:6, 0:3] = Dmbh
        De[..., 3:6, 3:6] = Dbh
        De[..., 6:8, 6:8] = Dsh_full

        # 5 Per-element-type Ke via batched block-wise einsums --------------------------------
        Ke_dict = {}
        types = np.asarray(self.ELS[4])
        for n_nodes, ndof in ((4, 24), (3, 18)):
            els = np.where(types == n_nodes)[0]
            E_count = len(els)
            if E_count == 0:
                continue

            gp, w = self.gauss_points(n_nodes, go)
            w_grid = np.outer(w, w).copy()                                  # (go, go)
            if n_nodes == 3 and go >= 2:
                w_grid[1, 1] = 0.0

            B_batch    = np.zeros((E_count, go, go, 8, ndof))
            Jdet_batch = np.zeros((E_count, go, go))
            for idx, k in enumerate(els):
                for i in range(go):
                    for j in range(go):
                        if n_nodes == 3 and i == 1 and j == 1:
                            continue
                        Bm   = B["Bm"]["r"][k][i][j]
                        Bb   = B["Bb"]["r"][k][i][j]
                        Bs   = B["Bs"]["r"][k][i][j]
                        B_batch[idx, i, j]    = np.vstack([Bm, Bb, Bs])
                        Jdet_batch[idx, i, j] = B["Jdet"][k][i][j]

            weight = w_grid[None, :, :] * Jdet_batch                        # (E, go, go)

            # Block-wise sub-B matrices
            Bm_b = B_batch[..., 0:3, :]                                     # (E, go, go, 3, ndof)
            Bb_b = B_batch[..., 3:6, :]                                     # (E, go, go, 3, ndof)
            Bs_b = B_batch[..., 6:8, :]                                     # (E, go, go, 2, ndof)

            Dmh_g  = Dmh[els]
            Dmbh_g = Dmbh[els]
            Dbh_g  = Dbh[els]
            Dsh_g  = Dsh_full[els]

            # K_block[e, a, b] = sum_{i, j, p, q} weight * B^T D B
            Kme_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmh_g,  Bm_b)
            Kbe_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dbh_g,  Bb_b)
            Kse_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bs_b, Dsh_g,  Bs_b)
            Kmbe_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmbh_g, Bb_b)

            # Ke = Kme + Kbe + Kse + Kmbe + Kmbe^T  (matches non-vec k_k)
            Ke_batch = (Kme_batch + Kbe_batch + Kse_batch
                        + Kmbe_batch + np.transpose(Kmbe_batch, (0, 2, 1)))   # (E, ndof, ndof)

            # Drilling / coplanar correction: add iscoplk * A_k * t_k * 33600 * Tkr * 1e-8
            # per element (Tkr / iscoplk don't easily vectorise).
            A_k_arr = weight.sum(axis=(1, 2))                               # (E,)
            for idx, k in enumerate(els):
                n_k = self.ELEMENTS[k]
                iscoplk = 1 if (n_k.any() in self.copln) else 0
                Ke_i = Ke_batch[idx]
                if iscoplk:
                    Tkr = self.rotLG(k)[1]
                    if n_nodes == 3:
                        Tkr = Tkr[:18, :18]
                    t_k = self.GEOMK["t"][k]
                    Ke_i = Ke_i + iscoplk * A_k_arr[idx] * t_k * 33600 * Tkr * 1e-8
                Ke_dict[int(k)] = Ke_i

        return Ke_dict, De


    def k_k_nn(self,Bm_k,Bb_k,Bs_k,Jdet_k,eh_k,sh_k, k, cm_k):
        ne_k = self.ELEMENTS[k, :]
        ne_k = ne_k[ne_k<10**5]
        a_k = self.GEOMK["ak"][k]
        NODESL = self.COORD["n"][2][a_k]
        if len(ne_k) == 4:
            v = np.array(NODESL[self.ELEMENTS[k]])
            Tkr = self.rotLG(k)[1]
        else:
            v = np.array(NODESL[self.ELEMENTS[k][0:3]])
            Tkr = self.rotLG(k)[1]
            Tkr = Tkr[0:18,0:18]

        Kbe = np.zeros((len(v) * 6, len(v) * 6))
        Kme = np.zeros((len(v) * 6, len(v) * 6))
        Kmbe= np.zeros((len(v) * 6, len(v) * 6))
        Kbme= np.zeros((len(v) * 6, len(v) * 6))
        Kse = np.zeros((len(v) * 6, len(v) * 6))

        """-------------------- Integration of Membrane,Bending,Shear and Coupling Stiffness Matrix ---------------------"""
        gp, w = self.gauss_points(self.ELS[4][k],self.gauss_order)
        K_el = np.zeros((len(v) * 6, len(v) * 6))
        A_k = 0
        t_k = self.GEOMK["t"][k]
        De_ = np.zeros((self.gauss_order,self.gauss_order,8,8))
        for i in range(len(gp)):
            for j in range(len(gp)):
                if self.ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                Bm = Bm_k[i][j]
                Bb = Bb_k[i][j]
                Bs = Bs_k[i][j]
                B_el = np.vstack((Bm, Bb, Bs))
                Jdet = Jdet_k[i][j]
                ######## collect input data for NN ########
                eh_flat = np.reshape(eh_k[i,j,:], (1,8))                
                if self.MATK["cm"][k] == 3:
                    index = np.where(dict_CC['Ec']==self.MATK['Ec'][k])[0]
                    CC = dict_CC['CC'][int(index)]
                    # input_j = np.concatenate((np.array(eh_flat), [[t_k]], [[self.GEOMK['rhox'][k][0]]], [[self.GEOMK['rhoy'][k][0]]], [[CC]]), axis = 1)
                    input_j = np.array(eh_flat[:,:6])
                    chosen_model_path = self.check_range_NN(input_j,'eps-t', predict = 'D', cmk = self.MATK["cm"][k])
                elif self.MATK["cm"][k] == 1 or self.MATK["cm"][k] == 10:
                    raise RuntimeError('This function should not be called if the material model is linear elastic.')
                ######## make prediction for D with NN ########
                if i==0 and j == 0 and k == 0:
                    mat_NN = make_NN_prediction(input_j, predict = 'D', model_path = chosen_model_path, non_batchwise=True)
                else:
                    with HiddenPrints():
                        mat_NN = make_NN_prediction(input_j, predict = 'D', model_path = chosen_model_path, non_batchwise=True)
                D_pred = mat_NN['D']
                self.check_range_NN(D_pred,'D', cmk = self.MATK["cm"][k])
                De = D_pred[0,:,:]
                De_[i,j,:,:] = De

                De[0:3,0:3] = self.filter_small_stiffness(De[0:3,0:3],k)
                # K_el = np.add(K_el, w[i]*w[j]*Jdet*np.matmul(np.transpose(B_el), np.matmul(De, B_el)))
                Dbh_, Dmh_, Dmbh_, Dbmh_, Dsh_ = De[3:6,3:6], De[0:3,0:3], De[0:3,3:6], De[3:6,0:3], De[6:8,6:8]
                
                Kbe = np.add(Kbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bb), np.matmul(Dbh_, Bb)))
                Kme = np.add(Kme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm), np.matmul(Dmh_, Bm)))
                Kmbe= np.add(Kmbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dmbh_, Bb)))
                Kbme= np.add(Kbme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dbmh_, Bb)))
                Kse = np.add(Kse, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bs), np.matmul(Dsh_, Bs)))
                A_k+=Jdet*w[i]*w[j]

        """--------------------------------- Stiffness Term for rotational DOF ------------------------------------------
        to avoid coplanar signularity effects -- see book chapter 8.9.3 ----------------------------------------------"""
        n_k = self.ELEMENTS[k]
        if n_k.any() in self.copln:
            iscoplk = 1
        else:
            iscoplk = 0
        """--------------------------------- Assembly of entire Stiffness Matrix ----------------------------------------"""
        Ke = Kme+Kbe+Kse+Kmbe+Kbme+iscoplk*A_k*self.GEOMK["t"][k]*33600*Tkr*10**-8
        # Ke = K_el+iscoplk*A_k*self.GEOMK["t"][k]*33600*Tkr*10**-8
        
        if Jdet < 0:
            print(self.ELS[0][k])
            print(self.ELS[0][k+1])
            print(Jdet)
        return Ke, De_

    def k_k_nn_num(self,Bm_k,Bb_k,Bs_k,Jdet_k,eh_k,sh_k, e_k,s_k, k, cm_k, model_dim, scenario):
        '''
        Determines k_k for NNs that only predict sub-parts of D-matrix
        '''

        ne_k = self.ELEMENTS[k, :]
        ne_k = ne_k[ne_k<10**5]
        a_k = self.GEOMK["ak"][k]
        NODESL = self.COORD["n"][2][a_k]
        if len(ne_k) == 4:
            v = np.array(NODESL[self.ELEMENTS[k]])
            Tkr = self.rotLG(k)[1]
        else:
            v = np.array(NODESL[self.ELEMENTS[k][0:3]])
            Tkr = self.rotLG(k)[1]
            Tkr = Tkr[0:18,0:18]

        Kbe = np.zeros((len(v) * 6, len(v) * 6))
        Kme = np.zeros((len(v) * 6, len(v) * 6))
        Kmbe= np.zeros((len(v) * 6, len(v) * 6))
        Kbme= np.zeros((len(v) * 6, len(v) * 6))
        Kse = np.zeros((len(v) * 6, len(v) * 6))
        
        gp, w = self.gauss_points(self.ELS[4][k],self.gauss_order)
        K_el = np.zeros((len(v) * 6, len(v) * 6))
        A_k = 0
        t_k = self.GEOMK["t"][k]
        De_ = np.zeros((self.gauss_order,self.gauss_order,8,8))
        for i in range(len(gp)):
            for j in range(len(gp)):
                if self.ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                Bm = Bm_k[i][j]
                Bb = Bb_k[i][j]
                Bs = Bs_k[i][j]
                B_el = np.vstack((Bm, Bb, Bs))
                Jdet = Jdet_k[i][j]
                ######## collect input data for NN ########
                eh_flat = np.reshape(eh_k[i,j,:], (1,8))                
                if self.MATK["cm"][k] == 3:
                    index = np.where(np.array(dict_CC['Ec'], dtype=int)==int(self.MATK['Ec'][k]))[0]
                    CC = dict_CC['CC'][int(index)]
                    # input_j = np.concatenate((np.array(eh_flat), [[t_k]], [[self.GEOMK['rhox'][k][0]]], [[self.GEOMK['rhoy'][k][0]]], [[CC]]), axis = 1)
                    input_j = np.array(eh_flat[:,:6])
                    chosen_model_path = self.check_range_NN(input_j,'eps-t', predict = 'D', cmk = self.MATK["cm"][k])
                elif self.MATK["cm"][k] == 1 or self.MATK["cm"][k] == 10:
                    raise RuntimeError('This function should not be called if the material model is linear elastic.')
                ######## make prediction for D with NN ########
                if k == 0 and i == 0 and j == 0:
                    mat_NN = make_NN_prediction(input_j, predict = 'D', model_path = chosen_model_path, non_batchwise=True)
                    # mat_NN = predict_sig_D(input_j, chosen_data_path, chosen_model_path, 'train', transf_type = 'st-stitched', predict = 'D', sc=False, model_dim = model_dim)
                else:
                    with HiddenPrints():
                        mat_NN = make_NN_prediction(input_j, predict = 'D', model_path = chosen_model_path, non_batchwise=True)
                        # mat_NN = predict_sig_D(input_j, chosen_data_path, chosen_model_path, 'train', transf_type = 'st-stitched', predict = 'D', sc=False, model_dim = model_dim)
                D_pred = mat_NN['D']
                self.check_range_NN(D_pred,'D', cmk = self.MATK["cm"][k])
                De_NN = D_pred[0,:,:]
                De_NN[0:3,0:3] = self.filter_small_stiffness(De_NN[0:3,0:3],k, scenario)
                ######## calculate D numerically ########
                Dmh,Dmbh,Dbh,Dsh = self.dh_kij(e_k[:,i,j,:],s_k[:,i,j], k, i, j, cm_k)
                De_1 = np.hstack([Dmh,Dmbh,np.zeros((3, 2))])
                De_2 = np.hstack([Dmbh,Dbh,np.zeros((3,2))])
                De_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), Dsh])
                De_num = np.vstack([De_1, De_2, De_3])

                ######## combine the D-matrices ########
                De = De_num
                if model_dim == 'ONEDIM_x':
                    De[0,0] = De_NN[0,0]
                elif model_dim == 'ONEDIM_y':
                    De[1,1] = De_NN[1,1]
                elif model_dim == 'TWODIM':
                    De[0:3,0:3] = De_NN[0:3,0:3]
                elif model_dim == 'THREEDIM':
                    De[0:6,0:6] = De_NN[0:6,0:6]
                De_[i,j,:,:] = De
                # K_el = np.add(K_el, w[i]*w[j]*Jdet*np.matmul(np.transpose(B_el), np.matmul(De, B_el)))
                Dbh_, Dmh_, Dmbh_, Dbmh_, Dsh_ = De[3:6,3:6], De[0:3,0:3], De[0:3,3:6], De[3:6,0:3], De[6:8,6:8]
                
                Kbe = np.add(Kbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bb), np.matmul(Dbh_, Bb)))
                Kme = np.add(Kme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm), np.matmul(Dmh_, Bm)))
                Kmbe= np.add(Kmbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dmbh_, Bb)))
                Kbme= np.add(Kbme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dbmh_, Bb)))
                Kse = np.add(Kse, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bs), np.matmul(Dsh_, Bs)))

                A_k+=Jdet*w[i]*w[j]

        """--------------------------------- Stiffness Term for rotational DOF ------------------------------------------
        to avoid coplanar signularity effects -- see book chapter 8.9.3 ----------------------------------------------"""
        n_k = self.ELEMENTS[k]
        if n_k.any() in self.copln:
            iscoplk = 1
        else:
            iscoplk = 0
        """--------------------------------- Assembly of entire Stiffness Matrix ----------------------------------------"""
        # Ke = K_el+iscoplk*A_k*self.GEOMK["t"][k]*33600*Tkr*10**-8
        Ke = Kme+Kbe+Kse+Kmbe+Kbme+iscoplk*A_k*self.GEOMK["t"][k]*33600*Tkr*10**-8
        
        if Jdet < 0:
            print(self.ELS[0][k])
            print(self.ELS[0][k+1])
            print(Jdet)

        return Ke, De_


    def k_k_nn_num_vec(self, B, e, s, eh, sh, cmk, go, model_dim, scenario, loaded_models=None):
        """
        Vectorised drop-in for k_k_nn_num. Implemented for model_dim == 'THREEDIM' only.

        For each (k, i, j):
            * Predict the upper-left 6x6 block of De from the NN using eh as input.
            * Compute the out-of-plane shear block Dsh (2x2) numerically via the same
              complex-step path as k_k_vec (`dh_klij_vec`).
            * Combine: De[0:6, 0:6] = D_NN; De[6:8, 6:8] = Dsh; off-block zeros.
            * Per-element Ke assembled block-by-block matching k_k_nn_num's original
              formula: Ke = Kme + Kbe + Kse + Kmbe + Kbme + drilling, with
                  Kmbe = w*Jdet * Bm^T Dmbh Bb
                  Kbme = w*Jdet * Bm^T Dbmh Bb 

        Args:
            B    (dict):   B-matrix container (uses ["Bm"]["r"], ["Bb"]["r"], ["Bs"]["r"], ["Jdet"]).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5).
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
            eh   (np.arr): generalised strains, shape (nel, go, go, 8) -- NN input.
            sh   (np.arr): generalised stresses, shape (nel, go, go, 8). Accepted for
                           signature parity; not consumed (matches k_k_nn_num).
            cmk  (array-like): per-element cm. Restricted to all cm == 3.
            go   (int): Gauss order.
            model_dim (str): only 'THREEDIM' implemented.
            scenario  (int): boundary scenario code, passed to filter_small_stiffness.

        Returns:
            Ke_dict (dict): {k: Ke[k]} per element.
            De      (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8).

        """
        if model_dim != 'THREEDIM':
            raise NotImplementedError(
                f"k_k_nn_num_vec: only model_dim='THREEDIM' implemented; got {model_dim!r}"
            )

        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        cm_arr = np.asarray(cmk)
        if not (cm_arr == 3).all():
            raise NotImplementedError("k_k_nn_num_vec: only cm == 3 supported.")
        cmk_val = int(cm_arr.reshape(-1)[0])

        # 1 Numerical Dsh (out-of-plane shear) -- analytical per-layer + layer integration ----
        # Per-layer Ds_kl = 5/6 * G * I2 with G = Ec/(2*(1+vc)).
        Ec = np.asarray(self.MATK["Ec"], dtype=float)                        # (nel,)
        vc = np.asarray(self.MATK["vc"], dtype=float)                        # (nel,)
        G  = Ec / (2.0 * (1.0 + vc))                                          # (nel,)
        I2 = np.eye(2)
        Ds = (5.0 / 6.0) * G[:, None, None, None] * np.broadcast_to(I2, (nel, nlk, 2, 2))
        Ds = np.ascontiguousarray(Ds)                                         # (nel, nlk, 2, 2)

        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)
        l_idx   = np.arange(nlk)
        dz = np.broadcast_to(t_arr[:, None] / nlk_arr[:, None], (nel, nlk)).copy()
        layer_mask = l_idx[None, :] < nlk_arr[:, None]
        dz = dz * layer_mask

        Dsh_kl   = np.sum(Ds * dz[:, :, None, None], axis=1)                 # (nel, 2, 2)
        Dsh_full = np.broadcast_to(Dsh_kl[:, None, None, :, :],
                                   (nel, go, go, 2, 2))                       # (nel, go, go, 2, 2)

        # 2 NN prediction for 6x6 D matrix ----------------------------------------------------
        n_tot = nel * go * go
        k_idx = np.repeat(np.arange(nel), go * go)
        ij    = np.tile(np.arange(go * go), nel)
        i_idx = ij // go
        j_idx = ij %  go
        types_arr = np.asarray(self.ELS[4])
        sentinel  = (types_arr[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)
        valid     = ~sentinel

        eh_flat  = eh.reshape(n_tot, 8)
        input_nn = eh_flat[valid, :6]                                         # (n_valid, 6)

        # Pick the model: if loaded_models is provided, use the preloaded triple; otherwise
        # fall back to the path (which make_NN_prediction will load on the spot).
        if loaded_models is not None:
            chosen_key   = self.check_range_NN(eh_flat[valid], 'eps-t', predict='D',
                                               cmk=cmk_val, return_key=True)
            chosen_model = loaded_models[chosen_key]
        else:
            chosen_model = self.check_range_NN(eh_flat[valid], 'eps-t', predict='D',
                                               cmk=cmk_val)

        mat_NN = make_NN_prediction(input_nn, predict='D', model=chosen_model, non_batchwise=True)
        D_pred_valid = np.asarray(mat_NN['D'])                                # (n_valid, 6, 6)
        self.check_range_NN(D_pred_valid, 'D', cmk=cmk_val)

        # 2b Apply filter_small_stiffness to the membrane 3x3 block (D_NN[0:3, 0:3]).
        # filter_small_stiffness only fires for specific scenarios; replicate that here.
        if scenario in (8, 9, 109, 110, 111, 112):
            # Uniform-material assumption: same Ec, t across elements.
            Ec0 = float(np.asarray(self.MATK["Ec"]).reshape(-1)[0])
            t0  = float(np.asarray(self.GEOMK["t"]).reshape(-1)[0])
            ff  = Ec0 / 10.0
            D_min_22 = t0 * (ff / 2.0)                                        # shear stiffness floor
            mask_lo = D_pred_valid[:, 2, 2] < D_min_22
            D_pred_valid[mask_lo, 2, 2] = D_min_22

        # Scatter to full (n_tot, 6, 6) and reshape to (nel, go, go, 6, 6).
        D_pred_full = np.zeros((n_tot, 6, 6))
        D_pred_full[valid] = D_pred_valid
        D_pred_struct = D_pred_full.reshape(nel, go, go, 6, 6)

        # 3 Combine 8x8 De -- NN for upper-left 6x6, numerical Dsh for [6:8, 6:8] -----------
        De = np.zeros((nel, go, go, 8, 8))
        De[..., 0:6, 0:6] = D_pred_struct
        De[..., 6:8, 6:8] = Dsh_full

        # 4 Per-element-type Ke via batched block-wise einsums -------------------------------
        Ke_dict = {}
        for n_nodes, ndof in ((4, 24), (3, 18)):
            els = np.where(types_arr == n_nodes)[0]
            E_count = len(els)
            if E_count == 0:
                continue

            gp, w = self.gauss_points(n_nodes, go)
            w_grid = np.outer(w, w).copy()
            if n_nodes == 3 and go >= 2:
                w_grid[1, 1] = 0.0

            B_batch    = np.zeros((E_count, go, go, 8, ndof))
            Jdet_batch = np.zeros((E_count, go, go))
            for idx, k in enumerate(els):
                for i in range(go):
                    for j in range(go):
                        if n_nodes == 3 and i == 1 and j == 1:
                            continue
                        Bm = B["Bm"]["r"][k][i][j]
                        Bb = B["Bb"]["r"][k][i][j]
                        Bs = B["Bs"]["r"][k][i][j]
                        B_batch[idx, i, j]    = np.vstack([Bm, Bb, Bs])
                        Jdet_batch[idx, i, j] = B["Jdet"][k][i][j]

            weight = w_grid[None, :, :] * Jdet_batch

            Bm_b = B_batch[..., 0:3, :]
            Bb_b = B_batch[..., 3:6, :]
            Bs_b = B_batch[..., 6:8, :]

            De_g   = De[els]
            Dmh_g  = De_g[..., 0:3, 0:3]
            Dmbh_g = De_g[..., 0:3, 3:6]
            Dbmh_g = De_g[..., 3:6, 0:3]
            Dbh_g  = De_g[..., 3:6, 3:6]
            Dsh_g  = De_g[..., 6:8, 6:8]

            # Block einsums matching k_k_nn_num's exact assembly:
            Kme_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmh_g,  Bm_b)
            Kbe_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bb_b, Dbh_g,  Bb_b)
            Kse_batch  = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bs_b, Dsh_g,  Bs_b)
            Kmbe_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dmbh_g, Bb_b)
            # Note: Kbme uses Bm^T ... Bb (not Bb^T ... Bm), matching the original k_k_nn_num.
            Kbme_batch = np.einsum('eij,eijpa,eijpq,eijqb->eab', weight, Bm_b, Dbmh_g, Bb_b)

            Ke_batch = Kme_batch + Kbe_batch + Kse_batch + Kmbe_batch + Kbme_batch

            # Drilling / coplanar correction (same as k_k_vec)
            A_k_arr = weight.sum(axis=(1, 2))
            for idx, k in enumerate(els):
                n_k = self.ELEMENTS[k]
                iscoplk = 1 if (n_k.any() in self.copln) else 0
                Ke_i = Ke_batch[idx]
                if iscoplk:
                    Tkr = self.rotLG(k)[1]
                    if n_nodes == 3:
                        Tkr = Tkr[:18, :18]
                    t_k = self.GEOMK["t"][k]
                    Ke_i = Ke_i + iscoplk * A_k_arr[idx] * t_k * 33600 * Tkr * 1e-8
                Ke_dict[int(k)] = Ke_i

        return Ke_dict, De



    def k_glob(self, B,e,s, cmk, go, perm = None, perm1 = None):
        """ ----------------------------------- Create global stiffness matrix ------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - u: Global displacement vector
            - e: Global strain matrix
            - cmk
                - 1 --> Linear Elasticity
                - 2 --> CMM-
                - 3 --> CMM
                - 10 --> Glas
                given per element
            - perm: if permutation according to random matrix shall occur
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - K: Global stiffness matrix
        -----------------------------------------------------------------------------------------------------------------"""
        # 0 Initiate Time Measurement
        start = time.time()

        """------------------------------------------ Calculation and Assembly ------------------------------------------"""
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        K = np.zeros((6 * nn, 6 * nn))
        D_tot = np.zeros((nk, go, go, 8, 8))
        # Ke_tot = np.zeros((nk, 1, 1, 24, 24))
        if perm1 is not None:
            # random_factor = 0.96
            random_factor = np.random.uniform(perm1[0], perm1[1])
            print('Random factor:', random_factor)
        for k in range(nk):
            """--------------------------------- Local Stiffness Matrix in Global Coordinates ---------------------------"""
            e_k = e[k,:,:,:]
            s_k = s[k]
            if perm1 is not None: 
                Ke, De = self.k_k(B["Bm"]["r"][k],B["Bb"]["r"][k],B["Bs"]["r"][k],B["Jdet"][k],e_k,s_k, k, cmk[k], go, perm = perm, perm1 = perm1, random_factor1 = random_factor)
            else:
                Ke, De = self.k_k(B["Bm"]["r"][k],B["Bb"]["r"][k],B["Bs"]["r"][k],B["Jdet"][k],e_k,s_k, k, cmk[k], go, perm = perm, perm1 = perm1)
            nodes = self.ELEMENTS[k, :][self.ELEMENTS[k, :] < 10**5]
            # Ke_tot[k][0][0][:][:] = Ke
            # Careful: Ke_tot only works for rectangular meshes, not triangular, because Ke_tot always assumes Ke of size 24x24 
            """------------------------------------ Assemble to global Stiffness Matrix ---------------------------------"""
            K = self.m_assemble(Ke, K, nodes)
            D_tot[k,:,:,:,:] = De
        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    def k_glob_vec(self, B, e, s, cmk, go):
        """
        Vectorised drop-in for k_glob (no perm / perm1 -- diagnostic-only branches dropped,
        see k_k_vec). All per-element stiffness matrices are computed in batch via k_k_vec;
        the assembly into the global K is still a per-element loop over m_assemble because
        each element's node set is different.

        Args:
            B    (dict):  B-matrix container (forwarded to k_k_vec).
            e    (np.arr): strains from find_e_vec, shape (nel, nlk, go, go, 5).
            s    (np.arr): stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
            cmk  (array-like): cm per element, shape (nel,). Restricted to all-cm=3 by k_k_vec.
            go   (int): Gauss order.

        Returns:
            K     (np.arr): global stiffness, shape (6*nn, 6*nn).
            D_tot (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8). Same content
                            as k_glob's D_tot.
        """
        start = time.time()

        NODESG = self.COORD["n"][0]
        nn  = len(NODESG[:, 0])
        nel = len(self.ELEMENTS[:, 0])

        Ke_dict, D_tot = self.k_k_vec(B, e, s, cmk, go)

        K = np.zeros((6 * nn, 6 * nn))
        for k in range(nel):
            nodes = self.ELEMENTS[k, :]
            nodes = nodes[nodes < 10**5]
            K = self.m_assemble(Ke_dict[int(k)], K, nodes)

        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    def k_glob_nn(self, B,eh,sh, cmk):
        """ ----------------------------------- Create global stiffness matrix ------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - u: Global displacement vector
            - e: Global strain matrix
            - cmk
                - 1 --> Linear Elasticity
                - 2 --> CMM-
                - 3 --> CMM
                given per element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - K: Global stiffness matrix
        -----------------------------------------------------------------------------------------------------------------"""
        # 0 Initiate Time Measurement
        start = time.time()

        """------------------------------------------ Calculation and Assembly ------------------------------------------"""
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        K = np.zeros((6 * nn, 6 * nn))
        D_tot_ = np.zeros((nk, self.gauss_order, self.gauss_order, 8, 8))
        for k in range(nk):
            """--------------------------------- Local Stiffness Matrix in Global Coordinates ---------------------------"""
            eh_k = eh[k,:,:,:]
            sh_k = sh[k,:,:,:]
            Ke, De = self.k_k_nn(B["Bm"]["r"][k],B["Bb"]["r"][k],B["Bs"]["r"][k],B["Jdet"][k],eh_k,sh_k, k, cmk[k])
            nodes = self.ELEMENTS[k, :][self.ELEMENTS[k, :] < 10**5]
            """------------------------------------ Assemble to global Stiffness Matrix ---------------------------------""" 
            K = self.m_assemble(Ke, K, nodes)
            D_tot_[k,:,:,:,:] = De
        D_tot = D_tot_
        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot
    
    def k_glob_nn_num(self, B, e, s, eh, sh, cmk, model_dim, scenario):
        """ --------------------- Create global stiffness matrix (partly NN, partly numerical)---------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - u: Global displacement vector
            - e: Global strain matrix
            - cmk
                - 1 --> Linear Elasticity
                - 2 --> CMM-
                - 3 --> CMM
                given per element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - K: Global stiffness matrix
        -----------------------------------------------------------------------------------------------------------------"""
        # 0 Initiate Time Measurement
        start = time.time()

        """------------------------------------------ Calculation and Assembly ------------------------------------------"""
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        K = np.zeros((6 * nn, 6 * nn))
        D_tot_ = np.zeros((nk, self.gauss_order, self.gauss_order, 8, 8))
        for k in range(nk):
            """--------------------------------- Local Stiffness Matrix in Global Coordinates ---------------------------"""
            eh_k = eh[k,:,:,:]
            sh_k = sh[k,:,:,:]
            e_k = e[k,:,:,:]
            s_k = s[k]
            Ke, De = self.k_k_nn_num(B["Bm"]["r"][k],B["Bb"]["r"][k],B["Bs"]["r"][k],B["Jdet"][k],eh_k,sh_k,e_k, s_k, k, cmk[k], model_dim, scenario)

            nodes = self.ELEMENTS[k, :][self.ELEMENTS[k, :] < 10**5]
            """------------------------------------ Assemble to global Stiffness Matrix ---------------------------------""" 
            K = self.m_assemble(Ke, K, nodes)
            D_tot_[k,:,:,:,:] = De
        D_tot = D_tot_
        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    def k_glob_nn_num_vec(self, B, e, s, eh, sh, cmk, go, model_dim, scenario, loaded_models=None):
        """
        Vectorised drop-in for k_glob_nn_num. Delegates per-element Ke assembly to
        k_k_nn_num_vec; the global assembly is still a per-element loop over m_assemble
        because each element's node set differs.

        Restrictions:
            * model_dim == 'THREEDIM' only (inherited from k_k_nn_num_vec).
            * cm == 3 only.

        Args:
            B    (dict):  B-matrix container (forwarded to k_k_nn_num_vec).
            e, s, eh, sh: standard FEM tensors.
            cmk  (array-like): per-element cm.
            go   (int):   Gauss order.
            model_dim (str), scenario (int): forwarded to k_k_nn_num_vec.
            loaded_models (dict|None): forwarded to k_k_nn_num_vec; if provided, the NN
                                       model is taken from this dict (key 'D') instead of
                                       being loaded from disk on every call.

        Returns:
            K     (np.arr): global stiffness, shape (6*nn, 6*nn).
            D_tot (np.arr): per-(k, i, j) 8x8 tangent, shape (nel, go, go, 8, 8).
        """
        start = time.time()

        NODESG = self.COORD["n"][0]
        nn  = len(NODESG[:, 0])
        nel = len(self.ELEMENTS[:, 0])

        Ke_dict, D_tot = self.k_k_nn_num_vec(B, e, s, eh, sh, cmk, go, model_dim, scenario,
                                             loaded_models=loaded_models)

        K = np.zeros((6 * nn, 6 * nn))
        for k in range(nel):
            nodes = self.ELEMENTS[k, :]
            nodes = nodes[nodes < 10**5]
            K = self.m_assemble(Ke_dict[int(k)], K, nodes)

        end = time.time()
        time_K._updatetime(delta_t=end - start)
        return K, D_tot


    """-------------------------------------------- Static Condensation--------------------------------------------------"""


    def c_dof(self):
        """ --------------------------------- Create Vector of condensed DOFs -------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Boundary conditions from Input file
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - cDOF --> Vector of DOFs to be condensed
        -----------------------------------------------------------------------------------------------------------------"""
        numb = len(self.BC[:,0])
        cDOF = np.array([])
        cVAL = np.array([])
        " ----------------------------------- condensed DOFs from Boundary Conditions -------------------------------------"
        for i in range(numb):
            xmin = self.BC[i,0]
            xmax = self.BC[i,1]
            ymin = self.BC[i,2]
            ymax = self.BC[i,3]
            zmin = self.BC[i,4]
            zmax = self.BC[i,5]
            nodes_i = self.find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
            if self.BC[i,6]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6)
                cVAL = np.append(cVAL, self.BC[i,6]*np.ones_like(nodes_i))
            if self.BC[i,7]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 1)
                cVAL = np.append(cVAL, self.BC[i, 7]*np.ones_like(nodes_i))
            if self.BC[i, 8]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 2)
                cVAL = np.append(cVAL, self.BC[i, 8]*np.ones_like(nodes_i))
            if self.BC[i, 9]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 3)
                cVAL = np.append(cVAL, self.BC[i, 9]*np.ones_like(nodes_i))
            if self.BC[i, 10]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 4)
                cVAL = np.append(cVAL, self.BC[i, 10]*np.ones_like(nodes_i))
            if self.BC[i, 11]!=1234:
                cDOF = np.append(cDOF, nodes_i * 6 + 5)
                cVAL = np.append(cVAL, self.BC[i, 11]*np.ones_like(nodes_i))
        " ------------------------------------------------ Sort and return ------------------------------------------------"
        indeces = np.argsort(cDOF)
        cDOF = cDOF[indeces]
        cVAL = cVAL[indeces]
        _,induni = np.unique(cDOF,return_index=True)
        cDOF = cDOF[induni]
        cVAL = cVAL[induni]
        # cDOF = np.sort(cDOF)
        # cDOF = np.unique(cDOF)
        cDOF= cDOF.astype(int)
        return cDOF,cVAL


    def v_stat_con(self, v, cDOF, cVAL):
        """ -------------------------------------- Static condensation --------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - v         --> (Force) Vector to be condensed
            - cond_DOF  --> Vector of DOFs to be condensed
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - v         --> Condensed (force) vector
        -----------------------------------------------------------------------------------------------------------------"""
        # v = np.delete(v, cDOF, 0)
        for i in range(len(cDOF)):
            v[cDOF[i]] = cVAL[i]
        return v


    def m_stat_con(self, M, cDOF):
        """ -------------------------------------- Static condensation --------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - M         --> (Stiffness) Matrix to be condensed
            - cDOF      --> Vector of DOFs to be condensed
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - M         --> Condensed (stiffness) matrix
        -----------------------------------------------------------------------------------------------------------------"""
        # M = np.delete(M, cDOF, 0)
        # M = np.delete(M, cDOF, 1)
        for i in range(len(cDOF)):
            M[cDOF[i],:] = np.zeros_like(M[cDOF[i],:])
            M[cDOF[i],cDOF[i]] = 1
        return M


    """------------------------------------------------- Assembly--------------------------------------------------------"""


    def m_assemble(self, Ke, K, nodes):
        """ ------------------ Assemble local nodal stiffness matrix to global stiffness matix --------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Ke    --> Local nodal stiffness matrix of regarded finite element
            - K     --> Global stiffness matrix
            - nodes --> Nodes of regarded element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -K: Global stiffness matrix with contributions of the regarded element
            -----------------------------------------------------------------------------------------------------------------"""

        for i in range(int(len(Ke[:, 0]) / 6)):
            for j in range(int(len(Ke[:, 0]) / 6)):
                Ke = np.array(Ke)
                K[nodes[i] * 6:nodes[i] * 6 + 6, nodes[j] * 6:nodes[j] * 6 + 6] = K[nodes[i] * 6:nodes[i] * 6 + 6,
                                                                                nodes[j] * 6:nodes[j] * 6 + 6] \
                                                                                + Ke[i * 6:i * 6 + 6, j * 6:j * 6 + 6]
        return K


    def v_assemble(self, ve, v, nodes):
        """ ------------------------------ Assemble local nodal vector to global vector --------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - ve    --> Local nodal (force) vector of regarded finite element. Column vector ndarray(n,1)
            - v     --> Global (force) vector. Column vector ndarray(n,1)
            - nodes --> Nodes of regarded element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -v: Global (force) vector with contributions of the regarded element. Column vector ndarray(n,1)
        -----------------------------------------------------------------------------------------------------------------"""
        for i in range(int(ve.size/6)):
            v[nodes[i]*6:nodes[i]*6+6] = v[nodes[i]*6:nodes[i]*6+6] + ve[i*6:i*6+6]
        return v


    def f_assemble(self, load_step, Load_el, Load_n):
        """ --------------------------------- Create Vector of applied forces -------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Applied force conditions from Input file
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - f_e --> vector of applied outer forces per DOF
        -----------------------------------------------------------------------------------------------------------------"""

        " -------------------------------------------- Element Loads ------------------------------------------------------"

        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        f_e = np.zeros((1,6*nn))
        nlel = len(Load_el[load_step][:, 0])
        for i in range(nlel):
            xmin = Load_el[load_step][i, 0]
            xmax = Load_el[load_step][i, 1]
            ymin = Load_el[load_step][i, 2]
            ymax = Load_el[load_step][i, 3]
            zmin = Load_el[load_step][i,4]
            zmax = Load_el[load_step][i,5]
            dir = int(Load_el[load_step][i, 6])

            els_i = self.find_el_range(xmin,xmax,ymin,ymax,zmin,zmax)

            q = Load_el[load_step][i, 7]
            q_e = np.zeros((6, 1))
            q_e[dir-1][0] = q

            for k in range(len(els_i)):
                el = els_i[k]
                nodes = self.find_nodes(el)
                nodes = nodes[nodes<10**5]
                J_det = 0
                q_e1 = np.zeros((6, 1))
                q_e2 = np.zeros((6, 1))
                q_e3 = np.zeros((6, 1))
                q_e4 = np.zeros((6, 1))
                for ii in range(2):
                    for jj in range(2):
                        if self.ELS[4][k] == 4:
                            J_det_ij = self.jacobi(el,ii,jj,2)[4]
                            J_det += J_det_ij
                            xi = self.gauss_points(4,2)[0][jj]
                            eta = self.gauss_points(4,2)[0][ii]
                            wxi = self.gauss_points(4,2)[1][jj]
                            weta = self.gauss_points(4,2)[1][ii]

                            n1 = 1 / 4 * (1 - xi) * (1 - eta)
                            n2 = 1 / 4 * (1 + xi) * (1 - eta)
                            n3 = 1 / 4 * (1 + xi) * (1 + eta)
                            n4 = 1 / 4 * (1 - xi) * (1 + eta)
                        else:
                            if jj == 1 and ii == 1:
                                continue
                            else:
                                J_det_ij = self.jacobi(el, ii, jj, 2)[4]
                                J_det += J_det_ij
                                xi = self.gauss_points(3,2)[0][jj]
                                eta = self.gauss_points(3,2)[0][ii]
                                wxi = self.gauss_points(3,2)[1][jj]
                                weta = self.gauss_points(3,2)[1][ii]

                                n1 = 1 - xi - eta
                                n2 = xi
                                n3 = eta
                                n4 = 0

                        N1 = np.zeros((6,6))
                        np.fill_diagonal(N1,n1)
                        N2 = np.zeros((6,6))
                        np.fill_diagonal(N2,n2)
                        N3 = np.zeros((6,6))
                        np.fill_diagonal(N3,n3)
                        N4 = np.zeros((6,6))
                        np.fill_diagonal(N4,n4)

                        q_e1 += N1@q_e*J_det_ij*wxi*weta
                        q_e2 += N2@q_e*J_det_ij*wxi*weta
                        q_e3 += N3@q_e*J_det_ij*wxi*weta
                        q_e4 += N4@q_e*J_det_ij*wxi*weta
                        q_e_all = [q_e1,q_e2,q_e3,q_e4]
                for n in range(len(nodes)):
                    node = nodes[n]
                    for vecit in range(6):
                        f_e[0][node*6+vecit] += q_e_all[n][vecit]
        " --------------------------------------------- Nodal Loads -------------------------------------------------------"
        nln = len(Load_n[load_step][:, 0])
        for i in range(nln):
            xmin = Load_n[load_step][i, 0]
            xmax = Load_n[load_step][i, 1]
            ymin = Load_n[load_step][i, 2]
            ymax = Load_n[load_step][i, 3]
            zmin = Load_n[load_step][i, 4]
            zmax = Load_n[load_step][i, 5]
            dir = Load_n[load_step][i, 6]

            nodes_i = self.find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
            # if load is not acting in any existing node: create equivalent force and moment in closest node
            if nodes_i.size == 0:
                ms = self.MATK['ms']
                step = ms/10
                count = 1
                while nodes_i.size == 0:
                    nodes_i = self.find_node_range(xmin-count*step,xmax+count*step,ymin-count*step,ymax+count*step,zmin-count*step,zmax+count*step)
                    count += 1
                if nodes_i.size > 1:
                    nodes_i = np.array([nodes_i[0]])
                coordsi = NODESG[nodes_i[0],:]
                diffcordsi = np.array([(xmin+xmax)/2, (ymin+ymax)/2,(zmin+zmax)/2])-coordsi
            else:
                diffcordsi = np.array([0,0,0])
            load_i = Load_n[load_step][i,7]
            for j in range(len(nodes_i)):
                node_j = nodes_i[j]
                coord = NODESG[node_j,:]
                node = self.find_node(coord)
                if dir == 1:
                    DOF = node*6
                elif dir == 2:
                    DOF = node*6+1
                elif dir == 3:
                    DOF = node*6+2
                elif dir == 4:
                    DOF = node*6+3
                elif dir == 5:
                    DOF = node*6+4
                elif dir == 6:
                    DOF = node*6+5
                f_e[0,DOF] = f_e[0,DOF] + load_i
            # Add bending moment for equivalent nodal force (only for offset translational loads)
                if dir in (1, 2, 3) and np.any(diffcordsi):
                    fii = np.zeros(3)
                    fii[int(dir)-1] = load_i
                    moments = np.cross(diffcordsi, fii)
                    for im in range(0, 3):
                        f_e[0, int(node*6+3+im)] += moments[im]
        f_e = np.transpose(f_e)
        return f_e


    def f0_assemble(self, B,s0,go):
        """ -------------------- Create Vector of external forces caused by internal stresses ---------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Internal Stresses (caused by shrinkage)
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - f_e0 --> vector of applied outer forces per DOF
        -----------------------------------------------------------------------------------------------------------------"""
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        sh0 = np.zeros((nk, go, go, 8))
        f_0 = np.zeros((6*nn,1))
        for k in range(0,nk):
            gp, w = self.gauss_points(self.ELS[4][k], go)
            n_k = self.find_nodes(k)
            n_k = n_k[n_k<100000]
            f0_k = np.zeros((1,6 * len(n_k)))
            for i in range(len(gp)):
                for j in range(len(gp)):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        continue
                    B_kij = np.append(np.append(B["Bm"]["r"][k][i][j],B["Bb"]["r"][k][i][j],axis=0),B["Bs"]["r"][k][i][j],axis=0)
                    sh0_kij = self.find_sh0_kij(s0[k],k,i,j)
                    sh0[k,i,j]=sh0_kij
                    f0_k = f0_k - w[i] * w[j] * B["Jdet"][k][i][j] * np.transpose(B_kij)@sh0_kij
            # print('Element Number', k)
            # print('initial stresses', sh0[k,:,:])
            f_0 = self.v_assemble(np.transpose(f0_k), f_0, n_k)
        return sh0,f_0
    
    def fh0_assemble(self, B,sh0,go):
        """ -------------------- Create Vector of external forces caused by internal stresses ---------------------------
            -------------------- Novel formulation for NN-hybrid analysis -----------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - Initial generalised stresses (output from find_sh0)
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - f_e0 --> vector of applied outer forces per DOF
        -----------------------------------------------------------------------------------------------------------------"""
        NODESG = self.COORD["n"][0]
        nn = len(NODESG[:, 0])
        nk = len(self.ELEMENTS[:, 0])
        # sh0 = np.zeros((nk, go, go, 8))
        f_0 = np.zeros((6*nn,1))
        for k in range(0,nk):
            gp, w = self.gauss_points(self.ELS[4][k], go)
            n_k = self.find_nodes(k)
            n_k = n_k[n_k<100000]
            f0_k = np.zeros((1,6 * len(n_k)))
            for i in range(len(gp)):
                for j in range(len(gp)):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        continue
                    B_kij = np.append(np.append(B["Bm"]["r"][k][i][j],B["Bb"]["r"][k][i][j],axis=0),B["Bs"]["r"][k][i][j],axis=0)
                    # sh0_kij = self.find_sh0_kij(s0[k],k,i,j)
                    # sh0[k,i,j]=sh0_kij
                    f0_k = f0_k - w[i] * w[j] * B["Jdet"][k][i][j] * np.transpose(B_kij)@sh0[k,i,j]
            f_0 = self.v_assemble(np.transpose(f0_k), f_0, n_k)
        return f_0




    
    """-------------------------------------------- Auxiliary Functions--------------------------------------------------"""

    def find_node(self, coord):
        """ -------------------------------- Find node number for given coordinates -------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - coord --> coordinates of regarded node
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - nr    --> node number of regarded node
        -----------------------------------------------------------------------------------------------------------------"""
        NODESG = self.COORD["n"][0]
        for j in range(len(NODESG[:, 0])):
            diff = abs(np.add(coord, -NODESG[j, :]))
            if max(diff) < pow(10, -10):
                nr = j
        return nr


    def find_nodes(self, el):
        """ ------------------------------------ Nodes connected to a given Element ----------------------------------------
                --------------------------------------------    INPUT: -----------------------------------------------------
                - el --> element number
                --------------------------------------------- OUTPUT:-------------------------------------------------------
                - nodes    --> vector of nodes connected to given element
        -----------------------------------------------------------------------------------------------------------------"""
        nodes = self.ELEMENTS[int(el)]
        return nodes


    def find_node_range(self, xmin, xmax, ymin, ymax, zmin, zmax):
        NODESG = self.COORD["n"][0]
        nodesx = NODESG[:,0]
        nodesy = NODESG[:,1]
        nodesz = NODESG[:,2]
        ind1 = np.array(np.where(nodesx<=xmax)).ravel()
        ind2 = np.array(np.where(nodesx>=xmin)).ravel()
        indx = np.intersect1d(ind1,ind2)

        ind1 = np.array(np.where(nodesy <= ymax)).ravel()
        ind2 = np.array(np.where(nodesy >= ymin)).ravel()
        indy = np.intersect1d(ind1, ind2)

        ind1 = np.array(np.where(nodesz <= zmax)).ravel()
        ind2 = np.array(np.where(nodesz >= zmin)).ravel()
        indz = np.intersect1d(ind1, ind2)

        indxy = np.intersect1d(indx,indy)
        ind = np.intersect1d(indxy,indz)

        return ind


    def find_el_range(self, xmin,xmax,ymin,ymax,zmin,zmax):
        centx = self.COORD["c"][0][:,0]
        centy = self.COORD["c"][0][:,1]
        centz = self.COORD["c"][0][:,2]

        ind1 = np.array(np.where(centx<=xmax)).ravel()
        ind2 = np.array(np.where(centx>=xmin)).ravel()
        indx = np.intersect1d(ind1,ind2)

        ind1 = np.array(np.where(centy <= ymax)).ravel()
        ind2 = np.array(np.where(centy >= ymin)).ravel()
        indy = np.intersect1d(ind1, ind2)

        ind1 = np.array(np.where(centz <= zmax)).ravel()
        ind2 = np.array(np.where(centz >= zmin)).ravel()
        indz = np.intersect1d(ind1, ind2)

        indxy  = np.intersect1d(indx,indy)
        ind = np.intersect1d(indxy,indz)
        return ind


    def find_el(self, node):
        """ ------------------------------------ Elements connected to a given node ----------------------------------------
                --------------------------------------------    INPUT: -----------------------------------------------------
                - node --> node number
                --------------------------------------------- OUTPUT:-------------------------------------------------------
                - ki    --> vector of elements connected to node
        -----------------------------------------------------------------------------------------------------------------"""
        els = []
        for k in range(len(self.ELEMENTS[:,1])):
            nodes = self.ELEMENTS[k,:]
            if node in nodes:
                els = np.append(els,k)
        return els


    def find_dofs_n(self, node):
        """ ----------------------------------- returns DOFs belonging to given node --------------------------------------
                --------------------------------------------    INPUT: -----------------------------------------------------
                - nodes --> node number
                --------------------------------------------- OUTPUT:-------------------------------------------------------
                - DOFS    --> vector DOFS belongin to given node
        -----------------------------------------------------------------------------------------------------------------"""
        dofs = np.zeros(6,dtype=int)
        dofs[0:6]=[node*6,node*6+1,node*6+2,node*6+3,node*6+4,node*6+5]
        return dofs


    def find_dofs_k(self, el):
        """ ----------------------------------- returns DOFs belonging to given element --------------------------------------
                --------------------------------------------    INPUT: -----------------------------------------------------
                - el --> element number
                --------------------------------------------- OUTPUT:-------------------------------------------------------
                - DOFS    --> vector DOFS belongin to given element
        -----------------------------------------------------------------------------------------------------------------"""
        nodes = self.find_nodes(el)
        nodes = nodes[nodes < 10**5]
        dofs = np.zeros(6*len(nodes),dtype=int)
        for n in range(len(nodes)):
            dofs[n*6:n*6+6] = [nodes[n] * 6, nodes[n] * 6 + 1, nodes[n] * 6 + 2, nodes[n] * 6 + 3, nodes[n] * 6 + 4, nodes[n]*6+5]
        return dofs


    def find_v_el(self, v, k):
        """ ---------------------- Calculate values of a vector at all nodes of regarded element-------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - v          --> Vector of searched values
            - el_nr      --> Element index of regarded element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - v_nodes    --> Values of v at DOFs of nodes of element with index k
        -----------------------------------------------------------------------------------------------------------------"""
        nodes = self.ELEMENTS[k, :]
        nodes = nodes[nodes < 10 ** 5]
        DOFS = [0]
        for j in range(len(nodes)):
            DOFS = np.append(DOFS, [nodes[j] * 6, nodes[j] * 6 + 1, nodes[j] * 6 + 2, nodes[j] * 6 + 3, nodes[j] * 6 + 4, nodes[j]*6+5])
        DOFS = np.delete(DOFS, 0)
        v_nodes = v[DOFS]
        return v_nodes


    def find_fi(self, B,sh):
        """ ----------------------------------- Calculate vector of inner forces-----------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - e_type
                - 1 --> Linear Elasticity
                - 2 --> CMM
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - fint  --> Vector of inner forces
        -----------------------------------------------------------------------------------------------------------------"""

        NODESG = self.COORD["n"][0]
        nel = len(self.ELEMENTS[:,0])
        fint = [np.zeros(len(NODESG[:, 0]) * 6)]
        fint = np.transpose(fint)
        for k in range(nel):
            nodes = self.ELEMENTS[k, :]
            nodes = nodes[nodes < 10 ** 5]
            DOFS = np.zeros(len(nodes) * 6, int)
            for j in range(len(nodes)):
                DOFS[j * 6] = nodes[j] * 6
                DOFS[j * 6 + 1] = nodes[j] * 6 + 1
                DOFS[j * 6 + 2] = nodes[j] * 6 + 2
                DOFS[j * 6 + 3] = nodes[j] * 6 + 3
                DOFS[j * 6 + 4] = nodes[j] * 6 + 4
                DOFS[j * 6 + 5] = nodes[j] * 6 + 5

            gp, w = self.gauss_points(self.ELS[4][k], self.gauss_order)
            fint_e = np.zeros((self.ELS[4][k]*6,1))
            for i in range(len(gp)):
                for j in range(len(gp)):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        continue
                    sh_kij = np.array(sh[k][i][j][:]).transpose()
                    sh_kij = np.ndarray.reshape(sh_kij,8,1)
                    # [Bm, Bb, Bs, Jdet] = b_kij(k, i, j, gauss_order, rot=True)
                    Bm = B["Bm"]["r"][k][i][j]
                    Bb = B["Bb"]["r"][k][i][j]
                    Bs = B["Bs"]["r"][k][i][j]
                    Jdet = B["Jdet"][k][i][j]
                    B_kij = np.append(np.append(Bm,Bb,axis=0),Bs,axis=0)
                    fint_e = np.add(fint_e, w[i] * w[j] * Jdet * np.transpose(B_kij)@sh_kij)
            fint = self.v_assemble(fint_e, fint, nodes)

        return fint


    """----------------------------------------------- Find Strains -----------------------------------------------------"""


    def find_eh_kij(self, Bm_kij,Bb_kij,Bs_kij,u_k,k,i,j,go):
        # from numpy import dot
        # from numpy.linalg import norm
        #
        # # 0 Initiate Time Measurement
        # start = time.time()
        # # 1 Values of Importance
        #
        # # 1.1 Nodes of element k
        # e_k = ELEMENTS[k, :]
        # e_k = e_k[e_k < 10 ** 5]
        #
        # # 1.2 Area of element k
        # a_k = GEOMK["ak"][k]
        #
        # # 1.3 Local coordinates of nodes of element k
        # NODESL = COORD["n"][2][a_k]
        # v = np.array(NODESL[e_k])
        #
        # def vcos(a, b):
        #     cos_ab = dot(a, b) / (norm(a) * norm(b))
        #     return cos_ab
        #
        # n1 = v[1, :]
        # n2 = v[2, :]
        # x = n2 - n1
        # phi = acos(vcos(x, [1, 0]))

        Tk = self.rotLG(k)[0]
        if self.ELS[4][k] == 3:
            Tk = Tk[0:18,0:18]
        u_k = Tk@u_k

        emh_kij = np.matmul(Bm_kij, u_k)
        ebh_kij = np.matmul(Bb_kij, u_k)
        esh_kij = np.matmul(Bs_kij, u_k)
        # esh_kij = np.matmul(Bs_kij, np.linalg.inv(Tk) @ u_k)
        # esh_kij = np.array([[1,1],[-1,1]])@esh_kij
        # esh_kij = np.array([[cos(phi),sin(phi)],[-sin(phi),cos(phi)]])@esh_kij
        eh_kij = np.append(np.append(emh_kij, ebh_kij, axis=0), esh_kij, axis=0)
        # eh_kij[abs(eh_kij) < 10**-9] = 10**-9
        # eh_kij[abs(eh_kij) < 10**-7] = 10**-7
        return eh_kij


    def find_eh(self, B,u,go):
        # 0 Initiate Time Measurement
        start = time.time()

        num_elements = len(self.ELEMENTS[:, 0])
        eh = np.zeros((num_elements, go, go, 8))
        for k in range(num_elements):
            u_k = self.find_v_el(u, k)
            for i in range(go):
                for j in range(go):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        eh[k][i][j][:] = -10**5*np.ones_like(eh[k][i][j][:])
                    else:
                        eh_kij = self.find_eh_kij(B["Bm"]["nr"][k][i][j],B["Bb"]["nr"][k][i][j],B["Bs"]["nr"][k][i][j],u_k, k, i, j, go)
                        eh[k][i][j][:] = np.transpose(eh_kij)
        end = time.time()
        time_eh._updatetime(delta_t=end - start)
        return eh


    def find_eh_vec(self, B, u, go):
        """
        Vectorised version of find_eh.

        Args:
            B   (dict): B-matrix container (uses B["Bm"]["nr"], B["Bb"]["nr"], B["Bs"]["nr"])
            u   (np.arr): global displacement vector
            go  (int): Gauss order

        Returns:
            eh  (np.arr): generalised strains at Gauss points, shape (nel, go, go, 8)

        Strategy: group elements by type (4-node quads / 3-node tris), assemble per-element local
        displacement u_k, rotation Tk, and stacked B matrix B_kij = vstack(Bm, Bb, Bs) for each
        Gauss point. Then `eh_kij = B_kij @ Tk @ u_k` is evaluated as two einsum contractions.
        Triangle elements get the -1e5 sentinel at the (i=1, j=1) Gauss slot, matching find_eh.
        """
        start = time.time()

        nel = len(self.ELEMENTS[:, 0])
        eh = np.zeros((nel, go, go, 8))
        types = np.asarray(self.ELS[4])

        for n_nodes, ndof in ((4, 24), (3, 18)):
            els = np.where(types == n_nodes)[0]
            E = len(els)
            if E == 0:
                continue

            # Per-element local displacement: (E, ndof). find_v_el can return either (ndof,)
            # or (ndof, 1) depending on whether u is 1-D or a column vector; ravel handles both.
            u_e = np.empty((E, ndof))
            for idx, k in enumerate(els):
                u_e[idx] = np.asarray(self.find_v_el(u, k)).ravel()

            # Per-element rotation Tk: (E, ndof, ndof). For tris use the 18x18 block.
            Tk = np.empty((E, ndof, ndof))
            for idx, k in enumerate(els):
                Tk_full = self.rotLG(k)[0]
                Tk[idx] = Tk_full[:ndof, :ndof] if n_nodes == 3 else Tk_full

            # Rotated local displacement: (E, ndof)
            u_rot = np.einsum('eij,ej->ei', Tk, u_e)

            # Stacked B matrix per Gauss point: (E, go, go, 8, ndof). The (i=1, j=1) slot stays
            # zero for triangles; we overwrite it with the sentinel below.
            B_batch = np.zeros((E, go, go, 8, ndof))
            for idx, k in enumerate(els):
                for i in range(go):
                    for j in range(go):
                        if n_nodes == 3 and i == 1 and j == 1:
                            continue
                        Bm = B["Bm"]["nr"][k][i][j]
                        Bb = B["Bb"]["nr"][k][i][j]
                        Bs = B["Bs"]["nr"][k][i][j]
                        B_batch[idx, i, j] = np.concatenate([Bm, Bb, Bs], axis=0)

            # eh_kij = B_kij @ u_rot : (E, go, go, 8)
            eh_e = np.einsum('eijkl,el->eijk', B_batch, u_rot)
            eh[els] = eh_e

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (types == 3)
            eh[tri_mask, 1, 1, :] = -10**5

        end = time.time()
        time_eh._updatetime(delta_t=end - start)
        return eh


    def find_sh(self, s,go):
        # 0 Initiate Time Measurement
        start = time.time()

        num_elements = len(self.ELEMENTS[:, 0])
        sh = np.zeros((num_elements, go, go, 8))
        for k in range(num_elements):
            nlk = self.GEOMK["nlk"][k]
            t_k = self.GEOMK["t"][k]
            cm_k = self.MATK["cm"][k]
            if cm_k == 10:
                t = np.zeros((1, nlk))
                nlk = self.GEOMK["nlk"][k]
                t_k1, t_k2 = self.GEOMK["t"][k], self.GEOMK["t2"][k]
                t_k_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
                for l_ in range(nlk):
                    if l_%2 == 0:
                        t[0,l_] = t_k1
                    else: 
                        t[0,l_] = t_k2
                t_cum = np.cumsum(t)


            for i in range(go):
                for j in range(go):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        sh[k][i][j][:] = -10**5*np.ones_like(sh[k][i][j][:])
                    else:
                        sh_kij = np.zeros((8,1))
                        for l in range(nlk):
                            st = s[k][l][i][j][0]
                            s_klij = np.array([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real])
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
        end = time.time()
        time_sh._updatetime(delta_t=end - start)
        return sh

    def find_sh_vec(self, s, go):
        """
        Vectorised drop-in for find_sh. Same signature as find_sh. Supports varying meshes
        (per-element thickness, layer count, cm) and any Gauss order.

        Args:
            s   (np.arr): layer stresses from find_s_vec, shape (nel, nlk, go, go, 3, 5).
                          Only s[..., 0, :].real is used (perturbation index 0, real part
                          = unperturbed stress; the complex-step perturbations are only
                          needed downstream for D, not for assembling sh).
            go  (int):    Gauss order

        Returns:
            sh  (np.arr): generalised stresses, shape (nel, go, go, 8), float

        Per (k, i, j), integrates the 5-component layer stress across thickness:
            sh[k, i, j] = sum_l S[k, l]^T @ s_real[k, l, i, j, :] * dz[k, l]
        where S[k, l] is the same lamina matrix used in find_e_klij_vec (5 x 8). Per-element
        thickness, layer count, and the cm_k == 10 alternating-thickness layout are honoured;
        layers beyond each element's nlk_k are masked out. Triangle (i=1, j=1) sentinel
        applied via boolean masking for go >= 2.
        """
        # t0 = time.perf_counter()

        nel     = len(self.ELEMENTS[:, 0])
        nlk_max = max(self.GEOMK["nlk"])

        # --- z[k, l] and dz[k, l] ------------------------------------------------------------
        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)             # (nel,)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)               # (nel,)
        l_idx   = np.arange(nlk_max)                                      # (nlk_max,)

        # Standard uniform-thickness layout (find_sh's default branch).
        z_uniform  = (-t_arr[:, None] / 2.0
                      + (2 * l_idx[None, :] + 1) * t_arr[:, None] / (2.0 * nlk_arr[:, None]))   # (nel, nlk_max)
        dz_uniform = np.broadcast_to(t_arr[:, None] / nlk_arr[:, None],
                                     (nel, nlk_max)).copy()                                    # (nel, nlk_max)

        # cm_k == 10 alternating-thickness layout (find_sh's special branch).
        cm_arr    = np.asarray(self.MATK["cm"])
        mask_cm10 = (cm_arr == 10)
        if mask_cm10.any():
            t1          = np.asarray(self.GEOMK["t"],  dtype=float)       # (nel,)
            t2          = np.asarray(self.GEOMK["t2"], dtype=float)       # (nel,)
            t_per_layer = np.where(l_idx[None, :] % 2 == 0,
                                   t1[:, None], t2[:, None])              # (nel, nlk_max)
            half        = nlk_arr // 2
            t_k_tot     = (half + 1) * t1 + half * t2                     # (nel,)
            t_cum       = np.cumsum(t_per_layer, axis=1)                  # (nel, nlk_max)
            z_cm10      = -t_k_tot[:, None] / 2.0 + (t_cum - 0.5 * t_per_layer)
            z           = np.where(mask_cm10[:, None], z_cm10, z_uniform)
            dz          = np.where(mask_cm10[:, None], t_per_layer, dz_uniform)
        else:
            z  = z_uniform
            dz = dz_uniform

        # Mask out layers beyond each element's nlk_k -- find_sh integrates only over the
        # element's actual layers, not up to nlk_max.
        layer_mask = l_idx[None, :] < nlk_arr[:, None]                   # (nel, nlk_max), bool
        dz = dz * layer_mask

        # --- S[k, l] of shape (5, 8) (same lamina matrix as find_e_klij_vec) ----------------
        S_const = np.zeros((5, 8))
        S_const[0, 0] = 1.0
        S_const[1, 1] = 1.0
        S_const[2, 2] = 1.0
        S_const[3, 6] = 1.0
        S_const[4, 7] = 1.0
        S_z = np.zeros((5, 8))
        S_z[0, 3] = -1.0
        S_z[1, 4] = -1.0
        S_z[2, 5] = -1.0
        S = S_const + z[..., None, None] * S_z                            # (nel, nlk_max, 5, 8)

        # --- Unperturbed real-part layer stresses ------------------------------------------
        s_real = s[..., 0, :].real                                        # (nel, nlk_max, go, go, 5)

        # --- Integrate over layers ----------------------------------------------------------
        # sh[k, i, j, b] = sum_l sum_a S[k, l, a, b] * s_real[k, l, i, j, a] * dz[k, l]
        sh = np.einsum('klab,klija,kl->kijb', S, s_real, dz)              # (nel, go, go, 8)

        # --- Triangle sentinel at (i=1, j=1) for go >= 2 -----------------------------------
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            sh[tri_mask, 1, 1, :] = -10**5

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated generalised stresses sh in {t1/60:.2f} min.')
        return sh

    def find_sh_nn(self, eh,go, model_dim):
        # 0 Initiate Time Measurement
        start = time.time()

        num_elements = len(self.ELEMENTS[:, 0])
        sh = np.zeros((num_elements, go, go, 8))
        helper_t = self.GEOMK['t'][0]*np.ones((num_elements, 1))
        
        for k in range(num_elements):
            # nlk = self.GEOMK["nlk"][k]
            t_k = self.GEOMK["t"][k]
            t_k2 = self.GEOMK["t2"][k]
            nlk = self.GEOMK["nlk"][k]
            for i in range(go):
                for j in range(go):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        sh[k][i][j][:] = -10**5*np.ones_like(sh[k][i][j][:])
                    else:
                        sh_kij = np.zeros((8,1))
                        eh_flat = np.reshape(eh[k,i,j,:], (1,8))
                        chosen_model_path = self.check_range_NN(np.concatenate((eh_flat, helper_t[k,:].reshape((-1,1))), axis = 1),'eps-t', cmk = self.MATK["cm"][0])
                        if self.MATK["cm"][k] == 10:
                            raise Warning('Predicting stresses for the glass scenario is outdated. Please check at next use.')
                            input_j = np.concatenate((np.array(eh_flat), [[t_k]], [[t_k2]], [[nlk]]), axis = 1)
                            input_j = transf_units(input_j, 'eps-t', forward = True)
                        elif self.MATK["cm"][k] == 1: 
                            raise Warning('Predicting stresses for the linear elastic scenario is outdated. Please check at next use.')
                            input_j = np.concatenate((np.array(eh_flat), [[t_k]]), axis = 1)
                            input_j = transf_units(input_j, 'eps-t', forward = True)
                        elif self.MATK["cm"][k] == 3:
                            index = np.where(np.array(dict_CC['Ec'], dtype = int)==int(self.MATK['Ec'][k]))[0]
                            CC = dict_CC['CC'][int(index)]
                            # input_j = np.concatenate((np.array(eh_flat), [[t_k]], [[self.GEOMK['rhox'][k][0]]], [[self.GEOMK['rhoy'][k][0]]], [[CC]]), axis = 1)
                            input_j = np.array(eh_flat[:,:6])
                        if k == 0 and i == 0 and j == 0:
                            # mat_NN = predict_sig_D(input_j, chosen_data_path, chosen_model_path, 'train', transf_type ='st-stitched', predict = 'sig', sc=False, model_dim = model_dim)
                            mat_NN = make_NN_prediction(input_j, predict = 'sig', model_path = chosen_model_path)
                        else: 
                            with HiddenPrints():
                                mat_NN = make_NN_prediction(input_j, predict = 'sig', model_path = chosen_model_path)
                                # mat_NN = predict_sig_D(input_j, chosen_data_path, chosen_model_path, 'train', transf_type ='st-stitched', predict = 'sig', sc=False, model_dim = model_dim)
                        sig_pred = mat_NN['sig']
                        sh[k][i][j][:6] = sig_pred.copy()
                        if self.MATK["cm"][k] == 10 or self.MATK["cm"][k] == 1:
                            raise Warning('Predicting stresses for the linear elastic scenario is outdated. Please check at next use.')
                            sh[k][i][j][:] = transf_units(sig_pred, 'sig', forward = False)
                        # check whether predicted sig is in range of sig_train
                        self.check_range_NN(sig_pred,'sig', cmk = self.MATK["cm"][k])
            # print('Element Number', k)
            # print('Generalised Strains', eh_flat)
            # print('Generalised Stresses (pred)', sh[k])
        end = time.time()
        time_sh._updatetime(delta_t=end - start)
        return sh

    def find_sh_nn_vec(self, eh, go, model_dim, loaded_models=None):
        """
        Vectorised drop-in for find_sh_nn. Same signature plus an optional `loaded_models`
        dict for skipping disk-loading of the NN on every call. One batched NN call replaces
        the triple-nested (k, i, j) loop.

        Args:
            eh             (np.arr): generalised strains from find_eh / find_eh_vec,
                                     shape (nel, go, go, 8)
            go             (int):    Gauss order
            model_dim:               accepted for signature parity with find_sh_nn (only used in
                                     commented-out predict_sig_D path of the original)
            loaded_models  (dict|None): {'sig_I': triple, 'sig_II': triple, 'sig_III': triple}
                                     where each triple is (inp, test_model, stats) from
                                     load_NN_model. If None, the model is loaded from disk on
                                     the fly via self.model_path (slow path, kept for back-compat).

        Returns:
            sh         (np.arr): generalised stresses, shape (nel, go, go, 8). The NN predicts
                                 the first 6 components (membrane + bending); the last 2
                                 (qx, qy) stay zero, matching find_sh_nn. Triangle (i=1, j=1)
                                 slots set to -1e5 for go >= 2.

        Restrictions (mirror find_sh_nn's active paths):
            * All elements must have cm == 3 (RC concrete). cm == 1 and cm == 10 raise Warning
              in find_sh_nn ("outdated, please check at next use") and are not exercised here.

        """
        start = time.time()

        nel = len(self.ELEMENTS[:, 0])
        sh  = np.zeros((nel, go, go, 8))

        cm_arr = np.asarray(self.MATK["cm"])
        if (cm_arr != 3).any():
            raise Warning(
                'find_sh_nn_vec: only cm == 3 (RC concrete) is supported. '
                'cm == 1 and cm == 10 paths raise Warning in find_sh_nn (outdated).'
            )

        # Sample-to-element mapping (n = k*go*go + i*go + j)
        n_tot = nel * go * go
        k_idx = np.repeat(np.arange(nel), go * go)
        ij    = np.tile(np.arange(go * go), nel)
        i_idx = ij // go
        j_idx = ij %  go

        # Triangle (i=1, j=1) sentinel positions
        types    = np.asarray(self.ELS[4])
        sentinel = (types[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)        # (n_tot,)
        valid    = ~sentinel

        # Flatten eh: (nel, go, go, 8) -> (n_tot, 8)
        eh_flat = eh.reshape(n_tot, 8)

        # Build the (eh, t) range-check input for valid samples only. helper_t uses
        # GEOMK['t'][0] for every sample, matching find_sh_nn's helper_t = t[0]*ones.
        helper_t    = self.GEOMK['t'][0] * np.ones((n_tot, 1))
        range_input = np.concatenate((eh_flat[valid], helper_t[valid]), axis=1)  # (n_valid, 9)
        cmk         = int(cm_arr.reshape(-1)[0])

        # Pick the model: if loaded_models is provided, use the preloaded triple; otherwise
        # fall back to the path (which make_NN_prediction will load on the spot).
        if loaded_models is not None:
            chosen_key   = self.check_range_NN(range_input, 'eps-t', cmk=cmk, return_key=True)
            chosen_model = loaded_models[chosen_key]
        else:
            chosen_model = self.check_range_NN(range_input, 'eps-t', cmk=cmk)

        # Single batched NN call on valid samples (cm == 3 path: first 6 components only).
        input_nn = eh_flat[valid, :6]                                       # (n_valid, 6)
        mat_NN   = make_NN_prediction(input_nn, predict='sig', model=chosen_model)
        sig_pred = mat_NN['sig']                                            # (n_valid, 6)

        # Output-side range check (same call shape as the original's per-sample post-check).
        self.check_range_NN(sig_pred, 'sig', cmk=cmk)

        # Scatter predictions back into the full (n_tot, 6) layout, then reshape to structured.
        sig_full = np.zeros((n_tot, 6))
        sig_full[valid] = sig_pred
        sh[..., :6] = sig_full.reshape(nel, go, go, 6)

        # Out-of-plane shear forces (qx, qy) -- analytical, matching ConstitutiveLaws.sigma_shear:
        #   txz_layer = 5/6 * G * gxz,   tyz_layer = 5/6 * G * gyz   (G = Ec / (2*(1+vc)))
        # gxz, gyz are constant through thickness (no z-dependence in find_e_klij's S), so the
        # thickness integral is just (5/6 * G * gxz) * t for qx (and analogously for qy).
        Ec    = np.asarray(self.MATK["Ec"], dtype=float)                # (nel,)
        vc    = np.asarray(self.MATK["vc"], dtype=float)                # (nel,)
        G     = Ec / (2.0 * (1.0 + vc))                                  # (nel,)
        t_arr = np.asarray(self.GEOMK["t"], dtype=float)                 # (nel,)
        coeff = (5.0 / 6.0) * G * t_arr                                  # (nel,) = (5/6) * G * t
        sh[..., 6] = coeff[:, None, None] * eh[..., 6]                   # qx
        sh[..., 7] = coeff[:, None, None] * eh[..., 7]                   # qy

        # Triangle (i=1, j=1) sentinel for go >= 2 (overwrites all 8 components, including
        # the qx, qy we just wrote -- matches find_sh_nn's behaviour).
        if go >= 2:
            tri_mask = (types == 3)
            sh[tri_mask, 1, 1, :] = -10**5

        end = time.time()
        time_sh._updatetime(delta_t=end - start)
        return sh

    def find_e_klij(self, eh_kij, k, l, i, j):
        """ ------------------------------------------- Shell Elements --------------------------------------------------
            --------------------------- Find strains in element k in layer l and gauss point (i,j) -------------------"""
        
        t_k = self.GEOMK["t"][k]
        cm_k = self.MATK["cm"][k]
        nlk = self.GEOMK["nlk"][k]
        z = -t_k/2+(2*l+1)*t_k/(2 * nlk)
        
        if cm_k == 10:
            t = np.zeros((1,nlk))
            t_k1, t_k2 = self.GEOMK["t"][k], self.GEOMK["t2"][k]
            t_k_tot = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
            for l_ in range(nlk):
                if l_%2 == 0:
                    t[0,l_] = t_k1
                else: 
                    t[0,l_] = t_k2
                t_cum = np.cumsum(t)
            if l == 0:
                z = -t_k_tot/2 + 0.5*t[0,l]
            else: 
                z = -t_k_tot/2 + t_cum[l-1] + 0.5*t[0,l]
        
        S = np.array(   [[1, 0, 0, -z, 0, 0, 0, 0],
                        [0, 1, 0, 0, -z, 0, 0, 0],
                        [0, 0, 1, 0, 0, -z, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])

        e_klij = np.matmul(S,eh_kij)
        # e_klij[e_klij == 0] = 10**-13
        
        # # control sizes of strains in x, y:
        # if abs(e_klij[0])<0.5e-6: 
        #     e_klij[0] = -0.5e-6
        # if abs(e_klij[1])<0.5e-6:
        #     e_klij[1] = -0.5e-6
        # if abs(e_klij[2])<0.5e-7:
        #     e_klij[2] = -0.5e-7

        return e_klij


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
        # 0 Initiate Time Measurement
        start = time.time()

        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])
        e = np.zeros((nel, nlk, go, go, 5))
        ex = np.zeros((nel, nlk, go, go))
        ey = np.zeros((nel, nlk, go, go))
        gxy = np.zeros((nel, nlk, go, go))
        e1 = np.zeros((nel, nlk, go, go))
        e3 = np.zeros((nel, nlk, go, go))
        th = np.zeros((nel, nlk, go, go))
        for k in range(nel):
            for l in range(nlk):
                for i in range(go):
                    for j in range(go):
                        if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            e[k][l][i][j][:] = -10**5*np.ones_like(e[k][l][i][j][:])
                            ex[k][l][i][j] = -10**5*np.ones_like(ex[k][l][i][j])
                            ey[k][l][i][j] = -10**5*np.ones_like(ey[k][l][i][j])
                            gxy[k][l][i][j] = -10**5*np.ones_like(gxy[k][l][i][j])
                            e1[k][l][i][j] = -10**5*np.ones_like(e1[k][l][i][j])
                            e3[k][l][i][j] = -10**5*np.ones_like(e3[k][l][i][j])
                            th[k][l][i][j] = -10**5*np.ones_like(th[k][l][i][j])
                        else:
                            eh_kij = eh[k, i, j, :]
                            e_klij = self.find_e_klij(eh_kij, k, l, i, j)
                            e0_klij = e0[k][l][i][j][:]
                            e[k][l][i][j][:]=np.transpose(e_klij)-np.transpose(e0_klij)
                            ex[k][l][i][j] = e_klij[0]-e0_klij[0]
                            ey[k][l][i][j] = e_klij[1]-e0_klij[1]
                            gxy[k][l][i][j] = e_klij[2]-e0_klij[2]
                            # gxy[k][l][i][j] = eh_kij[6]
                            [e1_klij, e3_klij, th_klij] = e_principal(ex[k][l][i][j], ey[k][l][i][j], gxy[k][l][i][j])
                            e1[k][l][i][j] = e1_klij
                            e3[k][l][i][j] = e3_klij
                            th[k][l][i][j] = th_klij
        end = time.time()
        time_strain._updatetime(delta_t=end - start)
        return e, ex, ey, gxy, e1, e3, th


    def find_e_klij_vec(self, eh):
        """
        Vectorised version of find_e_klij. Returns the layer strain S @ eh for every
        (element, layer, gauss-point) tuple in one shot, with no e0 subtraction (mirrors
        find_e_klij, which also doesn't subtract e0).

        Args:
            eh  (np.arr): generalised strains, shape (nel, go, go, 8)

        Returns:
            e_klij  (np.arr): layer strain components, shape (nel, nlk, go, go, 5)

        Builds the per-(k, l) lamina matrix S as S_const + z[k, l] * S_z and contracts it
        against eh via einsum. The cm_k == 10 variable-thickness layout (alternating t1/t2
        per layer) from find_e_klij's special branch is reproduced via np.where on z.
        """
        # t0 = time.perf_counter()
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        # --- z[k, l] : midplane offset of layer l in element k --------------------------------
        t_arr   = np.asarray(self.GEOMK["t"],   dtype=float)               # (nel,)
        nlk_arr = np.asarray(self.GEOMK["nlk"], dtype=int)                 # (nel,)
        l_idx   = np.arange(nlk)                                            # (nlk,)

        # Standard uniform-thickness layout (find_e_klij's default branch).
        z_uniform = (-t_arr[:, None] / 2.0
                     + (2 * l_idx[None, :] + 1) * t_arr[:, None] / (2.0 * nlk_arr[:, None]))   # (nel, nlk)

        # cm_k == 10: alternating t1/t2 per layer (find_e_klij's special branch).
        cm_arr    = np.asarray(self.MATK["cm"])
        mask_cm10 = (cm_arr == 10)
        if mask_cm10.any():
            t1          = np.asarray(self.GEOMK["t"],  dtype=float)         # (nel,)
            t2          = np.asarray(self.GEOMK["t2"], dtype=float)         # (nel,)
            t_per_layer = np.where(l_idx[None, :] % 2 == 0,
                                   t1[:, None], t2[:, None])                 # (nel, nlk)
            half        = nlk_arr // 2
            t_k_tot     = (half + 1) * t1 + half * t2                        # (nel,)
            t_cum       = np.cumsum(t_per_layer, axis=1)                     # (nel, nlk)
            z_cm10      = -t_k_tot[:, None] / 2.0 + (t_cum - 0.5 * t_per_layer)
            z           = np.where(mask_cm10[:, None], z_cm10, z_uniform)
        else:
            z = z_uniform

        # --- S[k, l] of shape (5, 8) ----------------------------------------------------------
        # S = [[1, 0, 0, -z,  0,  0, 0, 0],
        #      [0, 1, 0,  0, -z,  0, 0, 0],
        #      [0, 0, 1,  0,  0, -z, 0, 0],
        #      [0, 0, 0,  0,  0,  0, 1, 0],
        #      [0, 0, 0,  0,  0,  0, 0, 1]]
        S_const = np.zeros((5, 8))
        S_const[0, 0] = 1.0
        S_const[1, 1] = 1.0
        S_const[2, 2] = 1.0
        S_const[3, 6] = 1.0
        S_const[4, 7] = 1.0
        S_z = np.zeros((5, 8))
        S_z[0, 3] = -1.0
        S_z[1, 4] = -1.0
        S_z[2, 5] = -1.0
        S = S_const + z[..., None, None] * S_z                              # (nel, nlk, 5, 8)

        # S (k, l, a, b) x eh (k, i, j, b) -> (k, l, i, j, a)
        e_klij = np.einsum('klab,kijb->klija', S, eh)                       # (nel, nlk, go, go, 5)

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated layer strains e_klij in {t1/60:.2f} min.')
        return e_klij


    def find_e_vec(self, e0, eh, go):
        """
        Vectorised version of find_e. Drop-in replacement: same args / return as find_e.
        Works for any Gauss order; in particular both go = 1 and go = 2.

        Args:
            e0  (np.arr): initial strains, shape (nel, nlk, go, go, 5)
            eh  (np.arr): generalised strains, shape (nel, go, go, 8)
            go  (int):    Gauss order

        Returns:
            e, ex, ey, gxy, e1, e3, th — same shapes / semantics as find_e

        Delegates the S @ eh contraction to find_e_klij_vec, subtracts e0, and computes the
        principal strain components in bulk. Triangle (i=1, j=1) sentinel is applied via
        boolean masking, matching find_e.
        """
        # t0 = time.perf_counter()
        e_klij_all = self.find_e_klij_vec(eh)                                # (nel, nlk, go, go, 5)
        e = e_klij_all - e0

        ex  = e[..., 0]
        ey  = e[..., 1]
        gxy = e[..., 2]

        # Vectorised e_principal
        r  = 0.5 * np.sqrt((ex - ey) ** 2 + gxy ** 2)
        m  = 0.5 * (ex + ey)
        e1 = m + r
        e3 = m - r
        with np.errstate(divide='ignore', invalid='ignore'):
            th_nz = np.arctan(gxy / (2.0 * (e1 - ex)))
        th_z = np.where(ex > ey, np.pi / 2,
               np.where(ex < ey, 0.0, np.pi / 4))
        th   = np.where(np.abs(gxy) > 0, th_nz, th_z)

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            SENT = -10**5
            e  [tri_mask, :, 1, 1, :] = SENT
            ex [tri_mask, :, 1, 1]    = SENT
            ey [tri_mask, :, 1, 1]    = SENT
            gxy[tri_mask, :, 1, 1]    = SENT
            e1 [tri_mask, :, 1, 1]    = SENT
            e3 [tri_mask, :, 1, 1]    = SENT
            th [tri_mask, :, 1, 1]    = SENT

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated strains e, principals in {t1/60:.2f} min.')
        return e, ex, ey, gxy, e1, e3, th


    def find_e0(self, go):
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])
        e = np.zeros((nel, nlk, go, go, 5))
        ex = np.zeros((nel, nlk, go, go))
        ey = np.zeros((nel, nlk, go, go))
        gxy = np.zeros((nel, nlk, go, go))
        e1 = np.zeros((nel, nlk, go, go))
        e3 = np.zeros((nel, nlk, go, go))
        th = np.zeros((nel, nlk, go, go))
        for k in range(nel):
            for l in range(nlk):
                for i in range(go):
                    for j in range(go):
                        if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            e[k][l][i][j][:] = -10**5*np.ones_like(e[k][l][i][j][:])
                            ex[k][l][i][j] = -10**5*np.ones_like(ex[k][l][i][j])
                            ey[k][l][i][j] = -10**5*np.ones_like(ey[k][l][i][j])
                            gxy[k][l][i][j] = -10**5*np.ones_like(gxy[k][l][i][j])
                            e1[k][l][i][j] = -10**5*np.ones_like(e1[k][l][i][j])
                            e3[k][l][i][j] = -10**5*np.ones_like(e3[k][l][i][j])
                            th[k][l][i][j] = -10**5*np.ones_like(th[k][l][i][j])
                        else:
                            e_klij = np.array([1,1,0,0,0])*self.MATK["ecs"][k]
                            e[k][l][i][j][:]=np.transpose(e_klij)
                            ex[k][l][i][j] = e_klij[0]
                            ey[k][l][i][j] = e_klij[1]
                            gxy[k][l][i][j] = e_klij[2]
                            [e1_klij, e3_klij, th_klij] = e_principal(ex[k][l][i][j], ey[k][l][i][j], gxy[k][l][i][j])
                            e1[k][l][i][j] = e1_klij
                            e3[k][l][i][j] = e3_klij
                            th[k][l][i][j] = th_klij

                            # if go == 1:
                            #     print('Element number: ', k)
                            #     print('Initial generalised strains: ', e[k])

        return e, ex, ey, gxy, e1, e3, th


    def find_e0_vec(self, go):
        """
        Vectorised version of find_e0.

        Args:
            go  (int): Gauss order

        Returns:
            e   (np.arr): initial generalised strains, shape (nel, nlk, go, go, 5)
            ex  (np.arr): in-plane strain x-component,  shape (nel, nlk, go, go)
            ey  (np.arr): in-plane strain y-component,  shape (nel, nlk, go, go)
            gxy (np.arr): in-plane shear strain,        shape (nel, nlk, go, go)
            e1  (np.arr): max principal strain,         shape (nel, nlk, go, go)
            e3  (np.arr): min principal strain,         shape (nel, nlk, go, go)
            th  (np.arr): principal direction angle,    shape (nel, nlk, go, go)

        In the original find_e0, e_klij = [1,1,0,0,0]*ecs[k] is independent of layer l and Gauss point (i,j),
        so ex == ey == ecs[k] and gxy == 0 throughout. Under those conditions e_principal collapses to
        e1 = e3 = ecs[k] and th = pi/4, which lets us skip per-element calls entirely. Triangle elements get
        the -1e5 sentinel at the (i=1, j=1) Gauss slot, matching the original.
        """
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        ecs = np.asarray(self.MATK["ecs"], dtype=float)                          # (nel,)
        base = np.array([1.0, 1.0, 0.0, 0.0, 0.0])                                # (5,)

        e   = np.broadcast_to(ecs[:, None, None, None, None] * base,
                              (nel, nlk, go, go, 5)).copy()
        ex  = np.broadcast_to(ecs[:, None, None, None],
                              (nel, nlk, go, go)).copy()
        ey  = ex.copy()
        gxy = np.zeros((nel, nlk, go, go))

        e1 = ex.copy()
        e3 = ex.copy()
        th = np.full((nel, nlk, go, go), np.pi / 4)

        # Triangle sentinel at (i=1, j=1); only reachable when go >= 2.
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            SENT = -10**5
            e  [tri_mask, :, 1, 1, :] = SENT
            ex [tri_mask, :, 1, 1]    = SENT
            ey [tri_mask, :, 1, 1]    = SENT
            gxy[tri_mask, :, 1, 1]    = SENT
            e1 [tri_mask, :, 1, 1]    = SENT
            e3 [tri_mask, :, 1, 1]    = SENT
            th [tri_mask, :, 1, 1]    = SENT

        return e, ex, ey, gxy, e1, e3, th


    def find_eh0(self, go):
        '''
        Novel initialisation for NN-hybrid analysis
        '''
        nel = len(self.ELEMENTS[:, 0])
        # nlk = max(self.GEOMK["nlk"])
        eh0 = np.zeros((nel, go, go, 8))
        ehx0 = np.zeros((nel, go, go))
        ehy0 = np.zeros((nel, go, go))
        ehxy0 = np.zeros((nel, go, go))
        chix0 = np.zeros((nel, go, go))
        chiy0 = np.zeros((nel, go, go))
        chixy0 = np.zeros((nel, go, go))
        epsxz0 = np.zeros((nel, go, go))
        epsyz0 = np.zeros((nel, go, go))
        for k in range(nel):
            for i in range(go):
                for j in range(go):
                    if self.ELS[4][k] == 3 and i == 1 and j == 1:
                        eh0[k][i][j][:] = -10**5*np.ones_like(eh0[k][i][j][:])
                        ehx0[k][i][j] = -10**5*np.ones_like(ehx0[k][i][j])
                        ehy0[k][i][j] = -10**5*np.ones_like(ehy0[k][i][j])
                        ehxy0[k][i][j] = -10**5*np.ones_like(ehxy0[k][i][j])
                        chix0[k][i][j] = -10**5*np.ones_like(chix0[k][i][j])
                        chiy0[k][i][j] = -10**5*np.ones_like(chix0[k][i][j])
                        chixy0[k][i][j] = -10**5*np.ones_like(chixy0[k][i][j])
                        epsxz0[k][i][j] = -10**5*np.ones_like(epsxz0[k][i][j])
                        epsyz0[k][i][j] = -10**5*np.ones_like(epsyz0[k][i][j])

                    else:
                        '''
                        Initialise all with -10^(-5), instead of initialisation with eps_cs = 0 (as was before)
                        eps_cs is defined in mesh_gmsh_vb.py on line 2548
                        '''
                        # eh0[k][i][j][:] = -10**(-5)*np.ones_like(eh0[k][i][j][:])
                        # ehx0[k][i][j] = -10**(-5)*np.ones_like(ehx0[k][i][j])
                        # ehy0[k][i][j] = -10**(-5)*np.ones_like(ehy0[k][i][j])
                        # ehxy0[k][i][j] = -10**(-5)*np.ones_like(ehxy0[k][i][j])
                        # chix0[k][i][j] = -10**(-5)*np.ones_like(chix0[k][i][j])
                        # chiy0[k][i][j] = -10**(-5)*np.ones_like(chix0[k][i][j])
                        # chixy0[k][i][j] = -10**(-5)*np.ones_like(chixy0[k][i][j])
                        # epsxz0[k][i][j] = -10**(-5)*np.ones_like(epsxz0[k][i][j])
                        # epsyz0[k][i][j] = -10**(-5)*np.ones_like(epsyz0[k][i][j])
                        eh0[k][i][j][:] = 0*np.ones_like(eh0[k][i][j][:])
                        ehx0[k][i][j] = 0*np.ones_like(ehx0[k][i][j])
                        ehy0[k][i][j] = 0*np.ones_like(ehy0[k][i][j])
                        ehxy0[k][i][j] = 0*np.ones_like(ehxy0[k][i][j])
                        chix0[k][i][j] = 0*np.ones_like(chix0[k][i][j])
                        chiy0[k][i][j] = 0*np.ones_like(chix0[k][i][j])
                        chixy0[k][i][j] = 0*np.ones_like(chixy0[k][i][j])
                        epsxz0[k][i][j] = 0*np.ones_like(epsxz0[k][i][j])
                        epsyz0[k][i][j] = 0*np.ones_like(epsyz0[k][i][j])
                        # print('Element number: ', k)
                        # print('Initial generalised strains: ', eh0[k])
        return eh0, ehx0, ehy0, ehxy0, chix0, chiy0, chixy0, epsxz0, epsyz0



    def find_sh0_kij(self, s0_k,k,i,j):
        t_k = self.GEOMK["t"][k]
        cm_k = self.MATK["cm"]
        if cm_k == 10:
            nlk = self.GEOMK["nlk"][k]
            t_k1, t_k2 = self.GEOMK["t"][k], self.GEOMK["t2"][k]
            t_k = (int(nlk/2)+1)*t_k1 + int(nlk/2)*t_k2
        nlk = self.GEOMK["nlk"][k]
        dz_k = t_k/nlk
        for l in range(0,nlk):
            z = -t_k/2+(2*l+1)*t_k/(2 * nlk)
            S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                        [0, 1, 0, 0, -z, 0, 0, 0],
                        [0, 0, 1, 0, 0, -z, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
            s0_klij = np.append(s0_k[l][i][j],np.array([0,0]),axis=0)
            sh0_klij = np.transpose(S)@np.transpose(s0_klij)*dz_k
            if l == 0:
                sh0_kij = sh0_klij
            else:
                sh0_kij = sh0_kij + sh0_klij
        return sh0_kij

    """---------------------------------------------- Find Stresses -----------------------------------------------------"""


    def find_s(self, e, s_prev, go,dolinel = False):
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
        start = time.time()

        nel = len(self.ELEMENTS[:,0])
        nlk = max(self.GEOMK["nlk"])
        s = np.ndarray((nel, nlk, go, go,3),dtype=object)
        if isinstance(s_prev,int):
            s_prev = np.ndarray((nel, nlk, go, go,3),dtype=object)
        for k in range(nel):
            for l in range(nlk):
                for i in range(go):
                    for j in range(go):
                        if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            pass
                        else:
                            e_klij = e[k][l][i][j][:]
                            MATK = self.MATK
                            MAT = [MATK["Ec"][k],MATK["vc"][k],MATK["fcp"][k],MATK["fct"][k],MATK["ec0"][k],
                                0 * MATK["fct"][k] / MATK["Ec"][k], MATK["Dmax"][k],MATK["Esx"][k],MATK["Eshx"][k],MATK["fsyx"][k],
                                MATK["fsux"][k],MATK["Esy"][k],MATK["Eshy"][k],MATK["fsyy"][k],MATK["fsuy"][k],
                                MATK["tb0"][k],MATK["tb1"][k],MATK["Epx"][k],MATK["Epy"][k],MATK["tbp0"][k],
                                MATK["tbp1"][k],MATK["ebp1"][k],MATK["fpux"][k],MATK["fpuy"][k], 
                                MATK["Ec2"][k], MATK["vc2"][k]]
                            GEOMK = self.GEOMK
                            GEOM = [GEOMK["rhox"][k][l],GEOMK["rhoy"][k][l],GEOMK["dx"][k][l],GEOMK["dy"][k][l],
                                    GEOMK["sx"][k][l], GEOMK["sy"][k][l], GEOMK["rhopx"][k][l],GEOMK["rhopy"][k][l],
                                    GEOMK["dpx"][k][l],GEOMK["dpy"][k][l], GEOMK["t"], GEOMK["t2"], l]

                            if dolinel:
                                cm_k = 1
                            else: 
                                cm_k = MATK["cm"][k]

                            if self.it_type == 1 and MATK["cm"][k] != 10:
                                sig = stress(s_prev[k][l][i][j][0], MATK["cm"][k],l, k, i, j, MAT, GEOM)
                                sig.out(e_klij+[0.0000000000000001j,0,0,0,0])
                                s[k][l][i][j][0] = sig

                                sig = stress(s_prev[k][l][i][j][0], MATK["cm"][k],l, k, i, j, MAT, GEOM)
                                sig.out(e_klij+[0,0.0000000000000001j,0,0,0])
                                s[k][l][i][j][1] = sig

                                sig = stress(s_prev[k][l][i][j][0], MATK["cm"][k],l, k, i, j, MAT, GEOM)
                                sig.out(e_klij+[0,0,0.0000000000000001j,0,0])
                                s[k][l][i][j][2] = sig
                            elif self.it_type == 2 or MATK["cm"][k] == 10:
                                sig = stress(s_prev[k][l][i][j][0], MATK["cm"][k],l, k, i, j, MAT, GEOM)
                                sig.out(e_klij)
                                s[k][l][i][j][0] = sig
                            # for determining the maximum layer stresses, and element where it is located
                            # just for debugging, not relevant for the rest of the code
                            if (k == 0) and (l == 0) and (i == 0) and (j == 0):
                                s_max_init = sig.sy.real
                            if s[k][l][i][j][0].sy.real > s_max_init:
                                s_max = s[k][l][i][j][0].sy.real
                                k_max = k

        end = time.time()
        time_stress._updatetime(delta_t=end - start)

        return s

    def find_s_vec(self, e, s_prev, go, dolinel=False):
        """
        Vectorised drop-in for find_s. Same signature as find_s. Per-Gauss-point computation
        is preserved (every (k, i, j) becomes one sample in a flat batch of size nel*go*go
        passed to ConstitutiveLaws), then reshaped back to the structured layout matching e.

        Output:
            s.shape = (nel, nlk, go, go, 3, 5)        # complex128
                axes 0..3  : same (nel, nlk, go, go) prefix as e
                axis 4     : complex-step perturbation index (mirrors find_s's last axis)
                axis 5     : stress component (sx, sy, txy, txz, tyz)

        find_s's object-array entries (instances with .sx, .sy, .txy, .txz, .tyz attributes)
        become numerical entries indexed by axis 5. Imaginary parts of axis 4 carry the
        complex-step derivative d s / d(ex, ey, gxy) for the in-plane components; out-of-plane
        shear (txz, tyz) is linear in (gxz, gyz) and is not perturbed.

        Branching on dolinel:
            * dolinel = False (cm_klij = 3): 3 complex-step perturbations -> s[..., p, :] for p in 0..2.
            * dolinel = True  (cm_klij = 1): single linear-elastic evaluation -> s[..., 0, :] only.

        Mesh constraints (asserted upfront, since ConstitutiveLaws assumes uniform material):
            * All elements share the same MATK row (material).
            * All elements share the same per-layer GEOMK['rhox'], GEOMK['rhoy'].
            * Reinforcement is isotropic: Esx == Esy.
            * No cm_k == 10 element (ConstitutiveLaws supports cm_klij in {1, 3} only).
        Element-0 values are then used for ConstitutiveLaws. Mixed meshes need the masked-op
        rewrite in constitutive_laws.py; raise NotImplementedError on attempt.

        s_prev is accepted for signature parity with find_s; ConstitutiveLaws does not consume
        it for cm_klij in {1, 3} (no fixed-crack history), so it is otherwise unused.
        """
        # t0 = time.perf_counter()

        # 0 Setup ------------------------------------------------------------------------------
        nel   = len(self.ELEMENTS[:, 0])
        nlk   = max(self.GEOMK["nlk"])
        n_tot = nel * go * go

        # 1 Uniform-mesh assertions ------------------------------------------------------------
        def _uniform_MATK(key):
            arr = np.asarray(self.MATK[key]).reshape(-1)
            if arr.size > 1 and not np.all(arr == arr[0]):
                raise NotImplementedError(
                    f"find_s_vec: MATK[{key!r}] varies across elements; ConstitutiveLaws "
                    f"requires uniform material. Use find_s, or rewrite the masked ops in "
                    f"constitutive_laws.py to broadcast (n_tot, nl, 1) per-sample params."
                )
            return float(arr[0])

        def _uniform_GEOMK_layer(key):
            arr = np.asarray(self.GEOMK[key])
            if arr.ndim == 2 and arr.shape[0] > 1 and not np.all(arr == arr[0:1, :]):
                raise NotImplementedError(
                    f"find_s_vec: GEOMK[{key!r}] varies across elements; not supported."
                )
            return arr[0, :] if arr.ndim == 2 else arr

        cm_arr = np.asarray(self.MATK["cm"])
        if (cm_arr == 10).any():
            raise NotImplementedError(
                "find_s_vec: ConstitutiveLaws does not implement cm_k == 10; use find_s."
            )

        # 2 Decide cm_klij (linear elastic override via dolinel) ------------------------------
        cm_klij = 1 if dolinel else int(cm_arr.reshape(-1)[0])
        if cm_klij not in (1, 3):
            raise NotImplementedError(
                f"find_s_vec: cm_klij={cm_klij} not supported by ConstitutiveLaws."
            )

        # 3 Build mat_dict and constants from element 0 ---------------------------------------
        # ConstitutiveLaws stores a single steel material (Es, Esh, fsy, fsu) regardless of
        # direction (see constitutive_laws.py:43-48). Anisotropic reinforcement *ratios*
        # (rhox != rhoy) are fine and handled separately; what we forbid here is anisotropic
        # steel *material* properties.
        Esx = _uniform_MATK("Esx")
        for x_key, y_key, label in (("Esx",  "Esy",  "Es"),
                                     ("Eshx", "Eshy", "Esh"),
                                     ("fsyx", "fsyy", "fsy"),
                                     ("fsux", "fsuy", "fsu")):
            if _uniform_MATK(x_key) != _uniform_MATK(y_key):
                raise NotImplementedError(
                    f"find_s_vec: anisotropic steel material ({x_key} != {y_key}) not supported "
                    f"by ConstitutiveLaws -- it collapses both directions to a single {label!r} "
                    f"value. Note: anisotropic reinforcement *ratios* (rhox != rhoy) ARE "
                    f"supported and handled separately."
                )

        mat_dict = {
            'ect': 0.0 * _uniform_MATK("fct") / _uniform_MATK("Ec"),    # matches MAT[5] in find_s
            'ec0': _uniform_MATK("ec0"),
            'Ec':  _uniform_MATK("Ec"),
            'fct': _uniform_MATK("fct"),
            'fcp': _uniform_MATK("fcp"),
            'tb0': _uniform_MATK("tb0"),
            'tb1': _uniform_MATK("tb1"),
            'Es':  Esx,
            'Esh': _uniform_MATK("Eshx"),
            'fsy': _uniform_MATK("fsyx"),
            'fsu': _uniform_MATK("fsux"),
        }
        constants = {
            'nu':      _uniform_MATK("vc"),
            'n_layer': nlk,
            'rho_x':   _uniform_GEOMK_layer("rhox"),
            'rho_y':   _uniform_GEOMK_layer("rhoy"),
            'D':       _uniform_MATK("Dmax"),
        }

        # 4 Flatten e: (nel, nlk, go, go, 5) -> (n_tot, nlk, 5) -------------------------------
        # All 5 strain components are kept; ConstitutiveLaws uses the first 3 for the in-plane
        # constitutive law and the last 2 (gxz, gyz) for the analytical out-of-plane shear.
        # Sample order n = k*go*go + i*go + j.
        e_in   = np.transpose(e, (0, 2, 3, 1, 4))                            # (nel, go, go, nlk, 5)
        e_flat = e_in.reshape(n_tot, nlk, 5)                                  # (n_tot, nlk, 5)

        # Identify triangle (i=1, j=1) sentinel samples: find_e_vec writes -1e5 there. We must
        # exclude them from ConstitutiveLaws -- otherwise the garbage strain feeds into divides
        # and emits "invalid value encountered in divide" warnings. The scalar pipeline skips
        # those samples via an if-branch in find_s, so it never sees them either.
        if go >= 2:
            k_idx    = np.repeat(np.arange(nel), go * go)
            ij       = np.tile(np.arange(go * go), nel)
            i_idx    = ij // go
            j_idx    = ij %  go
            types    = np.asarray(self.ELS[4])
            sentinel = (types[k_idx] == 3) & (i_idx == 1) & (j_idx == 1)
        else:
            sentinel = np.zeros(n_tot, dtype=bool)
        valid   = ~sentinel
        e_valid = e_flat[valid]                                               # (n_valid, nlk, 5)

        # 5 Per-Gauss-point evaluation via ConstitutiveLaws -----------------------------------
        s_flat = np.zeros((n_tot, nlk, 3, 5), dtype=np.complex128)
        EPS_J = 1e-16j

        # Suppress masked-out divide-by-zero / 0/0 warnings inherent to vectorised constitutive
        # code: ConstitutiveLaws uses `np.where(cond, a/b, fallback)` patterns where numpy
        # evaluates a/b even on the False branch (and the result is discarded). The scalar
        # `stress` class avoids this by using Python if/else, so the unvectorised pipeline
        # doesn't emit the warning.
        with np.errstate(invalid='ignore', divide='ignore'):
            if cm_klij == 3:
                # 3 complex-step perturbations -> imag(s[..., p, :]) is d s / d e_p for p in {0,1,2}
                # Only the first 3 strain components (ex, ey, gxy) are perturbed -- matches find_s.
                for p, pert in enumerate([(EPS_J, 0, 0, 0, 0),
                                           (0, EPS_J, 0, 0, 0),
                                           (0, 0, EPS_J, 0, 0)]):
                    e_p = e_valid + np.array(pert, dtype=np.complex128)
                    cl  = ConstitutiveLaws(e_p, constants, mat_dict, cm_klij=cm_klij)
                    s_flat[valid, :, p, :] = cl.out().squeeze(-1)             # (n_valid, nlk, 5)
            else:  # cm_klij == 1, linear elastic: single eval, only slot 0
                cl = ConstitutiveLaws(e_valid, constants, mat_dict, cm_klij=cm_klij)
                s_flat[valid, :, 0, :] = cl.out().squeeze(-1).astype(np.complex128)

        # 6 Reshape back to structured (nel, nlk, go, go, 3, 5) -------------------------------
        s = s_flat.reshape(nel, go, go, nlk, 3, 5)
        s = np.transpose(s, (0, 3, 1, 2, 4, 5))                              # (nel, nlk, go, go, 3, 5)
        s = np.ascontiguousarray(s)

        # 7 Triangle sentinel at (i=1, j=1) — only reachable for go >= 2 ----------------------
        if go >= 2:
            tri_mask = (np.asarray(self.ELS[4]) == 3)
            s[tri_mask, :, 1, 1, :, :] = -10**5

        # t1 = (time.perf_counter() - t0)
        # print(f'Calculated layer stresses s in {t1/60:.2f} min.')
        return s


    def find_s0(self, go):
        """ ---------------- Calculate residual stress state caused by internal strains (shrinkage) ---------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - ecs      --> shrinkage strains per element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - s0       --> Matrix of residual element stresses [s_xi s_yi t_xyi] in form [k][l][i][j]
        -----------------------------------------------------------------------------------------------------------------"""
        nel = len(self.ELEMENTS[:,0])
        nlk = max(self.GEOMK["nlk"])
        s0 = np.zeros((nel, nlk, go, go, 3))
        for k in range(nel):
            for l in range(nlk):
                for i in range(go):
                    for j in range(go):
                        s0[k][l][i][j][:] = self.find_s0_klij(k,l,i,j)
        return s0


    def find_s0_vec(self, go):
        """
        Vectorised version of find_s0.

        Args:
            go  (int): Gauss order

        Returns:
            s0  (np.arr): residual element stresses [s_x, s_y, t_xy], shape (nel, nlk, go, go, 3)

        find_s0_klij(k, l, i, j) only depends on k via -Ec[k]*ecs[k]/(1+vc[k]^2) * [1,1,0], so we
        build that scalar coefficient once per element and broadcast over (l, i, j). The original
        find_s0 has no triangle-Gauss-point sentinel, so none is applied here either.
        """
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])

        Ec  = np.asarray(self.MATK["Ec"],  dtype=float)
        ecs = np.asarray(self.MATK["ecs"], dtype=float)
        vc  = np.asarray(self.MATK["vc"],  dtype=float)

        coeff = -Ec * ecs / (1.0 + vc ** 2)                       # (nel,)
        base  = np.array([1.0, 1.0, 0.0])                         # (3,)

        s0 = np.broadcast_to(coeff[:, None, None, None, None] * base,
                             (nel, nlk, go, go, 3)).copy()

        return s0


    def find_sh0(self):
        """ ---------------- Novel initialisation for NN-hybrid analysis  ---------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - ecs      --> shrinkage strains per element (are zero)
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - s0       --> Matrix of residual element stresses [nx, ny, nxy, mx, my, mxy, vx, vy] in form [k][i][j]
        -----------------------------------------------------------------------------------------------------------------"""
        nel = len(self.ELEMENTS[:,0])
        # nlk = max(self.GEOMK["nlk"])
        sh0 = np.zeros((nel, self.gauss_order, self.gauss_order, 8))
        for k in range(nel):
            # for l in range(nlk):
            for i in range(self.gauss_order):
                for j in range(self.gauss_order):
                    sh0[k][i][j][:] = self.find_sh0_klij(k,i,j)
                    # print('Element Number', k)
                    # print('initial stresses', sh0[k,:,:,:])
        return sh0


    def find_s0_klij(self, k,l,i,j):
        s0_klij = -np.array([1, 1, 0]) * self.MATK["Ec"][k] * self.MATK["ecs"][k] / (1 + self.MATK["vc"][k] ** 2)
        return s0_klij
    

    def find_sh0_klij(self, k,i,j):
        '''
        Novel calculation for NN-hybrid analysis
        '''
        s0_klij = -np.array([1, 1, 1, 0, 0, 0, 0, 0]) * self.MATK["Ec"][k] * self.MATK["ecs"][k] / (1 + self.MATK["vc"][k] ** 2)
        return s0_klij



    def find_ss(self, s,cmk):
        """ ---------------------------------- Calculate element steel stresses------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - strains
            - Material properties
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - ssx      --> Element steel stresses in x-Direction
            - ssy      --> Element steel stresses in y-Direction
        -----------------------------------------------------------------------------------------------------------------"""
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])
        ssx = np.zeros((nel, nlk, self.gauss_order, self.gauss_order))
        ssy = np.zeros((nel, nlk, self.gauss_order, self.gauss_order))
        spx = np.zeros((nel, nlk, self.gauss_order, self.gauss_order))
        spy = np.zeros((nel, nlk, self.gauss_order, self.gauss_order))
        for k in range(nel):
            for l in range(nlk):
                for i in range(self.gauss_order):
                    for j in range(self.gauss_order):
                        if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            ssx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                            ssy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                            spx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                            spy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                        elif cmk[k] < 1.5 or cmk[k] == 10:
                            ssx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                            ssy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                            spx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                            spy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                        else:
                            ssx[k][l][i][j] = s[k][l][i][j][0].ssx.real
                            ssy[k][l][i][j] = s[k][l][i][j][0].ssy.real
                            spx[k][l][i][j] = s[k][l][i][j][0].spx.real
                            spy[k][l][i][j] = s[k][l][i][j][0].spy.real
        return ssx,ssy,spx,spy


    def find_ss_vec(self, e, cmk):
        """
        Vectorised drop-in for find_ss. Signature differs: takes strain `e` (steel stress is
        a function of strain, not of the sigma stresses carried by the vec `s` array).

        Args:
            e    (np.arr): strain from find_e_vec, shape (nel, nlk, go, go, 5)
            cmk  (array-like): per-element cm identifier, shape (nel,)

        Returns:
            ssx, ssy, spx, spy : shape (nel, nlk, go, go). spx, spy are sentinel everywhere
            (CFRP is not implemented in ConstitutiveLaws, matching its "skipped" comment).

        Constitutive: bilinear steel law -- matches find_ss exactly in compression and
        elastic tension. For cracked-tension where the scalar pipeline uses ssr (TCM), this
        returns the bare bilinear value. ssx, ssy are post-processing outputs (the FEM solve
        does not consume them), so the simplification is generally fine.

        Sentinel positions (filled with -1e-5, matching find_ss):
            * cm < 1.5 or cm == 10
            * triangle (i=1, j=1) for go >= 2
        """
        nel = e.shape[0]
        nlk = e.shape[1]
        go  = e.shape[2]

        SENT = -1e-5
        ssx = np.full((nel, nlk, go, go), SENT)
        ssy = np.full((nel, nlk, go, go), SENT)
        spx = np.full((nel, nlk, go, go), SENT)
        spy = np.full((nel, nlk, go, go), SENT)

        cm_arr   = np.asarray(cmk)
        valid_cm = (cm_arr >= 1.5) & (cm_arr != 10)

        if valid_cm.any():
            # Uniform-mesh material assumption (matches find_s_vec): element-0 values.
            Es  = float(np.asarray(self.MATK["Esx"]).reshape(-1)[0])
            Esh = float(np.asarray(self.MATK["Eshx"]).reshape(-1)[0])
            fsy = float(np.asarray(self.MATK["fsyx"]).reshape(-1)[0])
            esy = fsy / Es

            def _ss_bilin(strain):
                return np.where(
                    strain >= esy,
                    fsy + Esh * (strain - esy),
                    np.where(strain <= -esy,
                             -fsy + Esh * (strain + esy),
                             strain * Es),
                )

            ssx_calc = _ss_bilin(e[..., 0])
            ssy_calc = _ss_bilin(e[..., 1])

            vmask = np.broadcast_to(valid_cm[:, None, None, None], (nel, nlk, go, go))
            ssx = np.where(vmask, ssx_calc, ssx)
            ssy = np.where(vmask, ssy_calc, ssy)

        # Triangle (i=1, j=1) sentinel for go >= 2
        if go >= 2:
            types    = np.asarray(self.ELS[4])
            tri_mask = (types == 3)
            ssx[tri_mask, :, 1, 1] = SENT
            ssy[tri_mask, :, 1, 1] = SENT
            spx[tri_mask, :, 1, 1] = SENT
            spy[tri_mask, :, 1, 1] = SENT

        return ssx, ssy, spx, spy


    def find_sc(self, e1, e3):
        """ ---------------------- Calculate element concrete principla compressive stresses-----------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - strains
            - Material properties
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - s_c3      --> Element concrete principal compressive stresses
        -----------------------------------------------------------------------------------------------------------------"""
        nel = len(self.ELEMENTS[:, 0])
        nlk = max(self.GEOMK["nlk"])
        sc3 = np.zeros((nel, nlk, self.gauss_order, self.gauss_order))
        for k in range(nel):
            for l in range(nlk):
                for i in range(self.gauss_order):
                    for j in range(self.gauss_order):
                        fc_p = self.MATK["fcp"][k]
                        e_c0 = self.MATK["ec0"][k]
                        if self.ELS[4][k] == 3 and i == 1 and j == 1:
                            sc3[k][l][i][j] = -10**-5*np.ones_like(sc3[k][l][i][j])
                        else:
                            sc3[k][l][i][j] = s_c3(e3[k][l][i][j], e1[k][l][i][j], fc_p, e_c0)
        return sc3


    """--------------------------------------------- Solve & Control ----------------------------------------------------"""
    def solve_sys_vec(self, B, fe, cDOF, cVAL, cmk, e, s):
        """
        Vectorised drop-in for solve_sys (NLFEA only -- no perm / perm1, no NN). Routes the
        stiffness assembly through k_glob_vec instead of k_glob, so the numerical s array
        from find_s_vec is consumed correctly instead of being fed back into get_et_vb which
        expects object-dtype stress instances (.sx, .sy, .txy attributes).

        Same return contract as solve_sys: (u, D_tot).
        """
        K, D_tot = self.k_glob_vec(B, e, s, cmk, self.gauss_order)

        Kcond  = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        return u, D_tot


    def solve_sys_nn_num_vec(self, B, fe, cDOF, cVAL, cmk, eh, sh, e, s, model_dim, scenario,
                             loaded_models=None):
        """
        Vectorised drop-in for solve_sys_nn_num. Routes the stiffness assembly through
        k_glob_nn_num_vec instead of k_glob_nn_num.

        Restrictions (inherited from k_k_nn_num_vec):
            * model_dim == 'THREEDIM' only.
            * cm == 3 only.

        Args:
            loaded_models (dict|None): dict of preloaded NN models from load_NN_model
                (key 'D' for the THREEDIM tangent predictor). If None, the model is loaded
                from disk on every prediction call (slow path).

        Same return contract as solve_sys_nn_num: (u, D_tot).
        """
        K, D_tot = self.k_glob_nn_num_vec(B, e, s, eh, sh, cmk, self.gauss_order,
                                          model_dim, scenario,
                                          loaded_models=loaded_models)

        Kcond  = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        return u, D_tot


    def solve_sys(self, B,fe, cDOF,cVAL, cmk,e,s, perm = None, perm1 = None):
        """ ------------------------------------------- Solve System ----------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - cmk: Constitutive model to be applied
                - 1 --> Linear Elasticity
                - 3 --> CMM
                - 10 --> Glass
                given for each element
            - perm: if permutation according to random matrix shall occur
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - u     --> Deformed Shape for current iteration step
            - K     --> Stiffness Matrix of current iteration step
        -----------------------------------------------------------------------------------------------------------------"""
        # K, Ke_tot = self.k_glob(B,e,s, cmk)
        K, D_tot = self.k_glob(B,e,s, cmk, self.gauss_order, perm = perm, perm1 = perm1)

        ### Calculation of spurious zero-energy modes: len(re01) must be equal to 6! ---------------------------------------
        # [e1,e2] = np.linalg.eig(K)
        # re1 = np.zeros_like(e1,dtype = float)
        # for i in range(len(e1)):
        #     re1[i] = e1[i].real
        # re10 = re1[re1<10]
        ### ----------------------------------------------------------------------------------------------------------------
        Kcond = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
        # u = np.matmul(Kinv, fecond)
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        # for i in range(len(cDOF)):
        #     if cDOF[i] < len(u):
        #         u = np.insert(u, int(cDOF[i]), 0)  # Add zero deformation from condensed nodes
        #     else:
        #         u = np.append(u, 0)
        # u = [u]
        # u = np.transpose(u)

        return u, D_tot
    

    def solve_sys_nn(self, B,fe, cDOF,cVAL, cmk,eh,sh):
        """ ------------------------------------------- Solve System ----------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - cmk: Constitutive model to be applied
                - 1 --> Linear Elasticity
                - 3 --> CMM
                given for each element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - u     --> Deformed Shape for current iteration step
            - K     --> Stiffness Matrix of current iteration step
        -----------------------------------------------------------------------------------------------------------------"""
        # K, Ke_tot = self.k_glob(B,e,s, cmk)
        K, D_tot = self.k_glob_nn(B,eh,sh, cmk)

        ### Calculation of spurious zero-energy modes: len(re01) must be equal to 6! ---------------------------------------
        # [e1,e2] = np.linalg.eig(K)
        # re1 = np.zeros_like(e1,dtype = float)
        # for i in range(len(e1)):
        #     re1[i] = e1[i].real
        # re10 = re1[re1<10]
        ### ----------------------------------------------------------------------------------------------------------------
        Kcond = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
        # u = np.matmul(Kinv, fecond)
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        # for i in range(len(cDOF)):
        #     if cDOF[i] < len(u):
        #         u = np.insert(u, int(cDOF[i]), 0)  # Add zero deformation from condensed nodes
        #     else:
        #         u = np.append(u, 0)
        # u = [u]
        # u = np.transpose(u)

        return u, D_tot

    def solve_sys_nn_num(self, B,fe, cDOF,cVAL, cmk,eh,sh, e,s, model_dim, scenario):
        """ ------------------------------------------- Solve System ----------------------------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - cmk: Constitutive model to be applied
                - 1 --> Linear Elasticity
                - 3 --> CMM
                given for each element
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            - u     --> Deformed Shape for current iteration step
            - K     --> Stiffness Matrix of current iteration step
        -----------------------------------------------------------------------------------------------------------------"""
        # K, Ke_tot = self.k_glob(B,e,s, cmk)
        K, D_tot = self.k_glob_nn_num(B,e, s, eh,sh, cmk, model_dim, scenario)

        Kcond = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        start = time.time()
        # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
        # u = np.matmul(Kinv, fecond)
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        end = time.time()
        time_Kinv._updatetime(delta_t=end - start)

        return u, D_tot

    def solve_0(self, B,fe,e, cDOF,cVAL, go):
        """ --------------------------- Solve Initial Iteration for linear elasticity -----------------------------------"""
        
        if self.MATK["cm"][0] == 10:
            # if the material is glass, also calculate the first iteration like lin.el. glass, no initialisation with general lin.el. material law
            K, D_tot = self.k_glob(B,e,e, 10*np.ones_like(fe), go)
        else: 
            # K, Ke_tot = self.k_glob(B,e,e, np.ones_like(fe))
            K, D_tot = self.k_glob(B,e,e, np.ones_like(fe), go)
        Kcond = self.m_stat_con(K, cDOF)
        fecond = self.v_stat_con(fe, cDOF, cVAL)

        # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
        # u = np.matmul(Kinv, fecond)
        lu, piv = lu_factor(Kcond)
        u = lu_solve((lu, piv), fecond)
        # for i in range(len(cDOF)):
        #     if cDOF[i] < len(u):
        #         u = np.insert(u, int(cDOF[i]), 0)  # Add zero deformation from condensed nodes
        #     else:
        #         u = np.append(u, 0)
        # u = [u]
        # u = np.transpose(u)
        return u, D_tot










