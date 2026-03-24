# vectorised version of stress simulator of Andreas
# vb, 10.03.2026

import config                       # for utilisation of GPU
from constitutive_laws import *
import sys
import os

if config.USE_GPU:
    import cupy as np
    import numpy as np_
    print(f'Imported {np.__name__} as np')
    print(f'Imported {np_.__name__} as np_')
else:
    import numpy as np
    np_ = np
    print(f'Imported {np.__name__} as np')
import time
import matplotlib.pyplot as plt


# potentially all these functions need to be carried out in batched manner / on gpu.

class SigSimulator:
    def __init__(self, constants):
        self.constants = constants


    def find_e_vec(self, eps_g, go = 1) -> np.array:
        """
        Vectorised version of Andreas' function find e for go = 1

        Args:
            eps_g   (np.arr): generalised strains (n_tot, 6)
        Returns:
            e       (np.arr): layer strains (n_tot, 20, 3)

        """
        t0 = time.perf_counter()
        n_tot = eps_g.shape[0]
        t = self.constants['t']                         # int
        nl = self.constants['n_layer']                  # int


        l = np.arange(nl)                            # (nl,)
        z = -t/2 + (2*l+1)*t/(2*nl)                  # (nl,)

        I3 = np.eye(3)                               # (3, 3)
        Z  = np.einsum('ij,k->kij', I3, -z)          # (nl, 3, 3)

        S = np.concatenate(
            [np.broadcast_to(I3, (nl, 3, 3)), Z],
            axis=-1
        )                                             # (nl, 3, 6)

        e = (S[np.newaxis] @ eps_g[:, np.newaxis, :, np.newaxis]).squeeze(-1)           #(1,nl,3,6) @ (n_tot,1,6,1) -> (n_tot,nl,3)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated layer strains e in {t1/60:.2f} min.')
        return e


    def find_s_vec(self, e, mat_dict, cm_klij=3, go = 1) -> np.array:
        """
        Vectorised version of Andreas' function find s for go = 1
        Args:
            e       (np.arr): layer strains (n_tot, 20, 3)
            mat_dict  (dict): material parameters (defined in main file)
        Returns:
            s       (np.arr): layer stresses (n_tot, 20, 3, 3), as complex array -> can use img part in d calculation.

        """
        t0 = time.perf_counter()
        # Entire file "Stresses_mixedreinf.py" in vectorised form

        if cm_klij == 3:
            # Reinforced Concrete Calculation
            s = np.zeros((e.shape[0], e.shape[1], 3, 3), dtype=np.complex64)

            e0 = e+np.array([0.0000000000000001j,0,0])
            material_law0 = ConstitutiveLaws(e0, self.constants, mat_dict, cm_klij=cm_klij)
            s[:,:,0,:] = material_law0.out().squeeze(-1)
            t1 =(time.perf_counter()-t0)
            print(f'Calculated 1/3 instance of layer stresses s in {t1/60:.2f} min.')


            e1 = e+np.array([0,0.0000000000000001j,0])
            material_law1 = ConstitutiveLaws(e1, self.constants, mat_dict, cm_klij=cm_klij)
            s[:,:,1,:] = material_law1.out().squeeze(-1)
            t2 =(time.perf_counter()-t0)
            print(f'Calculated 2/3 instance of layer stresses s in {t2/60:.2f} min.')

            e2 = e+np.array([0,0,0.0000000000000001j])
            material_law2 = ConstitutiveLaws(e2, self.constants, mat_dict, cm_klij=cm_klij)
            s[:,:,2,:] = material_law2.out().squeeze(-1)
            t3 =(time.perf_counter()-t0)
            print(f'Calculated 3/3 instance of layer stresses s in {t3/60:.2f} min.')

            # for debugging:
            # fig1, ax1 = plt.subplots(figsize=(10, 8))
            # im1 = ax1.imshow(s.imag.reshape((s.shape[0],180)), aspect='auto', cmap='viridis', interpolation='nearest')
            # plt.colorbar(im1, ax=ax1)
            # plt.show()

        elif cm_klij == 1: 
            # Linear Elastic calculation
            s = np.zeros((e.shape[0], e.shape[1], 3), dtype = np.float32)
            material_law0 = ConstitutiveLaws(e, self.constants, mat_dict, cm_klij=cm_klij)
            s = material_law0.out().real
            t1 =(time.perf_counter()-t0)
            print(f'Calculated linear elastic layer stresses s in {t1/60:.2f} min.')


        return s


    def find_sh_vec(self, s, cm_klij = 3, go = 1) -> np.array:
        """
        Vectorised version of Andreas' function find sh for go = 1

        Args:
            s   (np.arr): layer stresses (n_tot, 20, 3, 3)
        Returns:
            sh  (np.arr): generalised stresses (n_tot, 6), as non-complex number

        """
        t0 = time.perf_counter()
        n_tot = s.shape[0]                          
        t = self.constants['t']                         # int
        nl = self.constants['n_layer']                  # int
        l = np.arange(nl)                               # shape (nl,)
        z = -t/2+(2*l+1)*t/(2 * nl)                     # shape (nl,)


        I3 = np.eye(3)
        Z  = np.einsum('ij,k->kij', I3, -z)             # shape (nl, 3, 3)
        S = np.concatenate(                             # shape (n_tot, nl, 3, 6)
            [np.broadcast_to(I3, (n_tot, nl, 3, 3)),
            np.broadcast_to(Z,  (n_tot, nl, 3, 3))],
            axis=-1)                                            
        
        if cm_klij == 3:
            s = s[:,:,0,:].reshape(n_tot, nl, 3, 1)     # shape (n_tot, nl, 3, 1)
            s = s.real
        elif cm_klij == 1:
            s = s.reshape(n_tot, nl, 3, 1)              # shape (n_tot, nl, 3, 1)

        sh = (S.transpose(0,1,3,2)@s).squeeze(-1)       # shape (n_tot, nl, 6)

        sh = (t/nl)*sh.sum(axis = 1)                    # shape (n_tot, 6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated generalised stresses sh in {t1/60:.2f} min.')
        return sh


    def find_dh_vec(self, s, mat_dict, cm_klij = 3, go = 1) -> np.array:
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

        Dp = self.get_et(s, mat_dict, cm_klij = cm_klij)       # shape (n_tot, 20, 3, 3)

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
    

    def get_et(self, s, mat_dict, cm_klij = 3) -> np.array:
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
    

######################### batchwise calculations - wrapper #########################

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def sig_simulation_batchwise(eps_g, simulatesig:SigSimulator, cm, mat_dict, n_batches = 500):
    """
    Executes batch-wise calculation of sig_g (as opposed to full-batch)
    Args:
        eps_g           (np.arr)        : Sampled epsilon (ntot,6)
        simulatesig     (SigSimulator)  : Simulator class
        cm              (int)           : Material model (1 = Lin.el., 3 = RC)
        mat_dict        (dict)          : Material properties
        n_batches       (int)           : Amount of batches

    Returns:
        sig_g           (np.arr)        : Simulated sig (ntot, 6)
        dh              (np.arr)        : Simulated D   (ntot, 6,6)
    
    """
    sig_g = np_.zeros((eps_g.shape[0], 6))
    dh = np_.zeros((eps_g.shape[0],6,6))
    batch_size = int(eps_g.shape[0]/n_batches)
    
    t00 = time.perf_counter()
    for i in range(n_batches):
        if i%10 != 0:
            with HiddenPrints():
                sig_g, dh,_ = single_batch_execution(i, batch_size, simulatesig, mat_dict, cm, dh, sig_g, eps_g)

        else:
            sig_g, dh,t0 = single_batch_execution(i, batch_size, simulatesig, mat_dict, cm, dh, sig_g, eps_g)
            t_batch = time.perf_counter() - t0
            if i%10 == 0:
                print(f'Finished batch {i+1}/{n_batches} with batchsize = {batch_size} in {t_batch/60:.2f} min.')

    print(f'Finished calculation of all {n_batches} batches in {(time.perf_counter()-t00)/60:.2f} min')
    return sig_g, dh

def single_batch_execution(i:int, batch_size: int, simulatesig:SigSimulator, mat_dict, cm, dh, sig_g, eps_g):
    t0 = time.perf_counter()

    # 2.0 Define batch 
    start = i*batch_size
    end = (i+1)*batch_size
    eps_g_batch = eps_g[start:end,:]

    # 2.1 Find layer strains
    e = simulatesig.find_e_vec(eps_g_batch)

    # 2.2 Find layer stresses
    s = simulatesig.find_s_vec(e, mat_dict, cm_klij = cm)

    # 2.3 Find generalised stresses
    sig_g_batch = simulatesig.find_sh_vec(s, cm_klij = cm)

    # 2.4 Find stiffnesses
    dh_batch = simulatesig.find_dh_vec(s, mat_dict, cm_klij = cm)

    sig_g[start:end,:] = sig_g_batch.get().astype(sig_g.dtype)
    dh[start:end,:] = dh_batch.get().astype(dh.dtype)

    return sig_g, dh, t0