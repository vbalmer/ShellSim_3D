# vectorised version of stress simulator of Andreas
# vb, 10.03.2026


import numpy as np
import time

from constitutive_laws import *

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


    def find_s_vec(self, e, mat_dict, go = 1) -> np.array:
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
        s = np.zeros((e.shape[0], e.shape[1], 3, 3), dtype=np.complex64)

        e0 = e+np.array([0.0000000000000001j,0,0])
        material_law0 = ConstitutiveLaws(e0, self.constants, mat_dict, cm_klij=3)
        s[:,:,0,:] = material_law0.out().squeeze(-1)
        t1 =(time.perf_counter()-t0)
        print(f'Calculated 1/3 instance of layer stresses s in {t1/60:.2f} min.')

        e1 = e+np.array([0,0.0000000000000001j,0])
        material_law1 = ConstitutiveLaws(e1, self.constants, mat_dict, cm_klij=3)
        s[:,:,1,:] = material_law1.out().squeeze(-1)
        t2 =(time.perf_counter()-t0)
        print(f'Calculated 2/3 instance of layer stresses s in {t2/60:.2f} min.')

        e2 = e+np.array([0,0,0.0000000000000001j])
        material_law2 = ConstitutiveLaws(e2, self.constants, mat_dict, cm_klij=3)
        s[:,:,2,:] = material_law2.out().squeeze(-1)
        t3 =(time.perf_counter()-t0)
        print(f'Calculated 3/3 instance of layer stresses s in {t3/60:.2f} min.')

        return s


    def find_sh_vec(self, s, go = 1) -> np.array:
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
        
        s = s[:,:,0,:].reshape(n_tot, nl, 3, 1)                  # shape (n_tot, nl, 3, 1)
        s = s.real

        sh = (S.transpose(0,1,3,2)@s).squeeze(-1)       # shape (n_tot, nl, 6)

        sh = (t/nl)*sh.sum(axis = 1)                    # shape (n_tot, 6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated generalised stresses sh in {t1/60:.2f} min.')
        return sh


    def find_dh_vec(self, s, mat_dict, go = 1) -> np.array:
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

        Dmh = np.zeros((s.shape[0],3,3))
        Dbh = np.zeros((s.shape[0],3,3))
        Dmbh = np.zeros((s.shape[0],3,3))

        Dp = self.get_et(s, mat_dict, cm_klij = 3)       # shape (n_tot, 20, 3, 3)

        z_ = z.reshape(1, -1, 1, 1)

        Dmh     += np.sum((Dp)       *t/nl)               # shape (n_tot, 3, 3)
        Dmbh    += np.sum((-z_*Dp)    *t/nl)               # shape (n_tot, 3, 3)
        Dbh     += np.sum((z_**2*Dp)  *t/nl)               # shape (n_tot, 3, 3)

        De_1 = np.concatenate([Dmh, Dmbh], axis=2)          # (n_tot, 3, 6)
        De_2 = np.concatenate([Dmbh, Dbh], axis=2)          # (n_tot, 3, 6)
        De   = np.concatenate([De_1, De_2], axis=1)         # (n_tot, 6, 6)

        t1 =(time.perf_counter()-t0)
        print(f'Calculated stiffness matrix D in {t1/60:.2f} min.')
        return De
    

    def get_et(self, s, mat_dict, cm_klij = 1) -> np.array:
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
            dp = np.zeros((n_tot, nl, 3, 3))
            ET = s[:, :, :3, :3].imag / 1e-16
            
        return dp