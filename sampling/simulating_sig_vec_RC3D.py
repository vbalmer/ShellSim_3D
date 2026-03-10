# vectorised version of stress simulator of Andreas
# vb, 10.03.2026


import numpy as np

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

        n_tot = self.eps_g.shape[0]
        t = self.constants['t']                         # int
        nl = self.constants['n_layer']                  # int
        l=np.linspace(0,nl, nl+1, dtype=int)            # shape (nl,)
        z = -t/2+(2*l+1)*t/(2 * nl)                     # shape (nl,)
        
        I3 = np.eye(3)
        Z  = np.einsum('ij,kl->klij', I3, -z)           # shape (n_tot, nl, 3, 6)
        S = np.concatenate(
            [np.broadcast_to(I3, (n_tot, nl, 3, 3)), Z],
              axis=-1)  
        e = S @ self.eps_g                              # shape (n_tot, nl, 3)

        print('Calculated layer strains e')
        return e


    def find_s_vec(self, e, go = 1):
        """
        Vectorised version of Andreas' function find s for go = 1

        """
        # Entire file "Stresses_mixedreinf.py" in vectorised form
        # TODO!
        s = None


        print('Calculated layer stresses s')
        return s


    def find_sh_vec(self, s, go = 1):
        """
        Vectorised version of Andreas' function find sh for go = 1

        Args:
            s   (np.arr): layer stresses (n_tot, 20, 6)
        Returns:
            sh  (np.arr): generalised stresses (n_tot, 6)

        """

        n_tot = s.shape[0]                          
        t = self.constants['t']                         # int
        nl = self.constants['n_layer']                  # int
        l=np.linspace(0,nl, nl+1, dtype=int)            # shape (nl,)
        z = -t/2+(2*l+1)*t/(2 * nl)                     # shape (nl,)


        I3 = np.eye(3)
        Z  = np.einsum('ij,kl->klij', I3, -z)           # shape (n_tot, nl, 3, 6)
        S = np.concatenate(
            [np.broadcast_to(I3, (n_tot, nl, 3, 3)), Z],
              axis=-1)  
        
        s.reshape(n_tot, nl, 3, 1)                      # shape (n_tot, nl, 3, 1)

        sh = (S.transpose(0,1,3,2)@s).squeeze(-1)       # shape (n_tot, nl, 6)

        sh = (t/nl)*sh.sum(axis = 1)                    # shape (n_tot, 6)


        print('Calculated generalised stresses sh')
        return sh


    def find_dh_vec(self, s, go = 1):
        """
        Vectorised version of Andreas' function find dh for go = 1

        """
        dh = None


        print('Calculated stiffness matrix D')
        return dh