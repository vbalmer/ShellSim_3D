#### Translator file #####

# this dict states the relevant material parameters for the concrete classes C25/30,...,C50/60


dict_CC = {
    'CC': [0, 1, 2, 3, 4, 5],                                           # [-] just an identifier
    'Ec': [32e3, 33.6e3, 35e3, 36.3e3, 37.5e3, 39e3],                   # [MPa]
    'tb0': [5.2, 5.8, 6.4, 7.0, 7.6, 8.2],                              # [MPa]
    'tb1': [2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                              # [MPa]
    'ect': [0.08e-3, 0.09e-3, 0.09e-3, 0.1e-3, 0.1e-3, 0.11e-3],        # [-]
    'ec0': [2.2e-3, 2.3e-3, 2.3e-3, 2.4e-3, 2.4e-3, 2.5e-3],            # [-]
    'fcp': [25, 30, 35, 40, 45, 50],	                                # [MPa]    
    'fct': [2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                              # [MPa]
}