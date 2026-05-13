# Main_vb_vec.py
# 12.05.2026

import numpy as np
import time

from Mesh_gmsh_vb import input_definition
from fem_vb import fem_func
from main_utils_vb import *

def main_solver(mat: dict, conv_plt: bool, NN_hybrid: dict, model_path:str, new_folder_path = None):
    '''
    Function to start main solver, calls input definition function (with geometry and material definition)

    INPUTS:
    mat             (dict)      Containing all potential input features
    conv_plt        (bool-dict) To turn on or off plotting of convergence plots (conv_plt['conv']: convergence plots; conv_plt['else']: all other plots)
    NN_hybrid       (bool-dict) To turn on or off solving with NN solver
    model_path      (str)       Path to trained model
    new_folder_path (str)       Path to where outputs are being saved per iteration. If None: Saves after every iteration, 
                                doesn't continue load_path calculation based on previous load step.

    OUTPUTS:
    mat_res     (dict)      Contains all relevant outputs from simulation that are required for data set
    
    '''

    start_main = time.time()
    # -------------------------------------------------------------------------------------------------------------------- #
    # 1 Geometry definition (Running mesh_gmsh_vb script as "input_definition" function)
    # -------------------------------------------------------------------------------------------------------------------- #    

    print("-------------------------------------------------------")
    print("1 Geometry definition")
    print("-------------------------------------------------------")

    MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln = input_definition(mat, NN_hybrid)
    fem_func0 = fem_func(MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, model_path)
    fem_func1 = fem_func(MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln, model_path)

    # -------------------------------------------------------------------------------------------------------------------- #
    # 2 Initiation of Iteration
    # -------------------------------------------------------------------------------------------------------------------- #

    print("-------------------------------------------------------")
    print("2 Initialisation")
    print("-------------------------------------------------------")

    # 2.1 Vector of External Forces and Condensed DOFs
    " 2.1 Output:   - fe: External Forces per DOF"
    "               - cDOF: Condensed DOFs"
    print("2.1 Assembly of force vector and condensed DOFs")
    # 2.1.1 Vector of External Forces
    fe = fem_func0.f_assemble(0, Load_el, Load_n)
    B = fem_func0.find_b(go=2)

    # 2.1.2 Residual Strains
    [e0, ex0, ey0, gxy0, e10, e30, th0] = fem_func0.find_e0_vec(go=2)
    [e0c, ex0c, ey0c, gxy0c, e10c, e30c, th0c] = fem_func0.find_e0_vec(go=1)

    # 2.1.3 Vector of External Forces Caused by Internal Stresses (z.B. Schwinden)
    s0 = fem_func0.find_s0_vec(go=2)
    [sh0,f0] = fem_func0.f0_assemble(B,s0,1)

    # 2.1.4 Constrained DOFs
    cDOF,cVAL = fem_func0.c_dof()

    # 2.2 Iteration Initiation: Solve Equation for linear elasticity
    " 2.2 Output:   - u: Deformation for linear elasticity"
    "               - eh: Generalized strains for linear elasticity"
    "               - [e,ex,ey,gxy,e1,e3,th]: Strains epsilon(u)-e0 "
    "               - [un,thn]: Displacements and rotations separated"
    "               - eold, unold, thnold: Initiations as i-1-Iteration Step values"
    "               - Initiation of r (residual vector): size of u"
    
    e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_ = linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order)

    # 3.1 Number of Iteration Steps
    " 3.1 Output:   - numit: Number of iteration steps"
    if MATK['cm'][0] == 1 or MATK['cm'][0] == 10:
        numit = 0
    elif MATK['cm'][0] == 3:
        numit = NN_hybrid['numit']

    sh_cum, sh_cum_, eh_cum, eh_cum_, u_cum, u_cum_, De_cum, De_cum_, fi_cum, fi_cum_ = store_intermediate_solutions(numit, NN_hybrid, sh, sh_, eh, eh_, u, u_, De_tot, De_tot_)

    if conv_plt['else']:
        if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']: 
            sh_true_0 = sh_.reshape((-1,8))
            sh_pred_0 = sh.reshape((-1,8))
            norms_0 = check_plots_perit(model_path, -1, sh_.reshape((-1,8)), sh.reshape((-1,8)), eh.reshape((-1,8)), sh_true_0, sh_pred_0, sig = True)
        elif not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
            pass
            # this plot does not make any sense because the D matrix is not predicted by epsilon in this 0-th iteration


     # -------------------------------------------------------------------------------------------------------------------- #
    # 3 Nonlinear Solution: Iteration with Tangent Stiffness
    # -------------------------------------------------------------------------------------------------------------------- #

    print("-------------------------------------------------------")
    print("3 Solution")
    print("-------------------------------------------------------")

    if numit == 0:
        # Carrying out relevant calculations
        fi = fem_func0.find_fi(B, sh)
        r = np.zeros_like(r)
        rcond = fem_func0.v_stat_con(r, cDOF, np.zeros_like(cVAL))

        # Calculate values for convergence plots
        [un, thn] = un_thn(u)
        mat_convergence_un_thn = convergence_values_un_thn(u, unold = un, thnold = thn)
        mat_convergence_un_thn_nl = convergence_values_un_thn(u, unold, thnold)

    elif it_type == 2:
        raise RuntimeWarning('Secant stiffness iteration is outdated, please use it_type = 1')
    
    elif it_type == 1:
        # 3.2  Tangent Iteration
        print("3.2 Tangent Stiffness Iteration")
        for j in range(len(Load_n)):
            if isinstance(mat['F'], np.ndarray):
                if len(mat['F'])>1: 
                    print('----------------------------------------------------------------------------')
                    print(f'Started iteration for load step {j}, with load level {mat["F"][j]/mat["L"]}')
            else: 
                print('----------------------------------------------------------------------------')
                print(f'Started iteration with load level {mat["F"]/mat["L"]}')
            fe = fem_func0.f_assemble(j, Load_el, Load_n)
            for i in range(numit):
                start_time_iteration = time.perf_counter()

                # 3.2.1 Solution with Secant Stiffness for given Iteration Step
                " 3.2.1 Output: - s,sx,sy,txy: Stresses"
                "               - u: Deformations"
                "               - eh: Generalized strains"
                "               - [e, ex, ey, gxy, e1, e3, th]: Strains epsilon(u)-e0"

                fi = fem_func0.find_fi(B, sh)
                fi_ = fi.copy()
                if not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                    fi_cum[i,:,:] = fi
                elif NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                    pass
                elif NN_hybrid['predict_D']:
                    fi_cum[i,:,:] = fi
                    fi_cum_[i,:,:] = fi_
                
                if NN_hybrid['predict_D']:
                    if NN_hybrid['model_dim'] == 'ALLDIM':
                        du, De_tot=fem_func0.solve_sys_nn(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],eh,sh)
                        u -= du
                    else:
                        # in the case of ONEDIM_x, ONEDIM_y, TWODIM or THREEDIM
                        du, De_tot=fem_func0.solve_sys_nn_num(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],eh,sh, e, s, NN_hybrid['model_dim'], scenario = mat["s"])
                        u -= du
                    du_, De_tot_=fem_func1.solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
                    u_ -= du_
                elif (NN_hybrid['PERM'] is not None):
                    if NN_hybrid['predict_D'] or NN_hybrid['predict_sig']:
                        raise ValueError('The PERM function should solely be used for NLFEA-deplyoments without NN predictions.')
                    du, De_tot=fem_func0.solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s, perm = NN_hybrid['PERM'])
                    u -= du
                elif ('PERM1' in NN_hybrid and NN_hybrid['PERM1'] is not None):
                    if NN_hybrid['predict_D'] or NN_hybrid['predict_sig']:
                        raise ValueError('The PERM1 function should solely be used for NLFEA-deplyoments without NN predictions.')
                    du, De_tot=fem_func0.solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s, perm1 = NN_hybrid['PERM1'])
                    u -= du
                else:
                    du, De_tot=fem_func0.solve_sys_vec(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
                    u -= du
                    u_, De_tot_ = u.copy(), De_tot.copy()      # no prediction of D --> u is real u, D is real D

                if NN_hybrid['predict_sig']:
                    eh = fem_func0.find_eh_vec(B,u, gauss_order)
                    sh = fem_func0.find_sh_nn_vec(eh, gauss_order, NN_hybrid['model_dim'])
                    # Calculate the "real" strains and stresses for the D calculation (or just for checking)
                    eh_ = fem_func1.find_eh_vec(B,u, gauss_order)
                    [e, ex, ey, gxy, e1, e3, th] = fem_func1.find_e(e0, eh_, gauss_order)
                    s = fem_func1.find_s_vec(e,s_prev,gauss_order)
                    # This calculation is not strictly required but can help as a control
                    sh_ = fem_func1.find_sh_vec(s, gauss_order)
                else:
                    eh = fem_func0.find_eh_vec(B,u, gauss_order)
                    eh_ = fem_func0.find_eh_vec(B,u_, gauss_order)
                    [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e_vec(e0, eh, gauss_order)
                    s = fem_func0.find_s_vec(e,s_prev,gauss_order)
                    sh = fem_func0.find_sh_vec(s, gauss_order)
                

                # 3.2.2 Convergence Control: Residual Vector
                " 3.2.2 Output: - r: Residual = fi-fe"
                "               - rcond: Residual without condensed DOFs"
                r = np.add(fi, -(fe+f0))
                rcond = fem_func0.v_stat_con(r, cDOF,np.zeros_like(cVAL))
                print(" - Iteration step " + str(i) + " complete, maximum residual = " + str(np.round(np.max(abs(rcond)),1)))

                #############################################
                # Collecting values for plots
                #############################################

                # saving values for plotting per iteration
                sh_cum[i+1,:,:,:,:] = sh
                eh_cum[i+1,:,:,:,:] = eh
                u_cum[i+1,:,:,] = u
                De_cum[i+1,:,:,:,:,:] = De_tot
                if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']: 
                    sh_cum_[i+1,:,:,:,:] = sh_
                    eh_cum_[i+1,:,:,:,:] = eh_
                if NN_hybrid['predict_D'] and not NN_hybrid['predict_sig']:
                    u_cum_[i+1,:,:] = u_
                    De_cum_[i+1,:,:,:,:,:] = De_tot_
                    eh_cum_[i+1,:,:,:,:] = eh_
                if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
                    u_cum_[i+1,:,:] = u_
                    De_cum_[i+1,:,:,:,:,:] = De_tot_
                    eh_cum_[i+1,:,:,:,:] = eh_
                    sh_cum_[i+1,:,:,:,:] = sh_
                
                if i == 0: 
                    mat_conv = None
                else: 
                    mat_conv = mat_convergence_eps_r

                if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
                    r[cDOF.astype(int)] = 0
                    # for the calculation of convergence values still use the "layer-e" as it is anyways calculated as a check. 
                    mat_convergence_eps_r = convergence_values_eps_r(e_conv=e, eold=eold, r=r, mat_convergence_eps_r= mat_conv, i=i)
                else: 
                    r[cDOF.astype(int)] = 0
                    mat_convergence_eps_r = convergence_values_eps_r(e_conv=e, eold=eold, r=r, mat_convergence_eps_r= mat_conv, i=i)

                mat_convergence_un_thn_nl = convergence_values_un_thn(u, unold, thnold)
                wandb.log({'it_step': i})
                desired_keys_un = ['convun50', 'convun90', 'convun99', 'convthn50', 'convthn90', 'convthn99']
                desired_keys_eps_r = ['conve50', 'conve90', 'conve99', 'convrf', 'convrm']
                wandb.log({key: mat_convergence_un_thn_nl[key] for key in desired_keys_un})
                wandb.log({key: mat_convergence_eps_r[key] for key in desired_keys_eps_r})

                print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(np.round(abs(mat_convergence_eps_r['convrf'][-1]), 1)))
                
                # Settings for plots per iteration
                if conv_plt['conv']:
                    if NN_hybrid['predict_sig'] or NN_hybrid['predict_D']:
                        # if anything is predicted: use NN as argument of convergence_plot function
                        if NN_hybrid['PERM'] is not None: 
                            raise NameError('PERM should be set to None if NN is active.')
                        plot_convergence(i, mat_convergence_un_thn_nl, numit, mat_convergence_eps_r, r, 'NN')
                    else:
                        # if nothing is predicted: use NLFEA or PERM as argument of convergence_plot function
                        if NN_hybrid['PERM'] is not None:
                            plot_convergence(i, mat_convergence_un_thn_nl, numit, mat_convergence_eps_r, r, 'PERM')
                        elif ('PERM1' in NN_hybrid) and NN_hybrid['PERM1'] is not None:
                            plot_convergence(i, mat_convergence_un_thn_nl, numit, mat_convergence_eps_r, r, 'PERM1')
                        else: 
                            plot_convergence(i, mat_convergence_un_thn_nl, numit, mat_convergence_eps_r, r, 'NLFEA')                    
                        
                if conv_plt['else']:
                        if not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                            all_eps_sig_plots(i+1, numit, eh_cum, 0, sh_cum, NN = 'NLFEA', tag = 50)
                            all_u_fi_plots(i+1, numit, u_cum, fi_cum, NN = 'NLFEA')
                        elif NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                            all_eps_sig_plots(i+1, numit, eh_cum, 0, sh_cum, eh_cum_, sh_cum_, NN = 'NN', tag = 50)
                        elif not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
                            # all_u_fi_plots(i+1, numit, u_cum, fi_cum)
                            all_De_plots(i+1, numit, eh_cum, eh_cum_, None, De_cum, De_cum_,None, tag = 'max')
                        elif NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
                            all_De_plots(i+1, numit, eh_cum, eh_cum_, None, De_cum, De_cum_,None, tag = 'max')
                            all_eps_sig_plots(i+1, numit, eh_cum, 0, sh_cum, NN = 'NLFEA', tag = 50)
                        
                        if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
                            if gauss_order == 1: 
                                if i == 0:
                                    sh_true_0 = sh_[:,0,0,:]
                                    sh_pred_0 = sh[:,0,0,:]
                                    norms_0 = check_plots_perit(model_path, i, sh_[:,0,0,:], sh[:,0,0,:], eh[:,0,0,:], sh_true_0, sh_pred_0)
                                else:
                                    check_plots_perit(model_path, i, sh_[:,0,0,:], sh[:,0,0,:], eh[:,0,0,:], sh_true_0, sh_pred_0, norms0 = norms_0)
                            else: 
                                if i == 0:
                                    sh_true_1 = sh_.reshape((-1,8))
                                    sh_pred_1 = sh.reshape((-1,8))
                                    norms_1 = check_plots_perit(model_path, i, sh_.reshape((-1,8)), sh.reshape((-1,8)), eh.reshape((-1,8)), 
                                                                sh_true_1, sh_pred_1, sig = True)
                                else: 
                                    check_plots_perit(model_path, i, sh_.reshape((-1,8)), sh.reshape((-1,8)), eh.reshape((-1,8)), 
                                                    sh_true_1, sh_pred_1, norms_0 = norms_1, sig = True)
                        if NN_hybrid['predict_D'] and not NN_hybrid['predict_sig']: 
                            if i == 0: 
                                De_true_1 = De_tot_.reshape((-1,8,8))
                                De_pred_1 = De_tot.reshape((-1,8,8))
                                norms_1 = check_plots_perit(model_path, i, sh.reshape((-1,8)), sh.reshape((-1,8)), eh_cum[i,:,:,:,:].reshape((-1,8)), 
                                                            D_true = De_cum_[i+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = De_cum[i+1,:,:,:,:,:].reshape((-1,8,8)), 
                                                            sig = False)
                            else: 
                                check_plots_perit(model_path, i, sh.reshape((-1,8)), sh.reshape((-1,8)), eh_cum[i,:,:,:,:].reshape((-1,8)),
                                                D_true = De_cum_[i+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = De_cum[i+1,:,:,:,:,:].reshape((-1,8,8)), 
                                                sig = False)
                        if not NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']: 
                            check_plots_perit(model_path, i, sh.reshape((-1,8)), sh.reshape((-1,8)), eh_cum[i,:,:,:,:].reshape((-1,8)),
                                            D_true = De_cum[i+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = De_cum[i+1,:,:,:,:,:].reshape((-1,8,8)), 
                                            sig = False)
                            
                        if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
                            check_plots_perit(model_path, i, sh_cum_[i,:,:,:,:].reshape((-1,8)), sh_cum[i,:,:,:,:].reshape((-1,8)), eh_cum[i,:,:,:,:].reshape((-1,8)), 
                                            sh_true_1, sh_pred_1, norms_0 = norms_1, sig = True)
                            check_plots_perit(model_path, i, sh_cum_[i,:,:,:,:].reshape((-1,8)), sh_cum[i,:,:,:,:].reshape((-1,8)), eh_cum[i,:,:,:,:].reshape((-1,8)),
                                            D_true = De_cum_[i+1,:,:,:,:,:].reshape((-1,8,8)), D_pred = De_cum[i+1,:,:,:,:,:].reshape((-1,8,8)), 
                                            sig = False)
                            
                # save new values as the old values for the next iteration: 
                if i < numit-1:
                    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
                        # Leave it at e here, which is calculated as a check. potentially later change to eh
                        eold = e
                    else:
                        eold = e
                    [un, thn] = un_thn(u)
                    unold = un
                    thnold = thn
                s_prev = s

                if np.round(np.max(abs(rcond)),1) < 5 and i < NN_hybrid['numit']-1: 
                    print(f'Stopped iteration after {i} steps, because the results already converged.')
                    mat_res = save_data_inter_loop(NN_hybrid, eold, fem_func0, B, u, gauss_order, e0c, sh0, s_prev, MATK, COORD, GEOMK, ELS, GEOMA, BC, MASK, NODESG,
                                                    e0, e, s, fi, fe, f0, cDOF, start_main, mat_convergence_un_thn_nl, na, mat,
                                                    De_tot, u_cum, u_cum_, sh_cum, sh_cum_, eh_cum, eh_cum_, De_cum, De_cum_)
                    if isinstance(mat['F'], np.ndarray):
                        if len(mat['F'])>1:
                            save_deployment_loadpath(new_folder_path, int(mat["F"][j]/mat["L"]), NN_hybrid, conv_plt)
                    
                    break
            
                elif i == NN_hybrid['numit']-1:
                    print(f'Stopped iteration after {i} steps, because maximum iteration steps were reached. Solution did not converge for this load step.')
                    mat_res = save_data_inter_loop(NN_hybrid, eold, fem_func0, B, u, gauss_order, e0c, sh0, s_prev, MATK, COORD, GEOMK, ELS, GEOMA, BC, MASK, NODESG,
                                                    e0, e, s, fi, fe, f0, cDOF, start_main, mat_convergence_un_thn_nl, na, mat,
                                                    De_tot, u_cum, u_cum_, sh_cum, sh_cum_, eh_cum, eh_cum_, De_cum, De_cum_)
                    if isinstance(mat['F'], np.ndarray):
                        if len(mat['F'])>1:
                            save_deployment_loadpath(new_folder_path, int(mat["F"][j]/mat["L"]), NN_hybrid, conv_plt) 
                        if j < len(mat['F']):
                            print('Redoing lin.el. calculation to start next load step afresh.')
                            e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_ = linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order)         

                t1 =(time.perf_counter()-start_time_iteration)
                print(f'Time required for one iteration step: {t1/60:.2f} min.')

    print('Finished :)')
    return mat_res



























def linear_elastic_solution(fem_func0, fem_func1, B, fe, f0, e0, cDOF, cVAL, NN_hybrid, gauss_order):
    print("2.2 Solution for Linear Elasticity")

    # The first (initialising) solve should always be carried out based on the linear elastic stiffness.
    u, De_tot = fem_func0.solve_0(B,fe+f0,np.zeros_like(e0),cDOF,cVAL, go=2)
    u_, De_tot_ = u.copy(), De_tot.copy()
    uz = u[2::6]
    [un, thn] = un_thn(u)
    unold = un
    thnold = thn

    if NN_hybrid['predict_sig']:
        # if we use sigma from the NN we need to calculate sh with the NN
        eh = fem_func0.find_eh_vec(B,u,gauss_order)
        sh = fem_func0.find_sh_nn_vec(eh, gauss_order, NN_hybrid['model_dim'])
        print('Gauss order for NN-run is: ', gauss_order)
        eold = eh
        r = u
        # additionally require calculation of e, s to determine D without NN later / as a control in the case of predicting both sig and D
        eh_ = fem_func1.find_eh_vec(B,u,gauss_order)
        [e,ex,ey,gxy,e1,e3,th] = fem_func1.find_e_vec(e0,eh_,gauss_order)
        s_lin = fem_func1.find_s(e,0,gauss_order,True)
        s_prev = s_lin
        s = fem_func1.find_s_vec(e,s_prev,gauss_order)
        eold = e
        # Note: sh is theoretically not required but can help as a control
        sh_ = fem_func1.find_sh_vec(s, gauss_order)

    else:
        # if we don't use the NN, calculate the strains and stresses layer-wise
        # start_main = time.time()
        eh = fem_func0.find_eh_vec(B,u,gauss_order)
        eh_ = fem_func0.find_eh_vec(B,u_,gauss_order)
        [e,ex,ey,gxy,e1,e3,th] = fem_func0.find_e_vec(e0,eh,gauss_order)
        # print('eh', eh[0,0,0,:])
        # print('e', e[0,0,0,:])

        s_lin = fem_func0.find_s_vec(e,0,gauss_order,True)
        s_prev = s_lin
        s = fem_func0.find_s_vec(e,s_prev,gauss_order)
        # st = s[0][10][0][0][0]
        # print('s_klij', np.array([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real]))
        sh = fem_func0.find_sh_vec(s, gauss_order)
        sh_ = sh.copy()
        if NN_hybrid['predict_D']:
            print('Gauss order for NN-run is: ', gauss_order)
        else:
            print('Gauss order for NLFEA-run is: ', gauss_order)
        eold = e
        r = u

    return e, eold, eh, eh_, s, s_prev, sh, sh_, r, u, u_, unold, thnold, De_tot, De_tot_

def save_data_inter_loop(NN_hybrid, eold, fem_func0, B, u, gauss_order, e0c, sh0, s_prev, MATK, COORD, GEOMK, ELS, GEOMA, BC, MASK, NODESG,
                         e0, e, s, fi, fe, f0, cDOF, start_main, mat_convergence_un_thn_nl, na, mat,
                         De_tot, u_cum, u_cum_, sh_cum, sh_cum_, eh_cum, eh_cum_, De_cum, De_cum_):


    # 3.3 Collect Values for Convergence
    " 3.3 Output:   - strain values in [k][l][i][j] format"
    # plt.show(block=False)
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
        diffe = eold - e
        rele = np.divide(diffe,e)
    else:
        diffe = eold - e
        rele = np.divide(diffe,e)


    # 3.4 Solution Value Collection
    " 3.4 Output:   - ui,thi: Nodal displacements and rotations in global coordinates"
    "               - ehc, ec: Generalized strains and strains in element centers in local coordinates"
    "               - sh: Generalized stresses in element centers in local coordinates"
    "               - [s,sx,sy,txy]: Stresses in integration points in local coordinates"
    "               - [ssx,ssy]: Steel stresses in integration points in local coordinates"
    "               - [Nx,...Qy]: Stress Resultants (sectional forces) in element centers in local coordinates"
    # print('u2',u)
    ux = u[0::6]
    uy = u[1::6]
    uz = u[2::6]
    thx = u[3::6]
    thy = u[4::6]
    thz = u[5::6]
    eh = fem_func0.find_eh(B,u, gauss_order)
    # print('eh2', eh[:,0,0,0])

    Bc = fem_func0.find_b(1)
    ehc = fem_func0.find_eh(Bc,u,1)
    # print('ehc', ehc[:,0,0,4])


    if NN_hybrid['predict_sig']:
        sh = fem_func0.find_sh_nn(eh,1, NN_hybrid['model_dim'])+sh0
    else: 
        ec = fem_func0.find_e(e0c,ehc,1)[0]
        sc = fem_func0.find_s(ec,s_prev,1)
        sh = fem_func0.find_sh(sc,1)+sh0
    
    if NN_hybrid['predict_sig'] or NN_hybrid['predict_D']:
        ssx = np.zeros_like(eh)
        ssy = np.zeros_like(eh)
        spx = np.zeros_like(eh)
        spy = np.zeros_like(eh)

    else:
        if max(MATK["cm"]) > 1:
            [ssx, ssy, spx, spy] = fem_func0.find_ss_vec(e, MATK["cm"])
            print('This vectorised implementation was carried out pretty fast. Please check the results of steel stresses with caution.')
            # print('Max ssx:', max(ssx.all()), 'Max ssy:', max(ssy.all()))
        else:
            ssx = np.zeros_like(ex)
            ssy = np.zeros_like(ex)
            spx = np.zeros_like(ex)
            spy = np.zeros_like(ex)


    if NN_hybrid['predict_sig'] or NN_hybrid['predict_D']:
        # careful: this will yield some wrong solutions, please double-check after it runs.
        [e, ex, ey, gxy, e1, e3, th] = [0, 0, 0, 0, 0, 0, 0]
    else: 
        [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e(np.zeros_like(e0), eh, gauss_order)
    
    Nx = sh[:, :, :, 0]
    Ny = sh[:, :, :, 1]
    Nxy = sh[:, :, :, 2]
    Mx = sh[:, :, :, 3]
    My = sh[:, :, :, 4]
    Mxy = sh[:, :, :, 5]
    Qx = sh[:, :, :, 6]
    Qy = sh[:, :, :, 7]

    # 3.5 Print Information: Displacements, Applied Loads and Reactions
    print("2.4 Solution complete")
    r = np.add(fi, -(fe+f0))

    print(" - Maximum displacements:")
    print("   - ux_max = " + str(np.round(float(max(np.abs(ux))),3)))
    print("   - uy_max = " + str(np.round(float(max(np.abs(uy))),3)))
    print("   - uz_max = " + str(np.round(float(max(np.abs(uz))),3)))
    print(" - Sum of applied forces:")
    print("   - Fx = " + str(np.round(float(sum(fe[0::6])),3)))
    print("   - Fy = " + str(np.round(float(sum(fe[1::6])),3)))
    print("   - Fz = " + str(np.round(float(sum(fe[2::6])),3)))
    print(" - Sum of reaction forces:")
    print("   - Rx = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==0].astype(int)])),1)))
    print("   - Ry = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==1].astype(int)])),1)))
    print("   - Rz = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==2].astype(int)])),1)))

    # -------------------------------------------------------------------------------------------------------------------- #
    # 4 Postprocessing and Time Management
    # -------------------------------------------------------------------------------------------------------------------- #

    # 4.0 Import
    # from Mesh_gmsh import ELS, COORD,GEOMA,MASK,GEOMK,na,BC
    # ELEMENTS = ELS[0]
    # from Postprocess import post
    # import numpy as np
    from Postprocess_vb import post_func
    from numpy import save
    import pickle
    import os

    # 4.1 Time Management
    end_main = time.time()
    print("total time used: ", end_main-start_main)
    from fem_vb import time_stress,time_strain,time_K,time_B,time_Kinv,time_sh,time_eh
    print("time used in stress calculation", time_stress.time_spent)
    print("time used in strain calculation", time_strain.time_spent)
    print("time used in stiffness matrix calculation", time_K.time_spent)
    print("time used in B-matrix calculation", time_B.time_spent)
    print("time used in stiffness matrix inversion", time_Kinv.time_spent)
    print("time used in calculation of sh", time_sh.time_spent)
    print("time used in calculation of eh", time_eh.time_spent)

    # 4.2 Postprocess Data
    print("-------------------------------------------------------")
    print("3 Postprocess Data")
    print("-------------------------------------------------------")


    # 4.2.1 Import POST
    " 4.2.1 Output:   -POST: post processed data from postprocess function"
    post_func0 = post_func(COORD, GEOMK, ELS, GEOMA, NN_hybrid)
    # if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
    #     rele = np.array([rele])
    relun, relthn = mat_convergence_un_thn_nl['relun'], mat_convergence_un_thn_nl['relthn']
    POST = post_func0.post(ux,uy,uz,r,thx,thy,thz,Nx,Ny,Nxy,Mx,My,Mxy,Qx,Qy,ex,ey,gxy,e3,e1,th,ssx,ssy,spx,spy,rele,relun,relthn)

    # 4.3 Export Data

    mat_res = {
        # 'L': mat['L'],
        # 'B': mat['B'],
        'BC': BC,
        'COORD_c': COORD["c"],
        'COORD': COORD,
        'ELEMENTS': ELS[0],
        'MATK': MATK,
        'fe': fe,
        'GEOMA': GEOMA,
        'GEOMK': GEOMK, 
        'MASK': MASK,
        'NODESG': NODESG,
        'POST': POST,
        'gauss_order': gauss_order,
        'na': na,
        'sig_g': sh,
        'eps_g': ehc,
        'ux': ux,
        'uy': uy,
        'uz': uz, 
        'thx': thx,
        'thy': thy,
        'thz': thz,
        'De_tot': De_tot,
        # 'Ke_tot': Ke_tot,
        'u_cum': u_cum,
        'u_cum_': u_cum_,
        'sh_cum': sh_cum,
        'sh_cum_': sh_cum_,
        'eh_cum': eh_cum,
        'eh_cum_': eh_cum_,
        'De_cum': De_cum,
        'De_cum_': De_cum_,
        'ssx': ssx,
        'ssy': ssy,
        }
    


    mat_res.update(mat)



    # save the data: 
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']: 
        fname = 'mat_res_NN.pkl'
    elif NN_hybrid['predict_sig']:
        fname = 'mat_res_NN_sig.pkl'
    elif NN_hybrid['predict_D']:
        fname = 'mat_res_NN_D.pkl'
    elif NN_hybrid['PERM'] is not None:
        fname = 'mat_res_norm_perm.pkl'
    elif 'PERM1' in NN_hybrid and NN_hybrid['PERM1'] is not None: 
        fname = 'mat_res_norm_perm1.pkl'
    else: 
        fname = 'mat_res_norm.pkl'

    # mat_res_pd_NN = pd.DataFrame.from_dict(mat_res)
    # mat_res.to_pickle(os.path.join('05_Deploying\\data_out',fname))

    with open(os.path.join('deploying\\data_out', fname), 'wb') as f:
        pickle.dump(mat_res, f)
    

    return mat_res

def save_deployment_loadpath(new_folder_path, force_i, NN_hybrid, conv_plt):
    #########################################
    # Plotting and saving to folders
    #########################################


    # if data should be saved to folder instead of being overwritten with the next simulation, use save_folder = True

    import shutil
    import os

    ####### which data files to save: #######
    if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN.pkl']
    elif NN_hybrid['predict_sig']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_sig.pkl']
    elif NN_hybrid['predict_D']:
            relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_D.pkl']
    else:
            relative_path = ['data_out\\mat_res_norm.pkl']

    # if NN_hybrid_3['PERM'] is not None: 
    #         relative_path.append('data_out\\mat_res_norm_perm.pkl')

    # if NN_hybrid_3['PERM1'] is not None:
    #         relative_path.append('data_out\\mat_res_norm_perm1.pkl')

    ######## which plots to save: #######
    if conv_plt['conv'] and conv_plt['else']:
        if NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it', 'plots\\De_plots_it', 'plots\\mike_plots_D']
        elif NN_hybrid['predict_sig']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it']
        elif NN_hybrid['predict_D']:
            folder_paths = ['plots\\mike_plots', 'plots\\diagonal_plots_D', 'plots\\diagonal_plots_D_true', 'plots\\conv_plots', 
                        'plots\\eps_sig_plots_it', 'plots\\De_plots_it', 'plots\\mike_plots_D', 'plots\\mike_plots_D_true']
    elif conv_plt['conv']:
        folder_paths = ['plots\\conv_plots'] 
    elif not conv_plt['conv'] and not conv_plt['else']:
        pass 
        # do not save any plots to new folder


    # Copy files to new subfolder
    subfolder_path = os.path.join(new_folder_path, str(force_i))
    os.makedirs(subfolder_path, exist_ok = True)

    for i, file_path in enumerate(relative_path):
        file_path_n = os.path.join('deploying', file_path)
        destination_path = os.path.join(subfolder_path, os.path.basename(file_path_n))
        shutil.copy(file_path_n, destination_path)
        print(f'File {i + 1} copied to {destination_path}')

    # Copy folders
    for folder in folder_paths:
        folder_path = os.path.join('deploying', folder)
        source_folder = os.path.abspath(folder_path)
        destination_folder = os.path.join(subfolder_path, os.path.basename(folder_path))
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
        print(f'Folder "{folder}" copied to {destination_folder}')
        for root, dirs, files in os.walk(source_folder, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)