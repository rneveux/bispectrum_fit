import os
import argparse
from classy import Class
import yaml
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.stats import gaussian_kde
from iminuit import Minuit
import bicker.emulator as BICKER
import hitomipy
from powerbispectrum import ComputePowerBiSpectrum
import pocomc as pc
import time
from multiprocessing import Pool
import matryoshka.emulator as matry

group = [
            ['c2_b2_f', 'c2_b1_b2',  'c2_b1_b1',  'c2_b1_f', 'c2_b1_f', 'c1_b1_b1_f', 
                'c1_b2_f', 'c1_b1_b2', 'c1_b1_b1', 'c2_b1_b1_f', 'c1_b1_f'],
            ['c2_b1_f_f', 'c1_f_f', 'c1_f_f_f', 'c2_f_f', 'c2_f_f_f', 'c1_b1_f_f'],
            ['c1_c1_f_f', 'c2_c2_b1_f',  'c2_c1_b1_f',  'c2_c1_b1', 'c2_c1_b2', 
                'c2_c2_f_f', 'c1_c1_f', 'c2_c2_b1', 'c2_c2_b2', 'c2_c2_f', 'c2_c1_b1_f', 
                'c2_c1_f', 'c1_c1_b1_f', 'c1_c1_b1', 'c1_c1_b2', 'c1_c1_f_f'],
            ['c1_c1_bG2', 'c2_c2_bG2', 'c2_c1_bG2'],
            ['c1_b1_bG2', 'c1_bG2_f', 'c2_bG2_f', 'c2_b1_bG2'],
            ['b1_f_f', 'b1_b1_f_f', 'b1_b1_b2', 'b2_f_f', 'b1_b1_b1', 'b1_b1_b1_f', 
                'b1_b1_f', 'b1_f_f_f', 'f_f_f', 'f_f_f_f', 'b1_b2_f'],
            ['bG2_f_f', 'b1_b1_bG2', 'b1_bG2_f']
        ]

group_shot = [
            'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2',
            'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2',
        ]

def log_prior(theta, params):
    """
    Calculate the logarithm of the prior probability.

    Parameters
    ----------
    theta : list
        Model parameter values.
    params : dict
        Utility parameters.
        params['sorted'] includes model parameter names.
        params['prior'] is a dictionary containing the 'type'
        (Uniform 'Uni' or Gaussian 'Gauss') and the 'lim'
        (min and max for 'Uni'; mean and standard deviation for 'Gauss')
        for each parameter.

    Returns
    -------
    log_prior_prob : float
        Natural logarithm of the prior probability.
    """

    log_prior_prob = 0.

    for par, value in zip(params['sorted'], theta):
        prior_type = params['prior'][par]['type']
        if prior_type == 'Uni':
            if params['prior'][par]['lim'][0] < value < params['prior'][par]['lim'][1]:
                continue
            else:
                log_prior_prob = -np.inf
        elif prior_type == 'Gauss':
            log_prior_prob -= 0.5 * ((value - params['prior'][par]['lim'][0]) / params['prior'][par]['lim'][1]) ** 2

    return log_prior_prob

def log_likelihood(theta, data, icov, params):
    """
    Calculate the logarithm of the likelihood assuming Gaussianity.

    Parameters
    ----------
    theta : list
        Model parameter values.
    data : np.array
        Full data vector to fit.
    icov : np.array
        Inverse of the covariance matrix.
    params : dict
        Utility parameters.
        params['sorted'] includes model parameter names.

    Returns
    -------
    log_likelihood : float
        Natural logarithm of the likelihood.
    """

    params['values'].update({par: value for par, value in zip(params['sorted'], theta)})
    model_vect = model(params)

    log_likelihood = -0.5 * np.dot(np.dot((model_vect - data).T, icov), (model_vect - data))

    return log_likelihood

def log_posterior(theta, data, icov):
    """
    Calculate the logarithm of the posterior probability.

    Parameters
    ----------
    theta : list
        Model parameter values.
    data : np.array
        Full data vector to fit.
    icov : np.array
        Inverse of the covariance matrix.

    Returns
    -------
    log_posterior_prob : float
        Natural logarithm of the posterior probability.
    """

    log_prior_prob = log_prior(theta, utils)
    if log_prior_prob == -np.inf:
        log_posterior_prob = log_prior_prob
    else:
        log_posterior_prob = log_prior_prob + log_likelihood(theta, data, icov, utils)

    return log_posterior_prob


def logpost_minuit(theta):
    """Calculation of the logarithm of the posterior probability.
    Parameters
    ----------
    theta : list
        model parameter values.
    utils : dict
        Utility parameters.
    data : np.array
        Full data vector to fit.
    icov : np.array
        Inverse of the covariance matrix.
    Returns
    -------
    log posterior probability : float
        Natural logarithm of the posterior probability.
    """

    lprior = log_prior(theta, utils)
    if lprior == -np.inf:
        post = lprior
    else:
        post = -2 * (lprior + log_likelihood(theta, data, icov, utils))
    
    return post


def logprior_poco(theta):
    """Calculation of the logarithm of the prior probability.
    Parameters
    ----------
    theta : list
        model parameter values.
    utils : dict
        Utility parameters.
        utils['sorted'] includes model parameter names.
        utils['prior'] is a dictionary containing 
        the 'type' (Uniform 'Uni' or Gaussian 'Gauss') 
        and the 'lim' (min and max for 'Uni'; mean and standard deviation for 'Gauss') 
        for each parameter.
    Returns
    -------
    log prior probability : float
        Natural logarithm of the prior probability.
    """

    lp = 0.

    for par, value in zip(utils['sorted'], theta):
        prior_type = utils['prior'][par]['type']
        if prior_type == 'Uni':
            if utils['prior'][par]['lim'][0] < value < utils['prior'][par]['lim'][1]:
                continue 
            else:
                lp = -np.inf
        elif prior_type == 'Gauss':
            lp -= 0.5 * ((value - utils['prior'][par]['lim'][0]) / utils['prior'][par]['lim'][1]) ** 2

    return lp


def loglike_poco(theta, data, icov):
    """Calculation of the logarithm of the likelihood. Assuming Gaussianity.
    Parameters
    ----------
    theta : list
        Model parameter values.
    data : np.array
        Full data vector to fit.
    icov : np.array
        Inverse of the covariance matrix.
    utils : dict
        Utility parameters.
        utils['values'] includes model parameter values.
        utils['sorted'] includes model parameter names.
    Returns
    -------
    log likelihood : float
        Natural logarithm of the likelihood.
    """
    utils['values'].update({par: value for par, value in zip(utils['sorted'], theta)})
    #t0 = time.time()
    model_vect = model(utils)
    #t1 = time.time()
    diff = model_vect - data
    
    #print('tot: ', t1-t0)

    ll = -0.5 * np.dot(np.dot(diff.T, icov), diff)

    return ll


def model(params):
    """Calculation of power spectrum and bispectrum model.
    Parameters
    ----------
    params : dict
        Utility parameters.
        params['estimator'] : list, 'Pk' and/or 'Bk'.
        params['cosmo_inference'] : boolean, if True, cosmological parameters vary.
        params['with_kernels'] : boolean, if True, kernel decomposition is used.
        params['z_eff'] : float, effective redshift of the analysis.
        params['AP_fid'] : dict, values of angular distance and Hubble parameter in fiducial cosmology.
        params['values'] : list of model parameter values.
        params['k_fit'] : dict, wavenumbers used in the analysis for each multipole.
        params['multipoles_to_use'] : list, multipole needed to compute to account for window function and Alcock-Paczynski (AP) approximation.
        params['TT'] : dict, pre-computed AP distortion approximation.
    Returns
    -------
    model_vect : np.array
        Full model vector to compare to the data.
    """

    params['values'].setdefault('b1', 1)
    params['values'].setdefault('b2', 0)
    params['values'].setdefault('bG2', 0)
    params['values'].setdefault('bGamma3', 0)
    params['values'].setdefault('c0', 0)
    params['values'].setdefault('c2pp', 0)
    params['values'].setdefault('c4pp', 0)
    params['values'].setdefault('c1', 0)
    params['values'].setdefault('c2', 0)
    params['values'].setdefault('ch', 0)
    params['values'].setdefault('Pshot', 0)
    params['values'].setdefault('a0', 0)
    params['values'].setdefault('Bshot', 0)
    params['values'].setdefault('fnlequi', 0)
    params['values'].setdefault('fnlortho', 0)

    if params['cosmo_inference']:
        params['values_cosmo'] = {}
        params['values_cosmo_list'] = []
        for key in ['omega_cdm', 'omega_b', 'h', 'ln10^{10}A_s', 'n_s']:
                params['values_cosmo'][key] = params['values'][key]
                params['values_cosmo_list'].append(params['values'][key])

        #t0 = time.time()
        #c = Class()
        #c.set(params['values_cosmo'])
        #c.compute()
        #t1 = time.time()
        #print('class: ', t1-t0)

        if params['ap_approx']:
            params['alpha_perp'] = c.angular_distance(params['z_eff']) / params['AP_fid']['DA']
            params['alpha_parallel'] = params['AP_fid']['H'] / c.Hubble(params['z_eff'])
            epsilon, alpha = epsilon_alpha(params['alpha_parallel'], params['alpha_perp'])
            kp = {}
            for est in params['estimator']:
                kp[est] = k_prime(params['k_model'][est], epsilon=epsilon, alpha=alpha)
        '''t0 = time.time()
        #params['fz'] = c.scale_independent_growth_factor_f(params['z_eff'])

        h = params['values_cosmo']['h']
        Ob = params['values_cosmo']['omega_b']/h**2
        Om = params['values_cosmo']['omega_cdm']/h**2+Ob
        Neff = params['cosmo_fid']['N_ur'] + 1
        w0 = -1
        t2 = time.time()
        D_emul = params['emu_growth'].emu_predict([Om,Ob,0,h,0,Neff,w0])[0]
        t3 = time.time()
        print('emul: ', t3-t2)
        
        zz = np.linspace(0, 2, 200)
        
        logD = np.log(D_emul)
        loga = np.log(1/(1+zz))

        f_emul = np.gradient(logD, loga)

        funD = interpolate.interp1d(zz, D_emul, fill_value="extrapolate", kind="cubic")
        funf = interpolate.interp1d(zz, f_emul, fill_value="extrapolate", kind="cubic")

        params['Dz'] = funD(params['z_eff'])
        params['fz'] = funf(params['z_eff'])
        t1 = time.time()
        print('f: ', t1-t0)'''

    model_vect = []

    if 'Pk' in params['estimator']:
        if params['cosmo_inference']=='classpt':
            c = Class()
            c.set(params['cosmo_classpt'])
            for key in ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10^{10}A_s']:
                c.set({key: params['values'][key]})
            c.set({'non linear':'PT',
                'IR resummation':'Yes',
                'Bias tracers':'Yes',
                'cb':'Yes',
                'RSD':'Yes',
                'AP':'Yes',
                'Omfid':params['cosmo_fid']['omega_cdm']/params['cosmo_fid']['h']**2,
                'output':'mPk',
                'z_pk':params['z_eff']
            })
            c.compute()
            c.initialize_output(params['k_model']['Pk']*params['values']['h'], params['z_eff'], len(params['k_model']['Pk']))
            params['class_pt'] = c
        pk_evaluation = {}
        for ell in params['multipoles']['Pk']:
            #t0 = time.time()
            pk_evaluation[ell] = pk_model_evaluation(ell, params)
            #t1 = time.time()
            #print(f'eval {ell}: ', t1-t0)
        if params['ap_approx']:
            pk_evaluation = pk_ap_approx(pk_evaluation, epsilon, kp, params)
        if params['window']:
            model_vect.extend(window_mult(pk_evaluation, params))
        else:
            for ell in params['multipoles']['Pk']: 
                #t0 = time.time()
                if 'emu_pk' in params: keval = params['emu_pk'][ell].kbins
                else: keval = params['k_model']['Pk']
                f_pk = interpolate.interp1d(keval, pk_evaluation[ell], fill_value="extrapolate", kind="cubic")
                model_vect.extend(f_pk(params['k_fit']['Pk'][ell]))
                #t1 = time.time()
                #print(f'interpol {ell}: ', t1-t0)

    if 'Bk' in params['estimator']:
        bk_tmp = {}
        bk_evaluation = {}
        #t0 = time.time()
        if len(params['multipoles_to_use']['Bk']) > len(params['multipoles']['Bk']):
            params['sigma8z'] = params['values']['sigma8'] * c.scale_independent_growth_factor(params['z_eff'])
        #t1 = time.time()
        #print('s8: ', t1-t0)
        for ell in params['multipoles_to_use']['Bk']:
            if ell in params['multipoles']['Bk']:
                #t0 = time.time()
                bk_evaluation[ell] = bk_model_evaluation(ell, params)
                #t1 = time.time()
                #print(f'model {ell}: ', t1-t0)
            else:
                bk_evaluation[ell] = bk_model_evaluation(ell, params, full_fit=False)
        #bk_evaluation[ell][:8] *= corr_k_disc[ell]
        #if params['cosmo_inference']:
        #for ell in params['multipoles']['Bk']:
        #    bk_evaluation = bk_ap_approx(bk_evaluation, epsilon, kp, params)
        #window convolution a coder
        for ell in params['multipoles']['Bk']:  
            #t0 = time.time()
            if (params['k_fit']['Bk'][ell]==params['k_model']['Bk']).all():
                mask = (params['k_edges']['Bk'][ell][0] <= params['k_emul']) & (params['k_edges']['Bk'][ell][1] >= params['k_emul'])
                model_vect.extend(bk_evaluation[ell][mask])
            else:
                if 'k_emul' in params: 
                    mask = (params['k_edges']['Bk'][ell][0] <= params['k_emul']) & (params['k_edges']['Bk'][ell][1] >= params['k_emul'])
                    model_vect.extend(bk_evaluation[ell][mask])
                else: keval = params['k_model']['Bk']
                f_bk = interpolate.interp1d(keval, bk_evaluation[ell], fill_value="extrapolate", kind="cubic")
                model_vect.extend(f_bk(params['k_fit']['Bk'][ell]))
            #t1 = time.time()
            #print(f'interp {ell}: ', t1-t0)
             
    return np.array(model_vect)

def pk_model_evaluation(ell, params):
    """
    Evaluation of the power spectrum model.

    Parameters:
    ell (str): Multipole.
    params (dict): Util parameters.
        params['cosmo_inference'] (bool): If True, cosmological parameters vary.
        params['z_eff'] (float): Effective redshift of the analysis.
        params['values'] (list): List of model parameter values.

    Returns:
    np.array: Evaluation of the bispectrum model.
    """

    b1 = params['values']['b1']
    b2 = params['values']['b2']
    bG2 = params['values']['bG2']
    bGamma3 = params['values']['bGamma3']
    c0 = params['values']['c0']
    c2pp = params['values']['c2pp']
    c4pp = params['values']['c4pp']
    ch = params['values']['ch']
    Pshot = params['values']['Pshot']
    a0 = params['values']['a0']
    fnlequi = params['values']['fnlequi']  
    fnlortho = params['values']['fnlortho'] 
                                                                     
    mean_density = params['mean_density']
    fz = params['fz']
    
    if params['cosmo_inference']==True:
        Pstoch = 0                                        
        if ell == '0':
            #cl = c0 + fz / 3 * c1 + fz ** 2 / 5 * c2
            cl = c0
            Pstoch = (1 + Pshot + a0*params['emu_pk'][ell].kbins**2) / mean_density
        elif ell == '2':
            #cl = c1 + fz * 6 / 7 * c2
            cl = c2pp
        elif ell == '4':
            cl = c4pp
        pk_model = params['emu_pk'][ell].emu_predict(params['values_cosmo_list'], np.array([b1,b2,bG2,bGamma3,ch,cl]))[0] + Pstoch
    elif params['cosmo_inference']=='classpt':
        if ell == '0':
            if Pshot==0: Pshot_norm = 0
            else: Pshot_norm = (1 + Pshot + a0*params['k_model']['Pk']**2) / mean_density
            pk_model = params['class_pt'].pk_gg_l0(b1, b2, bG2, bGamma3, c0, 0, ch)
            pk_model += Pshot_norm
            if fnlequi != 0:
                pk_model += fnlequi * params['class_pt'].pk_gg_fNL_l0(b1, b2, bG2)
            if fnlortho != 0:
                pk_model += fnlortho * params['class_pt'].pk_gg_fNL_l0_ortho(b1, b2, bG2)
        elif ell == '2':
            cs2 = c2pp
            pk_model = params['class_pt'].pk_gg_l2(b1, b2, bG2, bGamma3, cs2, ch)
            if fnlequi != 0:
                pk_model += fnlequi * params['class_pt'].pk_gg_fNL_l2(b1, b2, bG2)
            if fnlortho != 0:
                pk_model += fnlortho * params['class_pt'].pk_gg_fNL_l2_ortho(b1, b2, bG2)
        elif ell == '4':
            cs4 = c4pp
            pk_model = params['class_pt'].pk_gg_l4(b1, b2, bG2, bGamma3, cs4, ch)
            if fnlequi != 0:
                pk_model += fnlequi * params['class_pt'].pk_gg_fNL_l4(b1, b2, bG2)
            if fnlortho != 0:
                pk_model += fnlortho * params['class_pt'].pk_gg_fNL_l4_ortho(b1, b2, bG2)
    elif 'f' in params['values']:
        f = params['values']['f']
        M_mult = params['class_fid'].get_pk_mult(params['k_model']['Pk']*params['cosmo_fid']['h'], 
                                                 params['z_eff'], len(params['k_model']['Pk']))
        pk_model = rebuild_pk_class_pt(M_mult, bias_params, params['fz'], params['cosmo_fid']['h'], params['k_model']['Pk'], ells=[int(ell)])
        if ell=='0':
            if Pshot==0: Pshot_norm = 0
            else: Pshot_norm = (1 + Pshot + a0*params['k_model']['Pk']**2) / mean_density
            pk_model += Pshot_norm
    else:
        if ell == '0':
            #cs0 = c0 + fz / 3 * c1 + fz ** 2 / 5 * c2
            cs0 = c0
            if Pshot==0: Pshot_norm = 0
            else: Pshot_norm = (1 + Pshot + a0*params['k_model']['Pk']**2) / mean_density
            pk_model = params['class_fid'].pk_gg_l0(b1, b2, bG2, bGamma3, cs0, 0, ch)
            pk_model += Pshot_norm
            if fnlequi != 0 or fnlortho != 0:
                tmp = params['class_fid'].pk_gg_fNL_l0(b1, b2, bG2)
                tmp[np.isnan(tmp)] = 0
                pk_model += fnlequi * tmp
                pk_model += fnlortho * params['class_fid'].pk_gg_fNL_l0_ortho(b1, b2, bG2)

        elif ell == '2':
            #cs2 = c1 + fz * 6 / 7 * c2
            cs2 = c2pp
            pk_model = params['class_fid'].pk_gg_l2(b1, b2, bG2, bGamma3, cs2, ch)
            if fnlequi != 0 or fnlortho != 0:
                pk_model += fnlequi * params['class_fid'].pk_gg_fNL_l2(b1, b2, bG2)
                pk_model += fnlortho * params['class_fid'].pk_gg_fNL_l2_ortho(b1, b2, bG2)
        elif ell == '4':
            cs4 = c4pp
            pk_model = params['class_fid'].pk_gg_l4(b1, b2, bG2, bGamma3, cs4, ch)
            if fnlequi != 0 or fnlortho != 0:
                pk_model += fnlequi * params['class_fid'].pk_gg_fNL_l4(b1, b2, bG2)
                pk_model += fnlortho * params['class_fid'].pk_gg_fNL_l4_ortho(b1, b2, bG2)

    return pk_model

def pk_model_stand_alone(ell, mod='emulator', fz_compute='emulator', redshift=.8, mean_density=1e-3, k=np.linspace(.01,.15,200),
                         Omfid=0, gg=False, emu=False, cache_path=False, **params):
    
    params.setdefault('b1', 1)
    params.setdefault('b2', 0)
    params.setdefault('bG2', 0)
    params.setdefault('bGamma3', 0)
    params.setdefault('c0', 0)
    params.setdefault('c2pp', 0)
    params.setdefault('c4pp', 0)
    params.setdefault('c1', 0)
    params.setdefault('c2', 0)
    params.setdefault('ch', 0)
    params.setdefault('Pshot', 0)
    params.setdefault('a0', 0)
    params.setdefault('Bshot', 0)
    params.setdefault('fnlequi', 0)
    params.setdefault('fnlortho', 0)
    
    if fz_compute=='emulator':

        if not gg:
            gg = matry.Growth()

        h = params['h']
        ob = params['omega_b']
        ocdm = params['omega_cdm']

        Ob = ob/h**2
        Om = ocdm/h**2+Ob
        Neff = 3.0328
        w0 = -1
        D_emul = gg.emu_predict([Om,Ob,0,h,0,Neff,w0])[0]

        zz = np.linspace(0, 2, 200)

        logD = np.log(D_emul)
        loga = np.log(1/(1+zz))

        f_emul = np.gradient(logD, loga)

        funf = interpolate.interp1d(zz, f_emul, fill_value="extrapolate", kind="cubic")

        fz = funf(redshift)

    else:
        params_cosmo = {
            'h': params['h'],
            'omega_b': params['omega_b'],
            'omega_cdm': params['omega_cdm'],
            'n_s': params['n_s'],
            'ln10^{10}A_s': params['ln10^{10}A_s'],
            'N_ncdm': 1.,
            'omega_ncdm': .0006442,
            'N_ur': 2.0328,
            }
        c = Class()
        c.set(params_cosmo)
        c.compute()

        fz = c.scale_independent_growth_factor_f(redshift)
    
    if mod=='emulator':
    
        if not emu:
            emu = BICKER.power(ell, cache_path)
        
        Pstoch = 0                                        
        if ell == 0:
            cl = params['c0'] + fz / 3 * params['c1'] + fz ** 2 / 5 * params['c2']
            Pstoch = (1 + params['Pshot'] + params['a0']*emu.kbins**2) / mean_density
        elif ell == 2:
            cl = params['c1'] + fz * 6 / 7 * params['c2']
        elif ell == 4:
            cl = params['c2']                                   
        pk_model_wk = emu.emu_predict([params['omega_cdm'], params['omega_b'], params['h'], params['ln10^{10}A_s'], params['n_s']], np.array([params['b1'],params['b2'],params['bG2'],params['bGamma3'],params['ch'],cl]))[0] + Pstoch
        
        fun_pk = interpolate.interp1d(emu.kbins, pk_model_wk, fill_value="extrapolate", kind="cubic")

        pk_model = fun_pk(k)
        
    else:
        if Omfid != 0: AP = 'Yes'
        else: AP = 'No'
        c = Class()
        c.set(params_cosmo)
        c.set({'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes',
            'RSD':'Yes',
            'AP':AP,
            'Omfid':Omfid,
             'output':'mPk',
               'z_pk':redshift
           })
        c.compute()
        c.initialize_output(k*params['h'], redshift, len(k))
        if ell == 0:
            cs0 = params['c0'] #params['c0'] + fz / 3 * params['c1'] + fz ** 2 / 5 * params['c2']
            if params['Pshot']==0: Pshot_norm = 0
            else: Pshot_norm = (1 + params['Pshot'] + params['a0']*k**2) / mean_density
            pk_model = c.pk_gg_l0(params['b1'],params['b2'],params['bG2'],params['bGamma3'],cs0,0,params['ch'])
            pk_model += Pshot_norm
            if params['fnlequi'] != 0:
                pk_model += params['fnlequi'] * c.pk_gg_fNL_l0(params['b1'], params['b2'], params['bG2'])
            if params['fnlortho'] != 0:
                pk_model += params['fnlortho'] * c.pk_gg_fNL_l0_ortho(params['b1'], params['b2'], params['bG2'])
        elif ell == 2:
            cs2 = params['c2pp'] #params['c1'] + fz * 6 / 7 * params['c2']
            pk_model = c.pk_gg_l2(params['b1'],params['b2'],params['bG2'],params['bGamma3'],cs2,params['ch'])
            if params['fnlequi'] != 0:
                pk_model += params['fnlequi'] * c.pk_gg_fNL_l2(params['b1'], params['b2'], params['bG2'])
            if params['fnlortho'] != 0:
                pk_model += params['fnlortho'] * c.pk_gg_fNL_l2_ortho(params['b1'], params['b2'], params['bG2'])
        elif ell == 4:
            cs4 = params['c4pp']         
            pk_model = c.pk_gg_l4(params['b1'],params['b2'],params['bG2'],params['bGamma3'],cs4,params['ch'])
            if params['fnlequi'] != 0:
                pk_model += params['fnlequi'] * c.pk_gg_fNL_l4(params['b1'], params['b2'], params['bG2'])
            if params['fnlortho'] != 0:
                pk_model += params['fnlortho'] * c.pk_gg_fNL_l4_ortho(params['b1'], params['b2'], params['bG2'])
            
    return pk_model

def rebuild_pk_class_pt(mult_kernels, bias_params, f_fid, h, k, ells=[0,2,4], density=1e-3):
    
    kernels = mult_kernels.copy()
    bias_params['bGG'] = 1.
    
    bias_kernels = { 
                0:{
                    15: 'f_f',
                    21: 'f_f',
                    16: 'f_b1',
                    22: 'f_b1',
                    17: 'b1_b1',
                    23: 'b1_b1', 
                    1: 'b2_b2',
                    30: 'b1_b2',
                    31: 'f_b2',
                    32: 'b1_bG2',
                    33: 'f_bG2',
                    4: 'b2_bG2',
                    5: 'bG2_bG2',
                    11: 'c0',
                    7: 'b1_bGG',
                    8: 'f_bGG',
                    13: 'spe',
                   },

                2:{
                    18: 'f_f',
                    24: 'f_f',
                    19: 'f_b1',
                    25: 'f_b1',
                    26: 'b1_b1',
                    34: 'b1_b2',
                    35: 'f_b2',
                    36: 'b1_bG2',
                    37: 'f_bG2',
                    12: 'f_c2pp',
                    9: 'f_bGG',
                    13: 'spe',
                    },

                4:{
                    20: 'f_f',
                    27: 'f_f',
                    28: 'f_b1',
                    29: 'b1_b1',
                    38: 'f_b2',
                    39: 'f_bG2',
                    13: 'spe',
                    }
              }
    
    pk_model = {}
    for ell in ells:
        pk_model[ell] = np.zeros_like(kernels[17])
        
        for ker in bias_kernels[ell]:
            if ker == 13:
                if ell == 0:
                    kernels[40] = kernels[ker] * bias_params['f']**2.*bias_params['ch']*k**2.*(bias_params['f']**2./9. + 2.*bias_params['f']*bias_params['b1']/7. + bias_params['b1']**2./5)*(35./8.)*h
                    pk_model[ell] += kernels[40]
                if ell == 2:
                    kernels[41] = kernels[ker] * bias_params['f']**2.*bias_params['ch']*k**2.*((bias_params['f']**2.*70. + 165.*bias_params['f']*bias_params['b1']+99.*bias_params['b1']**2.)*4./693.)*(35./8.)*h
                    pk_model[ell] += kernels[41]
                if ell == 4:
                    kernels[42] = kernels[ker] * bias_params['f']**2.*bias_params['ch']*k**2.*((bias_params['f']**2.*210. + 390.*bias_params['f']*bias_params['b1']+143.*bias_params['b1']**2.)*8./5005.)*(35./8.)*h
                    kernels[ker] *= 2*(bias_params['f']/f_fid)**2.*bias_params['c4pp']*h
                    pk_model[ell] += kernels[ker] + kernels[42]
            else:
                if 'bGG' in bias_kernels[ell][ker]:
                    kernels[ker] *= (2*bias_params['bG2']+.8*bias_params['bGamma3'])
                for b in bias_params:
                    kernels[ker] *= bias_params[b]**bias_kernels[ell][ker].count(b)
                    if b == 'f':
                        kernels[ker] /= f_fid**bias_kernels[ell][ker].count('f')
                    if (b=='c0'and 'c0' in bias_kernels[ell][ker]) or (b=='c2pp'  and 'c2pp' in bias_kernels[ell][ker]):
                        kernels[ker] *= 2
                    if b=='b2' and bias_kernels[ell][ker] == 'b2_b2':
                        kernels[ker] *= .25
                if ker in [11,12]:
                    kernels[ker] *= h
                else: kernels[ker] *= h**3
                pk_model[ell] += kernels[ker]
                
    return pk_model

def bk_model_evaluation(ell, params, full_fit=True):
    """
    Evaluation of the bispectrum model.

    Parameters:
    ell (str): Three multipoles of the bispectrum ell1, ell2, ELL (e.g., `000` or `202`).
    k (np.array): Wavenumber for the evaluation of the model.
    params (dict): Util parameters.
        params['cosmo_inference'] (bool): If True, cosmological parameters vary.
        params['with_kernels'] (bool): If True, kernel decomposition is used.
        params['z_eff'] (float): Effective redshift of the analysis.
        params['values'] (list): List of model parameter values.
        params['Bk_kernels'] (dict): Dictionary of precomputed kernel bispectra.
        params['mean_density'] (float): Mean density of the analysis.
        params['fz'] (float): Redshift evolution parameter.
        params['s8'] (float): Power spectrum normalization.
        params['params_zeros'] (set): Set of parameters that are set to zero.
        params['alpha_perp'] (float): Perpendicular scale factor.
        params['alpha_parallel'] (float): Parallel scale factor.
    full_fit (bool): If True, use full fitting mode for kernel bispectra.

    Returns:
    np.array: Evaluation of the bispectrum model.
    """

    b1 = params['values']['b1']
    b2 = params['values']['b2']
    bG2 = params['values']['bG2']
    c1 = params['values']['c1']
    c2 = params['values']['c2']
    Pshot = params['values']['Pshot']
    Bshot = params['values']['Bshot']  
    fnlequi = params['values']['fnlequi']  
    fnlortho = params['values']['fnlortho']    

    mean_density = params['mean_density']


    if params['cosmo_inference'] and full_fit:
        #t0 = time.time()
        kernels_k, kernels_Bk = kernels_from_emulator(ell, params)
        #t1 = time.time()
        #print(f'emul {ell}: ', t1-t0)
    else:
        kernels_k = params['k_model']['Bk']
        kernels_Bk = params['Bk_kernels'][ell]

    bk_model = np.zeros(len(kernels_Bk['b1_b1_b1']))

    for b, values in kernels_Bk.items():
        
        #if not 'shot' in b:
        #    continue
        
        bias = 1

        if 'b1' in b:
            bias *= b1 ** b.count('b1')
        if 'b2' in b:
            bias *= b2
        if 'bG2' in b:
            bias *= bG2
        if 'c1' in b:
            bias *= c1 ** b.count('c1')
        if 'c2' in b:
            bias *= c2 ** b.count('c2')
        if 'Pshot' in b:
            bias *= (1 + Pshot) / mean_density
        if 'Bshot' in b:
            bias *= Bshot / mean_density
        if 'fnlequi' in b:
            bias *= fnlequi
        if 'fnlortho' in b:
            bias *= fnlortho
            
        if not full_fit:
            bias *= params['fz'] ** b.count('f')

            if 'Pshot' in b or 'Bshot' in b:
                bias *= params['sigma8z'] ** 2
            else:
                bias *= params['sigma8z'] ** 4

        bk_model += bias * values

    if Pshot!=0: bk_model += ((1+Pshot)/mean_density)**2

    return bk_model

def bk_model_stand_alone(kernels_directory = False, ell='000', mean_density=1e-3, ortho_LSS=True, **params):
    """
    Evaluation of the bispectrum model.

    Parameters:
    ell (str): Three multipoles of the bispectrum ell1, ell2, ELL (e.g., `000` or `202`).
    k (np.array): Wavenumber for the evaluation of the model.
    params (dict): Util parameters.
        params['cosmo_inference'] (bool): If True, cosmological parameters vary.
        params['with_kernels'] (bool): If True, kernel decomposition is used.
        params['z_eff'] (float): Effective redshift of the analysis.
        params['values'] (list): List of model parameter values.
        params['Bk_kernels'] (dict): Dictionary of precomputed kernel bispectra.
        params['mean_density'] (float): Mean density of the analysis.
        params['fz'] (float): Redshift evolution parameter.
        params['s8'] (float): Power spectrum normalization.
        params['params_zeros'] (set): Set of parameters that are set to zero.
        params['alpha_perp'] (float): Perpendicular scale factor.
        params['alpha_parallel'] (float): Parallel scale factor.
    full_fit (bool): If True, use full fitting mode for kernel bispectra.

    Returns:
    np.array: Evaluation of the bispectrum model.
    """

    params.setdefault('b1', 1)
    params.setdefault('b2', 0)
    params.setdefault('bG2', 0)
    params.setdefault('bGamma3', 0)
    params.setdefault('c0', 0)
    params.setdefault('c1', 0)
    params.setdefault('c2', 0)
    params.setdefault('ch', 0)
    params.setdefault('Pshot', 0)
    params.setdefault('a0', 0)
    params.setdefault('Bshot', 0)
    params.setdefault('fnlequi', 0)
    params.setdefault('fnlortho', 0)   
    
    kernel_name = [
            'b1_b1_b1', 'b1_b1_b2','b1_b1_bG2','b1_b1_f','b1_b1_b1_f','b1_b1_f_f',
            'b1_b2_f','b1_bG2_f','b1_f_f',
            'b1_f_f_f','b2_f_f','bG2_f_f','f_f_f','f_f_f_f',
            'c1_b1_b1','c1_b1_b2','c1_b1_bG2','c1_b1_f','c1_b1_b1_f','c1_b1_f_f','c1_b2_f',
            'c1_bG2_f','c1_f_f','c1_f_f_f','c1_c1_b1','c1_c1_b2','c1_c1_bG2','c1_c1_f',
            'c1_c1_b1_f','c1_c1_f_f','c2_b1_b1','c2_b1_b2','c2_b1_bG2','c2_b1_f','c2_b1_b1_f',
            'c2_b1_f_f','c2_b2_f','c2_bG2_f','c2_f_f','c2_f_f_f','c2_c1_b1','c2_c1_b2',
            'c2_c1_bG2','c2_c1_f','c2_c1_b1_f','c2_c1_f_f','c2_c2_b1','c2_c2_b2','c2_c2_bG2',
            'c2_c2_f','c2_c2_b1_f','c2_c2_f_f',
            'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2', 
            'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2',
            'fnlloc_b1_b1_b1','fnlloc_b1_b1_f','fnlloc_b1_f_f','fnlloc_f_f_f',
            'fnlequi_b1_b1_b1','fnlequi_b1_b1_f','fnlequi_b1_f_f','fnlequi_f_f_f',
            'fnlortho_b1_b1_b1','fnlortho_b1_b1_f','fnlortho_b1_f_f','fnlortho_f_f_f',]


    if kernels_directory:
        Bk_kernels = {}
        for b in kernel_name:
            if 'Pshot' in b or 'Bshot' in b: bispectrum_part = 'SN'
            elif 'fnlequi' in b or 'fnlortho' in b: bispectrum_part = 'PNG'
            else: bispectrum_part = 'tree'
            if 'fnlortho' in b and ortho_LSS:
                b = b.replace('ortho', 'ortho_LSS')
            fichier = os.path.join(kernels_directory,ell,bispectrum_part,b+'.npy')
            if os.path.exists(fichier):
                tmp = np.load(fichier,allow_pickle=True).item()
                Bk_kernels[b] = tmp['K']

    bk_model = np.zeros(len(Bk_kernels['b1_b1_b1']))

    for b, values in Bk_kernels.items():
        #if not 'shot' in b:
        #    continue
        bias = 1

        if 'b1' in b:
            bias *= params['b1'] ** b.count('b1')
        if 'b2' in b:
            bias *= params['b2']
        if 'bG2' in b:
            bias *= params['bG2']
        if 'c1' in b:
            bias *= params['c1'] ** b.count('c1')
        if 'c2' in b:
            bias *= params['c2'] ** b.count('c2')
        if 'Pshot' in b:
            bias *= (1 + params['Pshot']) / mean_density
        if 'Bshot' in b:
            bias *= params['Bshot'] / mean_density
        if 'fnlequi' in b:
            bias *= params['fnlequi']
        if 'fnlortho' in b:
            bias *= params['fnlortho']

        bk_model += bias * values

    if params['Pshot']!=0: bk_model += ((1+params['Pshot'])/mean_density)**2

    return bk_model



def kernels_from_emulator(ell, params):
    """
    Evaluation of the kernels using the Bicker emulator.

    Parameters:
    ell (str): Three multipoles of the bispectrum ell1, ell2, ELL (e.g., `000` or `202`).
    params (dict): Util parameters.
        params['kernels_to_compute'] (list): Kernels to compute. Default is 'all'.
        params['values'] (list): List of model parameter values.

    Returns:
    np.array: Wavenumber of the kernel evaluation.
    dict: Evaluation of each kernel on emu.kbins.
    """

    '''params_cosmo_list = [
        params['values']['omega_cdm'],
        params['values']['omega_b'],
        params['values']['h'],
        params['values']['ln10^{10}A_s'],
        params['values']['n_s']
    ]'''
    params_cosmo_list = [
        params['values']['h'],
        params['values']['ln10^{10}A_s'],
        params['values']['omega_cdm'],
    ]
    
    kernels = {}
    for gp in params['emu'][ell].keys():
        predictions = params['emu'][ell][gp].emu_predict(params_cosmo_list, split=True)

        if gp==8:
            for i, k in enumerate(group_shot):
                kernels[k] = np.reshape(predictions[i], predictions[i].shape[1])
        else:
            for i, k in enumerate(group[gp]):
                kernels[k] = np.reshape(predictions[i], predictions[i].shape[1])

    return params['emu'][ell][gp].kbins, kernels

def window_convol(k_in, k_out, pks, f_win, ells, NNN=1000):
    """
    Convolution of the power spectrum model with window functions.

    Parameters:
    k_in (np.array): Input wavenumber array.
    k_out (np.array): Output wavenumber array.
    pks (list): List of multipoles to consider.
    f_win (dict): Dictionary of window function interpolators.
    ells (list): List of multipoles for xi_raw.
    NNN (int): Number of points for interpolation.

    Returns:
    dict: Output power spectrum values for each multipole.
    """

    kbin_for_zeta = np.logspace(np.log(k_in[0]), np.log(k_in[-1]), NNN, base=np.e)
    r = np.zeros(NNN)

    xi_raw = {}
    window = {}
    xi = {}
    pk_out = {}

    for ell in ells:
        xi_raw[ell] = np.zeros(NNN)
        f_pk_raw = interpolate.interp1d(k_in, pk[ell], fill_value="extrapolate", kind="cubic")
        pk_for_zeta = f_pk_raw(kbin_for_zeta)
        hitomipy.pk2xi_py(NNN, kbin_for_zeta, pk_for_zeta, r, xi_raw[ell])

    if 0 in pks:
        window[0] = f_win[0](r)
        window[2] = f_win[2](r)
        window[4] = f_win[4](r)
        xi[0] = xi_raw[0] * window[0] \
                + 1 / 5 * xi_raw[2] * window[2] \
                + 1 / 9 * xi_raw[4] * window[4]
    if 2 in pks:
        window[6] = f_win[6](r)
        xi[2] = xi_raw[0] * window[2] \
                + xi_raw[2] * (window[0] + 2 / 7 * window[2] + 2 / 7 * window[4]) \
                + xi_raw[4] * (2 / 7 * window[2] + 100 / 693 * window[4] + 25 / 143 * window[6])
    if 4 in pks:
        window[8] = f_win[8](r)
        xi[4] = xi_raw[0] * window[4] \
                + xi_raw[2] * (18 / 35 * window[2] + 20 / 77 * window[4] + 45 / 143 * window[6]) \
                + xi_raw[4] * (
                        window[0] + 20 / 77 * window[2] + 162 / 1001 * window[4] + 20 / 143 * window[6] + 490 / 2431 * window[8])

    for ell in ells:
        k_temp = np.zeros(NNN)
        pk_temp = np.zeros(NNN)
        hitomipy.xi2pk_py(NNN, r, xi[ell], k_temp, pk_temp)
        f_pk = interpolate.interp1d(k_temp, pk_temp, fill_value="extrapolate", kind="cubic")
        pk_out[ell] = f_pk(k_out)

    return pk_out


def window_mult(pk, params):
    """
    Multiply power spectrum by a window function matrix.

    Parameters:
    pk (dict): Dictionary of power spectrum values for each multipole.
    params (dict): Parameters for window function multiplication.

    Returns:
    np.array: Output power spectrum values after window function multiplication.
    """

    pk_vec = np.concatenate([pk[ell] for ell in params['multipoles']['Pk']])
    window = np.loadtxt(params['window'])
    poles_selection = [False] * 5
    for i in np.arange(5):
        if str(i) in params['multipoles']['Pk']:
            poles_selection[i] = True
    fit_selection = np.repeat(poles_selection, 40)
    win = window[np.ix_(fit_selection, fit_selection)]
    pk_out = np.matmul(win, pk_vec)

    k_range = params['k_data_full']
    k_selection = np.concatenate(
        [np.logical_and(params['k_edges']['Pk'][ell][0] < k_range, params['k_edges']['Pk'][ell][1] > k_range) for ell in params['multipoles']['Pk']])

    return pk_out[k_selection]

def epsilon_alpha(alpha_parallel, alpha_perp):
    """
    Change of the basis for the Alcock-Paczynski effect.
    Converts from alpha_parallel, alpha_perp to epsilon, alpha.

    Parameters:
    alpha_parallel (float): AP scaling parallel to the line-of-sight.
    alpha_perp (float): AP scaling perpendicular to the line-of-sight.

    Returns:
    tuple: Tuple containing epsilon (AP distortion) and alpha (isotropic AP scaling).
    """

    epsilon = (alpha_parallel / alpha_perp) ** (1 / 3) - 1
    alpha = (alpha_perp ** 2 * alpha_parallel) ** (1 / 3)

    return epsilon, alpha

def k_prime(k, alpha_parallel=1, alpha_perp=1, epsilon=0, alpha=1):
    """
    Isotropic rescaling of the wavenumber.
    Uses either alpha_parallel, alpha_perp or epsilon, alpha.

    Parameters:
    k (np.array): Wavenumber in the fiducial cosmology.
    alpha_parallel (float): AP scaling parallel to the line-of-sight. Default is 1.
    alpha_perp (float): AP scaling perpendicular to the line-of-sight. Default is 1.
    epsilon (float): AP distortion. Default is 0.
    alpha (float): Isotropic AP scaling. Default is 1.

    Returns:
    np.array: Rescaled wavenumber.
    """

    if epsilon == 0:
        epsilon, alpha = epsilon_alpha(alpha_parallel, alpha_perp)

    kp = k * (1 + epsilon) / alpha * (1 + 1 / 3 * ((1 + epsilon) ** (-6) - 1)) ** (1 / 2)

    return kp

def pk_ap_approx(pk, eps, kp, params):
    
    result = {}
    pd = {}
    pkkp = {}
    for ell in [0,2,4]:
        fun[ell] = {}
        result[ell] = np.zeros_like(pk[ell])
        f_pk = interpolate.interp1d(params['k_model']['Pk'], pk[ell], fill_value="extrapolate", kind="cubic")
        pkkp[ell] = f_pk(kp)
        pd[ell] = {n: derivative_n(n, pkkp, kp) for n in [0,1,2,3]}

    for ell in [0,2,4]:    
        for ell_dash in [0,2,4]:
            for n in [0,1,2,3]:
                fun = interpolate.interp1d(
                    params['TT'][f'{ell}_{ell_dash}_{n}']['epsilon'], 
                    params['TT'][f'{ell}_{ell_dash}_{n}']['T'], 
                    fill_value="extrapolate", 
                    kind="cubic"
                )
                result[ell] += fun(eps)*pd[ell_dash][n]*kp**n
        result[ell] /= (params['alpha_perp']**2*params['alpha_parallel'])

    return result

def bk_ap_approx(bk, eps, kp, params):
    
    result = {}
    bd = {}
    bkkp = {}
    for ell in params['multipoles_to_use']['Bk']:
        if ell in params['multipoles']['Bk']:
            result[ell] = np.zeros_like(bk[ell])
        fun[ell] = {}
        f_bk = interpolate.interp1d(params['k_model']['Bk'], bk[ell], fill_value="extrapolate", kind="cubic")
        bkkp[ell] = f_bk(kp)
        bd[ell] = {(n,m): derivative_nm(n, m, bkkp, kp) for n in [0,1] for m in [0,1]}

    for ell in params['multipoles']['Bk']:    
        for ell_dash in params['multipoles_to_use']['Bk']:
            for n in [0,1]:
                for m in [0,1]:
                    fun = interpolate.interp1d(
                        params['TT'][f'{ell}_{ell_dash}_{n}_{m}']['epsilon'], 
                        params['TT'][f'{ell}_{ell_dash}_{n}_{m}']['T'], 
                        fill_value="extrapolate", 
                        kind="cubic"
                    )
                    result[ell] += fun(eps)*bd[ell_dash][(n,m)]*kp**n
        result[ell] /= (params['alpha_perp']**4*params['alpha_parallel']**2)

    return result

def derivative_n(n, pktmp, k):
    """Calculates the derivative based on the order 'n'"""
    pd = pktmp
    for _ in range(n):
        pd = np.gradient(pd, k)

    return pd

def derivative_nm(n, m, pktmp, k):
    """Calculates the derivative based on the order 'n' 'm'"""
    bd = bktmp
    for _ in range(m):
        bd = np.gradient(bd, k)
    bd *= k**m
    for _ in range(n):
        bd = np.gradient(bd, k)

    return bd

def class_sigma8(params_cosmo_s8, redshift, PT=False):
    
    params_cosmo_s8.update({'z_pk':redshift})

    c_s8 = Class()
    c_s8.set(params_cosmo_s8)

    c_s8.compute()

    As_rescale = c_s8.A_s()/c_s8.sigma8()**2*params_cosmo_s8['sigma8']**2

    params_cosmo = params_cosmo_s8.copy()

    del params_cosmo['sigma8']
    
    params_cosmo['A_s'] = As_rescale
    
    c = Class()
    c.set(params_cosmo)
    
    if PT:
        c.set({'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes',
            'RSD':'Yes',
            'AP':'No',
           })

    c.compute()
    
    return c

def class_As(params_cosmo, redshift, PT=False):

    c = Class()
    c.set(params_cosmo)
    c.set(
        {'N_ur':2.0328, 
        'N_ncdm':1, 
        'omega_ncdm': 0.0006442,
        'z_pk':redshift}
        )

    if PT:
        c.set({'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes',
            'RSD':'Yes',
            'AP':'No',
           })

    c.compute()

    return c

class PrepareFit(ComputePowerBiSpectrum):
    """
    Analysis of galaxy clustering through multipoles of power spectrum and bispectrum.
    Data and model bispectrum decomposed with the tripoSH formalism (see ref. [1]_).
    .. [1] Sugiyama N. et al., 2019. **MNRAS** 484(1), 364-384.
    [arXiv: `1803.02132 <https://arxiv.org/abs/1803.02132>`_]

    Parameters
    ----------
    estimator : list
        List of estimators to use (`Pk` and/or `Bk`).
    multipoles : list
        List of multipoles to use (e.g., `0` for power spectrum, `000` for bispectrum).
    k_edges : dict
        Dictionary containing tuples for each multipole of each estimator
        (e.g., {`Pk`:{`0`:(0.01,0.2)}}).
        Specifies the minimal and maximal wavenumber to include in the analysis for each multipole.
    cov_mock_nb : int or str
        If int, the number of mocks used to compute the covariance matrix.
        If `analytic`, the covariance matrix is calculated analytically (not implemented yet!).

    Attributes
    ----------
    k_vec : np.array
        Concatenation of wavenumbers included in the analysis for all multipoles.
    data_vec : np.array
        Full data vector used in the fit, concatenation of all estimator multipoles.
    cov_tot : np.array
        Covariance matrix.
    """

    def __init__(self, estimator, multipoles, k_edges, cov_mock_nb):
        self.estimator = estimator
        self.cov_mock_nb = cov_mock_nb
        self.multipoles = {est: [] for est in self.estimator}
        for ell in multipoles:
            if len(ell) == 1:
                self.multipoles['Pk'].append(ell)
            if len(ell) == 3:
                self.multipoles['Bk'].append(ell)
        self.k_edges = k_edges


    def data_prep(self, name_file):
        """Preparation of the data used in the analysis.
        ----------
        name_file : dict
            path of the data files stored in a dictionary with the estimator as 
            first entry (`Pk` or `Bk`) and the multipole as second.
            files must be txt files in format provided by Triumvirate
        """

        self.data = {}
        self.full_data = {}

        for d in self.estimator:
            self.data[d] = {}
            self.full_data[d] = {}

            for ell in self.multipoles[d]:
                self.full_data[d][ell] = np.loadtxt(name_file[d][ell])
                mask = (self.k_edges[d][ell][0] <= self.full_data[d][ell][:, 0]) & (self.k_edges[d][ell][1] >= self.full_data[d][ell][:, 0])
                self.data[d][ell] = self.full_data[d][ell][mask]

        self.k_dict = {d: {ell: self.data[d][ell][:, 0] for ell in self.multipoles[d]} for d in self.estimator}
        self.data_dict = {d: {ell: self.data[d][ell][:, 1] for ell in self.multipoles[d]} for d in self.estimator}
        
        self.k_vec = np.concatenate([self.k_dict[d][ell] for d in self.estimator for ell in self.multipoles[d]])
        self.data_vec = np.concatenate([self.data_dict[d][ell] for d in self.estimator for ell in self.multipoles[d]])
    

    def cov_prep(self,directory=None,cov_name=None,to_save=False, rescale=False):
        """Construction of the full covariance matrix.
        ----------
        directory : dict, optional
            if not `None` (default is `None`), dict of path of the directory that contains the mocks measurement.
            for example: {'Pk':{'0':/home/pk_mocks_0/},'Bk':{'000':/home/bk_mocks_000/}}
        cov_name : str, optional
            if not `None` (default is `None`), path of the pre-computed covariance matrix.
        to_save : str, optional
            if not `False` (default is `False`), path to save the computed covariance matrix.
        """
        
        if cov_name: self.cov_tot = self.load(cov_name)
        elif self.cov_mock_nb=='analytic': self.cov_analytic(cosmo_fid,to_save)
        else: self.cov_mocks(directory,to_save)

        if 'Pk' not in self.estimator and 'Pk' in self.cov_tot['k']:
            len_pk = np.sum([len(self.cov_tot['k']['Pk'][i]) for i in self.cov_tot['k']['Pk']])
            self.cov_tot['cov'] = self.cov_tot['cov'][len_pk:,len_pk:]
            del self.cov_tot['k']['Pk']
        if 'Bk' not in self.estimator and 'Bk' in self.cov_tot['k']:
            len_bk = np.sum([len(self.cov_tot['k']['Bk'][i]) for i in self.cov_tot['k']['Bk']])
            self.cov_tot['cov'] = self.cov_tot['cov'][:-len_bk,:-len_bk]
            del self.cov_tot['k']['Bk']
        mask_mul = np.ones(len(self.cov_tot['cov']),bool)
        tmp = 0
        for est in self.estimator:
            for ell in self.cov_tot['k'][est]:
                if ell not in self.multipoles[est]:
                    mask_mul[np.sum(self.cov_tot['length_multi'][:tmp]):np.sum(self.cov_tot['length_multi'][:tmp+1])] = 0
                tmp += 1
        self.mask_mul = mask_mul
        self.cov = self.cov_tot['cov'][mask_mul,:][:,mask_mul]
            
        mask = np.concatenate(
                                [(self.k_edges[s][ell][0]<=self.cov_tot['k'][s][ell]) & (self.k_edges[s][ell][1]>=self.cov_tot['k'][s][ell]) 
                                    for s in self.estimator for ell in self.multipoles[s]])
        self.mask=mask
        self.cov = self.cov[mask,:][:,mask]
        if rescale:
            self.cov *= rescale

    def cov_mocks(self,directory,to_save=False):
        """Computation of the covariance matrix from mocks.
        ----------
        directory : dict, optional
            if not `None` (default is `None`), dict of path of the directory that contains the mocks measurement.
            for example: {'Pk':{'0':/home/pk_mocks_0/},'Bk':{'000':/home/bk_mocks_000/}}
        to_save : str, optional
            if not `False` (default is `False`), path to save the computed covariance matrix.
        """

        self.cov_tot = {'k':{}}
        self.cov_tot['nmocks'] = self.cov_mock_nb
        self.cov_tot['estimator'] = self.estimator
        self.cov_tot['multipoles'] = self.multipoles
        self.cov_tot['lenght_multi'] = []
        for s in self.estimator:
            self.cov_tot['k'][s] = {}
            if s=='Pk': line = 1
            elif s=='Bk': line = 2
            for ell in self.multipoles[s]:
                mock_res = []
                tmp_list_dir = os.listdir(directory[s][ell])
                for m in tmp_list_dir:
                    tmp = np.loadtxt(os.path.join(directory[s][ell],m))
                    mock_res.append(tmp[:,line])
                    mock_res[-1] = np.reshape(mock_res[-1], (len(mock_res[-1]),1))
                mock_data = np.concatenate([mock_res[i] for i in range(len(tmp_list_dir))],axis=1)
                self.cov_tot['k'][s][ell] = tmp[:,0]
                self.mm = mock_data
                if 'full_res' in locals(): full_res = np.concatenate((full_res,mock_data))
                else: full_res = mock_data
                self.cov_tot['lenght_multi'].append(len(self.cov_tot['k'][s][ell]))
        self.cov_tot['cov'] = np.cov(full_res)

        if to_save:
            os.makedirs(os.path.dirname(to_save), exist_ok=True)
            np.save(to_save,self.cov_tot)

    def cov_analytic(self,cosmo_fid,to_save=False):
        """Computation of the analytical covariance matrix.
        ----------
        cosmo_fid : dict
            cosmological parameters.
        to_save : str, optional
            if not `False` (default is `False`), path to save the computed covariance matrix.
        """

        pass

    def inv_cov(self):
        """Computation of the inverse covariance matrix including the Hartlap factor.
        """

        if self.cov_mock_nb == 'analytic':
            hartlap = 1
        else:
            hartlap = (self.cov_mock_nb - len(self.data_vec) - 2) / (self.cov_mock_nb - 1)

        inv_cov = hartlap * np.linalg.inv(self.cov)
        return inv_cov


    def fit(self, directory,  name, params, jup=False, minuit=False, minos=False, poco=False):

        self.params = params
        self.params['sorted'] = ['omega_cdm','omega_b','h','ln10^{10}A_s','n_s','f','b1','b2','bG2','bGamma3',
                                'c0', 'c2pp', 'c4pp', 'c1', 'c2','ch','Pshot','a0','Bshot','fnlloc','fnlequi','fnlortho']
        self.params['params_zeros'] = []
        self.params['params_fix'] = []
        rm = []
        fix = []
        for p in self.params['sorted']: 
            if p not in self.params['prior'] or (self.params['prior'][p]['type'] == 'Fix' and self.params['prior'][p]['lim']==0): 
                rm.append(p)
            elif self.params['prior'][p]['type'] == 'Fix' and self.params['prior'][p]['lim']!=0:
                fix.append(p)
        for p in rm:
            self.params['sorted'].remove(p)
            if p=='f': continue
            self.params['params_zeros'].append(p)
        for p in fix:
            self.params['sorted'].remove(p)
            self.params['params_fix'].append(p)
        self.params['estimator'] = self.estimator
        self.params['multipoles'] = self.multipoles
        self.params['k_edges'] = self.k_edges
        self.params['k_fit'] = self.k_dict
        if 'ap_approx' not in self.params:
            self.params['ap_approx'] = False

        self.set_model()

        ndim = len(self.params['sorted'])
        nwalkers = np.max([2*ndim,4])
        nsteps = 10000
        nchains = 2

        self.params['values'] = {}
        start = np.zeros((nwalkers,ndim))
        for i,p in enumerate(self.params['sorted']):
            if self.params['prior'][p]['type']=='Uni':
                self.params['values'][p] = np.mean(self.params['prior'][p]['lim'])
                start[:,i] = np.random.uniform(self.params['prior'][p]['lim'][0],self.params['prior'][p]['lim'][1],nwalkers)
            elif self.params['prior'][p]['type']=='Gauss':
                self.params['values'][p] = self.params['prior'][p]['lim'][0]
                start[:,i] = np.random.normal(self.params['prior'][p]['lim'][0],self.params['prior'][p]['lim'][1],nwalkers)
        for i,p in enumerate(self.params['params_fix']):
            self.params['values'][p] = self.params['prior'][p]['lim']

        if minuit:
            global data
            global icov
            theta=[]
            for i,p in enumerate(self.params['sorted']):
                theta.append(start[0,i])
            print('theta', theta)
            print('sorted', self.params['sorted'])
                
        data = self.data_vec
        icov = self.inv_cov()
        global utils
        utils = self.params
        
        if minuit:
            self.m = Minuit(logpost_minuit, theta, name=self.params['sorted'])
            eps = 1e-10
            for p in self.params['prior']:
                if self.params['prior'][p]['type'] == 'Uni':
                    self.m.limits[p] = (self.params['prior'][p]['lim'][0]+eps, self.params['prior'][p]['lim'][1]-eps)

            self.m.migrad(ncall=100000)
            if minos: 
                for p in minos:
                    self.m.minos(p)
            if name:
                np.save(f'{name}_iminuit.npy', self.m)
                
        elif poco:
            bounds = np.empty((ndim, 2))
            for i,p in enumerate(self.params['sorted']):
                if self.params['prior'][p]['type'] == 'Uni':
                    bounds[i] = self.params['prior'][p]['lim']
                else:
                    bounds[i] = [None,None]
                    
            n_particles = 1000

            prior_samples = np.empty((n_particles, ndim))
            for i,p in enumerate(self.params['sorted']):
                if self.params['prior'][p]['type'] == 'Uni':
                    prior_samples[:,i] = np.random.uniform(low=bounds[i,0], high=bounds[i,1], size=(n_particles))
                elif self.params['prior'][p]['type'] == 'Gauss':
                    prior_samples[:,i] = np.random.normal(loc=self.params['prior'][p]['lim'][0], scale=self.params['prior'][p]['lim'][1], size=(n_particles))
            
            if jup:
                
                sampler = pc.Sampler(n_particles = n_particles,
                     n_dim = ndim,
                     log_likelihood = loglike_poco,
                     log_prior = logprior_poco,
                     bounds = bounds,
                     log_likelihood_args = [data,icov],
                     output_dir = '/home/rneveux/fit_test/',
                     output_label=name,
                     infer_vectorization=False,
                    )
                
                sampler.run(prior_samples,
                            save_every = 3,
                           )
            else:

                ncpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
                print(ncpus)
                
                with Pool(ncpus) as pool:

                    sampler = pc.Sampler(n_particles = n_particles,
                                     n_dim = ndim,
                                     log_likelihood = loglike_poco,
                                     log_prior = logprior_poco,
                                     bounds = bounds,
                                     log_likelihood_args = [data,icov],
                                     pool = pool,
                                     output_dir = directory,
                                     output_label=name,
                                     infer_vectorization=False,
                                    )

                    sampler.run(prior_samples = prior_samples,
                        #save_every = 3,
                        )
                    sampler.add_samples(5000)
            sampler.save_state(path=sampler.output_dir+sampler.output_label+'.state')
            self.sampler = sampler
            self.sampler_results = sampler.results
            self.save_fit()
                
        
        elif jup:
            self.cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
            self.cb1 = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
            self.cb2 = zeus.callbacks.MinIterCallback(nmin=500)
            self.sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, args=[data, icov])
            self.sampler.run_mcmc(start, nsteps, callbacks=[self.cb0, self.cb1, self.cb2])
            chain = self.sampler.get_chain(flat=True, discard=0.5)
            np.save(f'{name}_chain.npy', chain)
        else:
            with ChainManager(nchains) as cm:
                rank = cm.get_rank
                cb0 = zeus.callbacks.ParallelSplitRCallback(epsilon=0.01, chainmanager=cm)
                cb1 = zeus.callbacks.MinIterCallback(nmin=500)
                cb2 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
                cb3 = zeus.callbacks.SaveProgressCallback(f'{name}_chain_{rank}.h5', ncheck=100)
                sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, pool=cm.get_pool, args=[data, icov])
                sampler.run_mcmc(start, nsteps, callbacks=[cb0,cb1, cb2, cb3])
                chain = sampler.get_chain(flat=True, discard=0.5)
                if rank == 0:
                    print('R =', cb.estimates, flush=True)
                np.save(f'{name}_chain_{rank}.npy', chain)
                
    def save_fit(self):
        
        to_save = {}
        to_save['k_edges'] = self.k_edges
        to_save['k'] = self.k_dict
        to_save['sampler'] = self.sampler_results
        
        argmax = np.argmax(self.sampler_results['loglikelihood']+self.sampler_results['logprior'])
        to_save['max_logposterior'] = self.sampler_results['samples'][argmax]
        to_save['params_fit_sorted'] = self.params['sorted']
        to_save['prior'] = self.params['prior']
        to_save['mean'] = np.mean(self.sampler_results['samples'], axis=0)
        to_save['std'] = np.std(self.sampler_results['samples'], axis=0)
        max_cosmo = {}
        for i,p in enumerate(self.params['sorted']):
            kde = gaussian_kde(self.sampler_results['samples'][:, i])
            x = np.linspace(np.min(self.sampler_results['samples'][:, i]), np.max(self.sampler_results['samples'][:, i]), 1000)
            max_cosmo[p] = x[np.argmax(kde(x))]
        to_save['max_a_posteriori'] = max_cosmo

        np.save(self.sampler.output_dir+self.sampler.output_label+'.npy', to_save)

    def set_model(self):
        """Computation needed before MCMC iterations.
        """

        if 'Pk' in self.params['estimator'] and self.params['cosmo_inference']==False:
            PT = True
        else:
            PT = False
        if 'sigma8' in self.params['cosmo_fid']: self.params['class_fid'] = class_sigma8(self.params['cosmo_fid'], self.params['z_eff'], PT)
        else: self.params['class_fid'] = class_As(self.params['cosmo_fid'], self.params['z_eff'], PT) 
        self.params['fz'] = self.params['class_fid'].scale_independent_growth_factor_f(self.params['z_eff'])

        self.params['k_model'] = {}
        for est in self.params['estimator']:
            if est == 'Pk': ell = '0'
            elif est == 'Bk': ell = '000'
            if self.params['window']:
                self.params['k_model'][est] = np.arange(self.k_edges[est][ell][0]/2, self.k_edges[est][ell][1]*2+.0025, .0025)
            else:
                self.params['k_model'][est] = self.params['k_fit'][est][ell]

        if self.params['cosmo_inference']:
            self.params['AP_fid'] = {}
            self.params['AP_fid']['DA'] = self.params['class_fid'].angular_distance(self.params['z_eff'])
            self.params['AP_fid']['H'] = self.params['class_fid'].Hubble(self.params['z_eff'])
            self.cache_path = os.path.expanduser('~')+f'/bicker_cache/z{self.params["z_eff"]}/'


        if 'Pk' in self.estimator: self.set_power_spectrum_model()
        if 'Bk' in self.estimator: self.set_bispectrum_model()

    def set_bispectrum_model(self):

        self.cl = ComputePowerBiSpectrum(self.params['cosmo_fid'], self.params['z_eff'])
        self.cl.initial_power_spectrum(self.params['class_fid'])

        self.kernel_name_Bk = [
            'b1_b1_b1', 'b1_b1_b2','b1_b1_bG2','b1_b1_f','b1_b1_b1_f','b1_b1_f_f',
            'b1_b2_f','b1_bG2_f','b1_f_f',
            'b1_f_f_f','b2_f_f','bG2_f_f','f_f_f','f_f_f_f',
            'c1_b1_b1','c1_b1_b2','c1_b1_bG2','c1_b1_f','c1_b1_b1_f','c1_b1_f_f','c1_b2_f',
            'c1_bG2_f','c1_f_f','c1_f_f_f','c1_c1_b1','c1_c1_b2','c1_c1_bG2','c1_c1_f',
            'c1_c1_b1_f','c1_c1_f_f','c2_b1_b1','c2_b1_b2','c2_b1_bG2','c2_b1_f','c2_b1_b1_f',
            'c2_b1_f_f','c2_b2_f','c2_bG2_f','c2_f_f','c2_f_f_f','c2_c1_b1','c2_c1_b2',
            'c2_c1_bG2','c2_c1_f','c2_c1_b1_f','c2_c1_f_f','c2_c2_b1','c2_c2_b2','c2_c2_bG2',
            'c2_c2_f','c2_c2_b1_f','c2_c2_f_f',
            'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2', 
            'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2',
            'fnlloc_b1_b1_b1','fnlloc_b1_b1_f','fnlloc_b1_f_f','fnlloc_f_f_f',
            'fnlequi_b1_b1_b1','fnlequi_b1_b1_f','fnlequi_b1_f_f','fnlequi_f_f_f',
            'fnlortho_b1_b1_b1','fnlortho_b1_b1_f','fnlortho_b1_f_f','fnlortho_f_f_f',
            'fnlortho_LSS_b1_b1_b1','fnlortho_LSS_b1_b1_f','fnlortho_LSS_b1_f_f','fnlortho_LSS_f_f_f',]
        rm = []
        if self.params['ortho_LSS']: rm.extend(['fnlortho_b1_b1_b1','fnlortho_b1_b1_f','fnlortho_b1_f_f','fnlortho_f_f_f'])
        else: rm.extend(['fnlortho_LSS_b1_b1_b1','fnlortho_LSS_b1_b1_f','fnlortho_LSS_b1_f_f','fnlortho_LSS_f_f_f'])
        for i in self.kernel_name_Bk:
            for j in self.params['params_zeros']:
                    if j in i and i not in rm:
                        rm.append(i)
        for i in range(len(rm)): 
            self.kernel_name_Bk.remove(rm[i])
          
        self.params['kernel_name_Bk'] = self.kernel_name_Bk

        if self.params['cosmo_inference']:
            self.TT = {}
            emu = {}

            group_to_emul = []
            for gp in range(len(group)):
                for kernel in group[gp]:
                    if kernel in self.params['kernel_name_Bk']:
                        group_to_emul.append(gp)
                        break
            if 'Bshot' in self.params['sorted']:
                group_to_emul.append(8)

            z = self.params['z_eff']
            for ell in self.params['multipoles']['Bk']: 
                #for ell_dash in self.params['multipoles_to_use']['Bk']:
                #    for n in [0,1]:
                #        for m in [0,1]:
                #            self.TT[ell+'_'+ell_dash+'_'+str(n)+str(m)] = np.load(os.path.join('/home/rneveux/bispectrum/theory/approx_epsilon',
                #ell,ell_dash,str(n)+str(m)+'.npy'),allow_pickle=True).item()
                emu[ell] = {}
                for gp in group_to_emul:
                    if gp==8:
                        emu[ell][gp] = BICKER.component_emulator('shot', ell, self.params['k_model']['Bk'], self.cache_path)
                    else: emu[ell][gp] = BICKER.component_emulator(gp, ell, self.params['k_model']['Bk'], self.cache_path)
            self.params['emu'] = emu
            self.params['k_emul'] = np.loadtxt(os.path.join(self.cache_path, 'bispec/k_emul.txt'))
            #self.params['TT'] = self.TT

            self.Bk_kernels = {}
            for ell in self.params['multipoles_to_use']['Bk']:
                if ell not in self.multipoles['Bk']:
                    if ell not in self.Bk_kernels:
                        self.Bk_kernels[ell] = {}
                    for b in self.kernel_name_Bk:
                        if 'Pshot' in b or 'Bshot' in b: 
                            self.cl.kernel_computation_Bk(b, self.params['k_model']['Bk'], int(ell[0]), int(ell[1]), int(ell[2]), integrand='SN',)
                            self.Bk_kernels[ell][b] = self.cl.BK['K']
                        else:
                            self.cl.kernel_computation_Bk(b, self.params['k_model']['Bk'], int(ell[0]), int(ell[1]), int(ell[2]))
                            self.Bk_kernels[ell][b] = self.cl.BK['K']

        else:
            self.Bk_kernels = {}
            self.Bk_kernels_k = {}
            if self.params['with_kernels'] == 'pre-computed':
                self.params['k_emul'] = {}
                for ell in self.params['multipoles_to_use']['Bk']:
                    self.Bk_kernels[ell] = {}
                    for b in self.kernel_name_Bk:
                        if 'Pshot' in b or 'Bshot' in b: bispectrum_part = 'SN'
                        elif 'fnlequi' in b or 'fnlortho' in b: bispectrum_part = 'PNG'
                        else: bispectrum_part = 'tree'
                        fichier = os.path.join(self.params['kernels_directory'],ell,bispectrum_part,b+'.npy')
                        tmp = np.load(fichier,allow_pickle=True).item()
                        self.Bk_kernels[ell][b] = tmp['K']
                    self.Bk_kernels_k[ell] = tmp['kbin']
                self.params['k_emul'] = self.Bk_kernels_k['000']


            elif self.params['with_kernels'] == 'to_compute':
                for ell in self.params['multipoles_to_use']['Bk']:
                    self.Bk_kernels_k[ell] = self.params['k_model']['Bk']
                    self.Bk_kernels[ell] = {}
                    for b in self.kernel_name_Bk:
                        if 'Pshot' in b or 'Bshot' in b: 
                            self.cl.kernel_computation_Bk(b, self.params['k_model']['Bk'], 
                                                          int(ell[0]), int(ell[1]),int(ell[2]), integrand='SN')
                            self.Bk_kernels[ell][b] = self.cl.BK['K']
                        elif 'fnlloc' in b or 'fnlequi' in b or 'fnlortho' in b: 
                            self.cl.kernel_computation_Bk(b, self.params['k_model']['Bk'], 
                                                          int(ell[0]), int(ell[1]),int(ell[2]), integrand='PNG')
                            self.Bk_kernels[ell][b] = self.cl.BK['K']
                        else:
                            self.cl.kernel_computation_Bk(b, self.params['k_model']['Bk'], 
                                                          int(ell[0]), int(ell[1]), int(ell[2]))
                            self.Bk_kernels[ell][b] = self.cl.BK['K']
        self.params['Bk_kernels'] = self.Bk_kernels

    def set_power_spectrum_model(self):

        if self.params['cosmo_inference'] == True:
            z = self.params['z_eff']
            kemul = np.loadtxt(os.path.join(self.cache_path, 'powerspec/k_emul.txt'))
            self.params['emu_pk'] = {ell: BICKER.power(ell, kemul, self.cache_path) for ell in self.multipoles['Pk']}
            self.params['emu_growth'] = matry.Growth()
            if self.params['ap_approx']:
                for ell in self.params['multipoles']['Pk']: 
                    for ell_dash in [0,2,4]:
                        for n in [0,1,2,3]:
                            self.TT[ell+'_'+ell_dash+'_'+str(n)] = np.load(os.path.join('/home/rneveux/bispectrum/theory/approx_epsilon',
                                                 ell,ell_dash,str(n)+'.npy'),allow_pickle=True).item()
        else:
            self.params['class_fid'].initialize_output(self.params['k_model']['Pk']*self.params['cosmo_fid']['h'], self.params['z_eff'], 
                                                        len(self.params['k_model']['Pk']))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='config file to load')
    parser.add_argument('-config', type=str, help='config file', required=True)
    cmdline = parser.parse_args()
    
    with open(cmdline.config, 'r') as file:
        config = yaml.safe_load(file)

    estimator = config['estimator']
    multipoles = config['multipoles']
    cov_mock_nb = config['cov_mock_nb']
    k_edges = config['k_edges']
    
    data_dir = config['data_dir']
    
    name_file = {}
    for est in estimator:
        name_file[est] = {}
    for ell in multipoles:
        if len(ell)==1:
            name_file['Pk'][ell] = os.path.join(data_dir,config['name_file']['Pk'][ell])
        if len(ell)==3:
            name_file['Bk'][ell] = os.path.join(data_dir,config['name_file']['Bk'][ell])
    
    cov_file = config['cov_file']

    h_fid = config['h_fid']
    omega_b_fid = config['omega_b_fid']
    omega_cdm_fid = config['omega_cdm_fid']
    n_s_fid = config['n_s_fid']
    A_s_fid = config['A_s_fid']

    cosmo_fid = {
                'output': 'mPk',
                'h': h_fid,
                'omega_b': omega_b_fid,
                'omega_cdm': omega_cdm_fid,
                'n_s': n_s_fid,
                'A_s': A_s_fid,
                'tau_reio': 0.0544,
                'N_ncdm': 1.,
                'm_ncdm': 0.1, #'omega_ncdm': .0006442,
                'N_ur': 2.0328,
                'z_max_pk': 4.5, #3.,
                'P_k_max_h/Mpc': 50.,
                }
    
    ortho_LSS = False
    if 'fnlortho_LSS' in config['prior']:
        config['prior']['fnlortho'] = config['prior']['fnlortho_LSS']
        ortho_LSS = True
        del config['prior']['fnlortho_LSS']
    
    prior = config['prior']
    
    z_eff = config['z_eff']

    mean_density = config['mean_density']
    
    save_directory = config['save_directory']
    
    spec = config['spec']

    name_save = '_'.join(['_'.join(estimator),'_'.join(multipoles),spec])
    
    c_inference = False
    kernels_bk = False
    k_dir = None
    kernels_pk = False
    for p in ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10^{10}A_s']:
        if prior[p]['type']!= 'Fix':
            c_inference = True
        if config['direct_classpt']:
            c_inference = 'classpt'
            cosmo_classpt = cosmo_fid.copy()
            del cosmo_classpt['A_s']
        else:
            cosmo_classpt = None
    if 'Bk' in estimator:
        if c_inference:
            kernels_bk = True
        else:
            kernels_bk = 'pre-computed'
            k_dir = config['k_dir']
    if 'Pk' in estimator and c_inference:
        kernels_pk = True
    rescale = config['rescale']
            
            
    multipoles_to_use = config['multipoles_to_use']
    window = config['window']
    
    sampler = config['sampler']
    if sampler == 'minimizer':
        poco = False
        minuit = True
    if sampler == 'poco':
        poco = True
        minuit = False
        
    
    params = {'prior':prior, 'with_kernels':kernels_bk, 'kernels_directory':k_dir, 'cosmo_inference':c_inference, 'z_eff':z_eff,
              'cosmo_fid':cosmo_fid, 'cosmo_classpt':cosmo_classpt, 'mean_density':mean_density, 'window':window, 'kernels_pk':kernels_pk,
              'multipoles_to_use':multipoles_to_use, 'ortho_LSS': ortho_LSS}

    cl = PrepareFit(estimator, multipoles, k_edges, cov_mock_nb)
    cl.data_prep(name_file)
    cl.cov_prep(cov_name=cov_file, rescale=rescale)
    cl.fit(save_directory, name_save, params, poco=poco, minuit=minuit)


