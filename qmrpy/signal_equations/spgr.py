"""
Signal models for fitting magnetization parameters from MR Images acquired with a Gradient Echo (GRE) Ultrashort TE (UTE) pulse sequence
"""
__author__ = "Dharshan Chandramohan"

import numpy as np

def T2strw_mag_simplified(K, TE, T2str, N):
    """Signal Model of T2str-weighted UTE GRE Magnitude Image
    
    S = K * [ exp(-TE/T2*) ] + N
    
    parameters:
      K :: constant (proportional to proton density)
      TE :: sequence echo time
      T2str :: relaxation due to spin-spin effects and dephasing
      N :: constant offset "noise" term
    
    @return expected (magnitude) signal
    """
    S = K * np.exp((-1.0 * TE)/T2str) + N
    return S

def T2strw_power(P_0, TE, T2str):
    """Signal Model for the "power" (square of the magnitude) of a complex signal from a T2* weighted GRE image
    
    P = M_c^2 = Re{S}^2 + Im{S}^2
    P = P_0 * [ exp(-2TE/T2*) ]
    
    M_c stands for the "noise-corrected" signal magnitude
    M_c^2 = M^2 - 2*sigma^2
    where sigma is the standard deviation in a noise region (this information is relevant to caluclating the residuals)
    
    parameters:
      P_0 :: constant (proportional to proton density)
      TE :: sequence echo time
      T2str :: relaxationn due to spin-spin effects and dephasing
    
    @return P :: predicted power of the signal
    """

    P = P_0 * np.exp((-2.0 * TE)/T2str)
    return P

def T2strw_cplx(K, TE, T2str, df, phi):
    """Signal Model of T2str-weighted UTE GRE Magnitude Image
    
    S = K * [ exp(-TE/T2*) ] + N
    
    parameters:
      K :: constant (proportional to proton density)
      TE :: sequence echo time
      T2str :: relaxation due to spin-spin effects and dephasing
      df :: frequency shift
      phi :: phase
    
    @return expected (magnitude) signal
    """
    S = K * np.exp((-1.0 * TE)/T2str - 1j*2*np.pi*df*TE + 1j*phi)
    return S

def spgr_mag(PD, T1, T2str, TR, TE, alph, k=1.0):
    """Spoiled Gradient Recall at Steady State (SPGR) signal equation"""
    S = k * PD * np.exp(-TE/T2str) * ((np.sin(alph) * (1 - np.exp(-TR/T1)))/(1 - (np.cos(alph) * np.exp(-TR/T1))))
    return S

def spgr_complex(M0, T1, T2str, TR, TE, alph, c_shift, del_B0, phi):
    """Spoiled Gradient Recall at Steady State (SPGR) signal equation"""
    df = c_shift + (42.577478518e6 * del_B0) # del_B0 in T, c_shift in Hz
    S = M0 * np.exp(-TE/T2str(-1.0 * TE)/T2str - 1j*2*np.pi*df*TE + 1j*phi) * ((np.sin(alph) * (1 - np.exp(-TR/T1)))/(1 - (np.cos(alph) * np.exp(-TR/T1))))
    return S

def T1w_mag(K, T1, TR, alph):
    """Expected signal for T1w UTE GRE 'magnitude' images"""
    S = K * ((np.sin(alph) * (1 - np.exp((-1.0 * TR)/T1))) / (1 - (np.cos(alph) * np.exp((-1.0 * TR)/T1))))
    return S

def spgr_complex_flex_experimental(M0=None, T1=None, T2str=None, TR=None, TE=None, alph=None, c_shift=None, del_B0=None, phi=None):
    """Spoiled Gradient Recall at Steady State (SPGR) signal equation"""

    if not M0:
        M0 = 1.0

    if not (T1 and TR and alph):
        T1_part = 1.0
    else:
        T1_part = ((np.sin(alph) * (1 - np.exp(-TR/T1)))/(1 - (np.cos(alph) * np.exp(-TR/T1))))

    if not (T2str and TE):
        T2_part = 1.0
    else:
        np.exp(-TE/T2str(-1.0 * TE)/T2str)

    if not phi:
        rf_phase_part = 1.0
    else:
        rf_phase_part = np.exp(1j*phi)

    if not (c_shift and del_B0):
        freq_shift_part = 1.0
    else:
        df = 0.0
        if c_shift is not None:
            df = df + c_shift
        if del_B0 is not None:
            df = df + (42577478518e6 * del_B0)

        freq_shift_part = np.exp(-1j*2*np.pi*df*TE)

    S = M0 * T1_part * T2_part * rf_phase_part * freq_shift_part  
    return S
