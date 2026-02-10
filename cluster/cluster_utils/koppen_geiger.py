"""
Köppen-Geiger climate classification functionality taken by [1] and adapted to work with setup in this repository. The original code is available at [2].

References:
[1] Beck, H. E. et al. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023).

[2] https://github.com/hylken/Koppen-Geiger_maps/blob/3cc46c78d68bc9071fe17a3b75ec195b5ce4bdc0/tools.py#L91

Change log:
- 10-02-26: Initial implementation by [BSC]
    - Updated description to include arguments and return descriptions.
    - Changed np.NaN to np.nan to work with numpy > 2.0
"""

import numpy as np

def koppen_geiger(T,P,koppen_table):
    """This function classifies monthly temperature and precipitation
    climatologies according to the Koppen-Geiger climate classification. The
    inputs 'T' and 'P' represent the temperature and precipitation
    climatologies respectively, and should be provided as three-dimensional
    arrays with the third dimension representing time (12 months). The
    temperature data should be in units of degrees Celsius and the
    precipitation data in units of mm/month. The 'koppen_table' input is a
    table used to classify the climate. The function returns the climate class
    and major type of the area. 

    Args:
        T (numpy.ndarray): 3D array of monthly temperature climatology (degrees Celsius), shape (12, lat, lon).
        P (numpy.ndarray): 3D array of monthly precipitation climatology (mm/month), shape (12, lat, lon).
        koppen_table (pandas.DataFrame): Table used for classifying the climate.

    Returns:
        dict: A dictionary containing:
            - 'Class': 2D array of the final Köppen-Geiger subclass classification.
            - 'Major': 2D array of the final Köppen-Geiger major class classification
    """

    # Make boolean indexing array to select summer months
    # Summer is defined here as the warmest 6-month period between Oct-Mar
    # (ONDJFM) and Apr-Sep (AMJJAS). In the Northern Hemisphere, summer will 
    # usually be from Apr-Sep, while in the Southern Hemisphere, summer will
    # usually be from Oct-Mar.
    T_ONDJFM = np.mean(T[np.array([0,1,2,9,10,11]),:,:],axis=0)
    T_AMJJAS = np.mean(T[np.array([3,4,5,6,7,8]),:,:],axis=0)
    tmp = T_AMJJAS>T_ONDJFM
    sum_index = np.zeros(T.shape,dtype=bool)
    sum_index[np.array([3,4,5,6,7,8]),:,:] = np.tile(tmp,(6,1,1))
    sum_index[np.array([0,1,2,9,10,11]),:,:] = np.tile(~tmp,(6,1,1))
    del tmp
    
    # Total P in summer and winter and P in driest month
    Pw = np.sum(P*(~sum_index).astype(np.single),axis=0)
    Ps = np.sum(P*sum_index.astype(np.single),axis=0)
    Pdry = np.min(P,axis=0)

    # P in wettest and driest months in summer and winter
    tmp = sum_index.astype(np.single)
    tmp[tmp==0] = np.nan
    Psdry = np.nanmin(P*tmp,axis=0)
    Pswet = np.nanmax(P*tmp,axis=0)
    tmp = (~sum_index).astype(np.single) 
    tmp[tmp==0] = np.nan
    Pwdry = np.nanmin(P*tmp,axis=0)
    Pwwet = np.nanmax(P*tmp,axis=0)

    # Mean annual temperature and preciptiation (MAT and MAP)
    # Number of months with temperature >10 degrees C
    # Temperature of hottest and coldest months
    MAT = np.mean(T,axis=0)
    MAP = np.sum(P,axis=0)
    Tmon10 = np.sum(T>10,axis=0)
    Thot = np.max(T,axis=0)
    Tcold = np.min(T,axis=0)

    # Incorrect in Beck et al. (2018, 2023)
    # Threshold P
    #Pthresh = 2*MAT+14
    #Pthresh[Pw*2.333>Ps] = 2*MAT[Pw*2.333>Ps]
    #Pthresh[Ps*2.333>Pw] = 2*MAT[Ps*2.333>Pw]+28
    
    # Corrected for V3 of Beck et al. (2023) maps
    # Threshold P
    Pthresh = 2*MAT+14
    Pthresh[Pw>Ps*2.333] = 2*MAT[Pw>Ps*2.333]
    Pthresh[Ps>Pw*2.333] = 2*MAT[Ps>Pw*2.333]+28
    
    # Classification of B classes
    B = MAP<10*Pthresh
    BW = (B) & (MAP<5*Pthresh)
    BWh = (BW) & (MAT>=18)
    BWk = (BW) & (MAT<18)
    BS = (B) & (MAP>=5*Pthresh)
    BSh = (BS) & (MAT>=18)
    BSk = (BS) & (MAT<18)

    # Classification of A classes
    A = (Tcold>=18) & (~B)
    Af = (A) & (Pdry>=60)
    Am = (A) & (~Af) & (Pdry>=100-MAP/25)
    Aw = (A) & (~Af) & (Pdry<100-MAP/25)

    # Classification of C classes
    C = (Thot > 10) & (Tcold > 0) & (Tcold<18) & (~B)
    Cs = (C) & (Psdry<40) & (Psdry<Pwwet/3)
    Cw = (C) & (Pwdry<Pswet/10)
    overlap = (Cs) & (Cw)
    Cs[(overlap) & (Ps>Pw)] = 0
    Cw[(overlap) & (Ps<=Pw)] = 0
    Csa = (Cs) & (Thot>=22)
    Csb = (Cs) & (~Csa) & (Tmon10>=4)
    Csc = (Cs) & (~Csa) & (~Csb) & (Tmon10>=1) & (Tmon10<4)
    Cwa = (Cw) & (Thot>=22)
    Cwb = (Cw) & (~Cwa) & (Tmon10>=4)
    Cwc = (Cw) & (~Cwa) & (~Cwb) & (Tmon10>=1) & (Tmon10<4)
    Cf = (C) & (~Cs) & (~Cw)
    Cfa = (Cf) & (Thot>=22)
    Cfb = (Cf) & (~Cfa) & (Tmon10>=4)
    Cfc = (Cf) & (~Cfa) & (~Cfb) & (Tmon10>=1) & (Tmon10<4)

    # Classification of D classes
    D = (Thot>10) & (Tcold<=0) & (~B)
    Ds = (D) & (Psdry<40) & (Psdry<Pwwet/3)
    Dw = (D) & (Pwdry<Pswet/10)
    overlap = (Ds) & (Dw)
    Ds[(overlap) & (Ps>Pw)] = 0
    Dw[(overlap) & (Ps<=Pw)] = 0
    Dsa = (Ds) & (Thot>=22)
    Dsb = (Ds) & (~Dsa) & (Tmon10>=4)
    Dsd = (Ds) & (~Dsa) & (~Dsb) & (Tcold<-38)
    Dsc = (Ds) & (~Dsa) & (~Dsb) & (~Dsd)

    Dwa = (Dw) & (Thot>=22)
    Dwb = (Dw) & (~Dwa) & (Tmon10>=4)
    Dwd = (Dw) & (~Dwa) & (~Dwb) & (Tcold<-38)
    Dwc = (Dw) & (~Dwa) & (~Dwb) & (~Dwd)
    Df = (D) & (~Ds) & (~Dw)
    Dfa = (Df) & (Thot>=22)
    Dfb = (Df) & (~Dfa) & (Tmon10>=4)
    Dfd = (Df) & (~Dfa) & (~Dfb) & (Tcold<-38)
    Dfc = (Df) & (~Dfa) & (~Dfb) & (~Dfd)

    # Classification of E classes
    E = (Thot <= 10) & (~B)
    ET = (E) & (Thot>0)
    EF = (E) & (Thot<=0)
    
    # Make maps if final KG subclass and final KG major class
    # Using eval() is bad practice, but the code above is more concise
    # without dict
    Class = np.zeros((T.shape[1],T.shape[2]),dtype=np.single)*np.nan
    Major = np.zeros((T.shape[1],T.shape[2]),dtype=np.single)*np.nan
    for ii in np.arange(koppen_table.shape[0]):
        mask = eval(koppen_table['Symbol'][ii])
        Class[mask] = koppen_table['Class'][ii]
        Major[mask] = koppen_table['Major'][ii]

    # Make oceans NaN
    mask = np.isnan(P[0,:,:]+T[0,:,:])
    Class[mask] = np.nan
    Major[mask] = np.nan

    return {'Class': Class, 'Major': Major}