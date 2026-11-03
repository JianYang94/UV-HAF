"""Reconstruct UKESM1-0-LL erythemal UV for the three SSP pathways."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
try:
    from netCDF4 import Dataset, num2date
except ModuleNotFoundError:  # Cached UV fields do not require NetCDF access.
    Dataset = num2date = None

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT = Path(os.environ.get("UV_HAF_UKESM_ROOT", PROJECT_ROOT / "data" / "raw" / "UKESM1-0-LL_r1i1p1f2"))
OUT = PROJECT_ROOT / "outputs"
sys.path.insert(0, str(HERE))
import spectral_uv as uv

SCENARIOS = ("ssp126", "ssp245", "ssp585")
PERIODS = {"reference": (2015, 2025), "mid": (2041, 2060), "late": (2081, 2100)}
MONTH_DAYS = np.array([31,28,31,30,31,30,31,31,30,31,30,31], float)
G, M_AIR, NA, DU = 9.80665, 0.0289647, 6.02214076e23, 2.687e20

def files(scenario, var):
    return sorted((ROOT/scenario/var).glob("*.nc"))

def climatology(scenario, var, period):
    if Dataset is None:
        raise RuntimeError("netCDF4 is required when a UV cache must be rebuilt")
    sums=[None]*12; counts=np.zeros(12,int); lat=lon=None
    y0,y1=period
    for p in files(scenario,var):
        with Dataset(p) as d:
            tv=d.variables['time']; dates=num2date(tv[:],tv.units,getattr(tv,'calendar','standard'))
            take=np.array([y0 <= x.year <= y1 for x in dates])
            if not take.any(): continue
            for idx,x in enumerate(dates):
                if not take[idx]: continue
                v=np.ma.filled(d.variables[var][idx],np.nan).astype('float64')
                j=x.month-1; sums[j]=np.nan_to_num(v) if sums[j] is None else sums[j]+np.nan_to_num(v); counts[j]+=1
            lat=np.asarray(d.variables['lat'][:]); lon=np.asarray(d.variables['lon'][:])
    return np.stack([sums[j]/counts[j] for j in range(12)]),lat,lon

def ozone_du(scenario, period, ps12):
    if Dataset is None:
        raise RuntimeError("netCDF4 is required when a UV cache must be rebuilt")
    y0,y1=period; sums=[None]*12; counts=np.zeros(12,int); meta=None
    for p in files(scenario,'o3'):
        with Dataset(p) as d:
            tv=d.variables['time']; dates=num2date(tv[:],tv.units,getattr(tv,'calendar','standard'))
            take=np.array([y0 <= x.year <= y1 for x in dates])
            if not take.any(): continue
            for idx,x in enumerate(dates):
                if not take[idx]: continue
                v=np.ma.filled(d.variables['o3'][idx],np.nan).astype('float64')
                j=x.month-1; sums[j]=np.nan_to_num(v) if sums[j] is None else sums[j]+np.nan_to_num(v); counts[j]+=1
            if meta is None:
                zb=np.asarray(d.variables['lev_bnds'][:],float)
                bb=np.asarray(d.variables['b_bnds'][:],float)
                orog=np.asarray(d.variables['orog'][:],float)
                meta=(zb,bb,orog)
    vmr=np.stack([sums[j]/counts[j] for j in range(12)])
    zb,bb,orog=meta
    # Hybrid-height layer interfaces z=a+b*orography. Hydrostatic pressure uses
    # an isothermal 7-km scale height; adequate for column-ozone scenario contrasts.
    z0=zb[:,0,None,None]+bb[:,0,None,None]*orog
    z1=zb[:,1,None,None]+bb[:,1,None,None]*orog
    p0=ps12[:,None]*np.exp(-np.maximum(z0[None]-orog[None,None],0)/7000.)
    p1=ps12[:,None]*np.exp(-np.maximum(z1[None]-orog[None,None],0)/7000.)
    dp=np.abs(p0-p1)
    molecules=np.nansum(vmr*dp/(G*M_AIR)*NA,axis=1)
    return molecules/DU

def daily_dose(scenario, period, cache=True):
    stem=f"uv_{scenario}_{period[0]}_{period[1]}"
    cp=OUT/f"{stem}.npz"
    if cache and cp.exists():
        z=np.load(cp); return z['annual'],z['monthly'],z['ozone'],z['lat'],z['lon']
    rsds,lat,lon=climatology(scenario,'rsds',period)
    rsdscs,_,_=climatology(scenario,'rsdscs',period)
    rsuscs,_,_=climatology(scenario,'rsuscs',period)
    huss,_,_=climatology(scenario,'huss',period)
    ps,_,_=climatology(scenario,'ps',period)
    aod,_,_=climatology(scenario,'od550aer',period)
    ozone=ozone_du(scenario,period,ps)
    with np.errstate(divide='ignore',invalid='ignore'):
        cmf=np.clip(rsds/rsdscs,0,1.2); alb=np.clip(rsuscs/rsdscs,0,1)
    water=np.clip(huss*ps/G/10.,.02,10.); aod=np.clip(aod,.001,2.)
    ny,nx=len(lat),len(lon); phi=np.deg2rad(np.repeat(lat,nx)); monthly=[]
    for mi,day in enumerate(uv.DAYS):
        dec=np.deg2rad(23.44)*np.sin(2*np.pi*(day-81)/365.); total=np.zeros(ny*nx)
        for hour in np.arange(.5,24,1):
            ha=np.deg2rad(15*(hour-12)); cz=np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.cos(ha); ok=cz>0
            if not ok.any(): continue
            f={'sza':np.rad2deg(np.arccos(np.clip(cz[ok],-1,1))), 'ozone':np.clip(ozone[mi].ravel()[ok],100,600),
               'aod':aod[mi].ravel()[ok], 'ssa':np.full(ok.sum(),.95), 'alpha':np.full(ok.sum(),1.3),
               'albedo':alb[mi].ravel()[ok], 'pressure':ps[mi].ravel()[ok]/100., 'cot':np.zeros(ok.sum()),
               'reff':np.full(ok.sum(),10.), 'cloud_fraction':np.zeros(ok.sum()), 'water_cm':water[mi].ravel()[ok],
               'day':np.full(ok.sum(),day)}
            total[ok]+=uv.predict_erythema(uv.features(f))*cmf[mi].ravel()[ok]*3600
        monthly.append(total.reshape(ny,nx))
        print(f"{scenario} {period} month {mi+1}/12",flush=True)
    monthly=np.asarray(monthly); annual=np.average(monthly,axis=0,weights=MONTH_DAYS)
    np.savez_compressed(cp,annual=annual.astype('float32'),monthly=monthly.astype('float32'),ozone=ozone.astype('float32'),lat=lat,lon=lon)
    return annual,monthly,ozone,lat,lon

def regrid(src,slat,slon,dlat,dlon):
    return uv.interp_grid(src[None],slat,slon,dlat,dlon)[0]
