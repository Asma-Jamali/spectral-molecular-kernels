"""
MBDF: Many-Body Distribution Functionals
=======================================

Source: https://github.com/dkhan42/MBDF/blob/main/MBDF.py

  local=True  -> (N, pad, 6) padded per-atom array (same convention as acsf.npy)
  local=False -> (N, D) fixed-size global vector per molecule (D depends on
                  the per-element bag sizes across the dataset)
"""

import numpy as np
import numba
from joblib import Parallel, delayed

half_rootpi = (np.pi ** 0.5) / 2
a2b = 1.88973


@numba.jit(nopython=True, cache=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 +
                        t * ( 0.09678418 +
                        t * (-0.18628806 +
                        t * ( 0.27886807 +
                        t * (-1.13520398 +
                        t * ( 1.48851587 +
                        t * (-0.82215223 +
                        t * ( 0.17087277))))))))))
    return ans


@numba.jit(nopython=True, cache=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 0:
        return 1
    elif degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2


@numba.jit(nopython=True, cache=True)
def generate_data(size,z,atom,charges,coods,cutoff_r=12):
    """returns 2 and 3-body internal coordinates"""
    twob=np.zeros((size,2))
    threeb=np.zeros((size,size,5))
    z1=z**0.8

    for j in range(size):
        rij=atom-coods[j]
        rij_norm=np.linalg.norm(rij)

        if rij_norm!=0 and rij_norm<cutoff_r:
            z2=charges[j]**0.8
            twob[j]=rij_norm,z1*charges[j]

            for k in range(size):
                if j!=k:
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]**0.8

                        rkj=coods[k]-coods[j]

                        rkj_norm=np.linalg.norm(rkj)

                        threeb[j][k][0] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        threeb[j][k][1] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        threeb[j][k][2] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))

                        atm = rij_norm*rik_norm*rkj_norm

                        charge = z1*z2*z3

                        threeb[j][k][3:] =  atm, charge

    return twob, threeb


@numba.jit(nopython=True, cache=True)
def angular_integrals(size,threeb,alength=158,a=2,grid1=None,grid2=None,angular_scaling=2.4):
    """evaluates the 3-body functionals using the trapezoidal rule"""
    arr=np.zeros((alength,2))
    theta=0

    for i in range(alength):
        f1,f2=0,0
        num1,num2=grid1[i],grid2[i]

        for j in range(size):

            for k in range(size):

                if threeb[j][k][-1]!=0:

                    angle1,angle2,angle3,atm,charge=threeb[j][k]

                    x=theta-np.arccos(angle1)

                    exponent,h1=np.exp(-a*x**2),hermite_polynomial(x,1,a)

                    f1+=(charge*exponent*num1)/(atm**4)

                    f2+=(charge*h1*exponent*(1+(num2*angle1*angle2*angle3)))/(atm**angular_scaling)

        arr[i]=f1,f2
        theta+=0.02

    trapz=[np.trapz(arr[:,i],dx=0.02) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True, cache=True)
def radial_integrals(size,rlength,twob,step_r,a=1,normalized=False):
    """evaluates the 2-body functionals using the trapezoidal rule"""
    arr=np.zeros((rlength,4))
    r=0

    for i in range(rlength):
        f1,f2,f3,f4=0,0,0,0

        for j in range(size):

            if twob[j][-1]!=0:
                dist,charge=twob[j]
                x=r-dist

                if normalized==True:
                    norm=(erfunc(dist)+1)*half_rootpi
                    exponent=np.exp(-a*(x)**2)/norm

                else:
                    exponent=np.exp(-a*(x)**2)

                h1,h2=hermite_polynomial(x,1,a),hermite_polynomial(x,2,a)

                f1+=charge*exponent*np.exp(-10.8*r)

                f2+=charge*exponent/(2.2508*(r+1)**3)

                f3+=charge*(h1*exponent)/(2.2508*(r+1)**6)

                f4+=charge*h2*exponent*np.exp(-1.5*r)

        r+=step_r
        arr[i]=f1,f2,f3,f4

    trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True, cache=True)
def mbdf_local(charges,coods,grid1,grid2,rlength,alength,pad=29,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """returns the local MBDF representation for a molecule"""
    size = len(charges)
    mat=np.zeros((pad,6))

    assert size > 1, "No implementation for monoatomics"

    if size>2:
        for i in range(size):

            twob,threeb = generate_data(size,charges[i],coods[i],charges,coods,cutoff_r)

            mat[i][:4] = radial_integrals(size,rlength,twob,step_r)

            mat[i][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

    elif size==2:
        z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]

        pref, dist = z1*z2, np.linalg.norm(rij)

        twob = np.array([[pref, dist], [pref, dist]])

        mat[0][:4] = radial_integrals(size,rlength,twob,step_r)

        mat[1][:4] = mat[0][:4]

    return mat


def mbdf_global(charges,coods,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """returns the flattened, bagged MBDF feature vector for a molecule"""
    elements = {k:[[],k] for k in keys}

    size = len(charges)

    for i in range(size):
        elements[charges[i]][0].append(coods[i])

    mat, ind = np.zeros((rep_size,6)), 0

    assert size > 1, "No implementation for monoatomics"

    if size>2:

        for key in keys:

            num = len(elements[key][0])

            if num!=0:
                bags = np.zeros((num,6))

                for j in range(num):
                    twob,threeb = generate_data(size,key,elements[key][0][j],charges,coods,cutoff_r)

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)

                    bags[j][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)

            ind += asize[key]

    elif size == 2:

        for key in keys:

            num = len(elements[key][0])

            if num!=0:
                bags = np.zeros((num,6))

                for j in range(num):
                    z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]

                    pref, dist = z1*z2, np.linalg.norm(rij)

                    twob = np.array([[pref, dist], [pref, dist]])

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)

            ind += asize[key]

    return mat


@numba.jit(nopython=True, cache=True)
def fourier_grid():
    angles = np.arange(0,np.pi,0.02)

    grid1 = np.cos(angles)
    grid2 = np.cos(2*angles)
    grid3 = np.cos(3*angles)

    return (3+(100*grid1)+(-200*grid2)+(-164*grid3),grid1)


@numba.jit(nopython=True, cache=True)
def normalize(A,normal='mean'):
    """normalizes the functionals based on the given method"""
    A_temp = np.zeros(A.shape)

    if normal=='mean':
        for i in range(A.shape[2]):

            avg = np.mean(A[:,:,i])

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg

            else:
                pass

    elif normal=='min-max':
        for i in range(A.shape[2]):

            diff = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))

            if diff!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff

            else:
                pass

    return A_temp


def generate_mbdf(nuclear_charges,coords,local=True,n_jobs=-1,pad=None,step_r=0.1,cutoff_r=8.0,step_a=0.02,angular_scaling=4,normalized='min-max',progress_bar=False):
    """
    Generates the MBDF representation arrays for a set of given molecules.

    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset
    :param coords: array of arrays of input coordinates of the atoms (same ordering as nuclear_charges)
    :param local: True -> (N, pad, 6) per-atom array; False -> (N, D) global vector per molecule
    :param n_jobs: number of cores to parallelise over (-1 = all available)
    :param pad: number of atoms in the largest molecule; auto-computed if None
    :param normalized: 'min-max', 'mean', or False

    :return: local=True -> (N, pad, 6) array; local=False -> (N, D) array
    """
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"

    lengths, charges = [], []

    for i in range(len(nuclear_charges)):

        q, r = nuclear_charges[i], coords[i]

        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    grid1,grid2 = fourier_grid()

    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    if local:
        if progress_bar==True:
            from tqdm import tqdm
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))
        else:
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        mbdf=np.array(mbdf)

        if normalized==False:
            return mbdf
        else:
            return normalize(mbdf,normal=normalized)

    else:
        keys = np.unique(np.concatenate(charges))

        asize = {key:max([(mol == key).sum() for mol in charges]) for key in keys}

        rep_size = sum(asize.values())

        if progress_bar==True:
            from tqdm import tqdm
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))
        else:
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        arr = np.array(arr)

        if normalized==False:
            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            return mbdf
        else:
            arr = normalize(arr,normal=normalized)
            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            return mbdf
