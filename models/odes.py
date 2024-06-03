import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from scipy.integrate import odeint
from matplotlib.colors import LogNorm, Normalize

def model_2D(X,t, r, R, L, n0, ym, A=0, alpha=1000, beta=1, gamma=1):
    y, n = X
    #print (t,X)
    return [-n*r*y/(y+ym) + A*(1-pow(y,beta)), R*(pow(np.abs(y),gamma)-pow(y,alpha)) - L*n/(n+n0)]    

def jacobian(y, n, r, R, L, n0, ym, A=0, alpha=1000, beta=1, gamma=1):
    return np.array([
                    [ -n*r*ym/(y+ym)**2 - A*beta*pow(y,beta-1) , -r*y/(y+ym) ],
                    [ R*( gamma*pow(y,gamma-1) - alpha*pow(y,alpha-1)) , -L*n0/(n+n0)**2 ]
                    ])

def eigenvalues(trivialeq, r, R, L, n0, ym, A=0, alpha=1000, beta=1, gamma=1):
    if trivialeq:
        eq = [1,0]
    else:
        eq = equil_value([0,0], r, R, L, n0, ym, A, alpha, beta, gamma)[0]
    J = jacobian(eq[0], eq[1], r, R, L, n0, ym, A, alpha, beta, gamma)
    #print(eq)
    #print(J)

    return np.linalg.eig(J)[0], eq

def model_2D_noisy(X0,tmax, r, R, L, n0, ym, A=0, noisemag=0.005, alpha=1000, beta=1,gamma=1, absnoise=True):

    t_int = 1
    nsteps = int(tmax/t_int)+1
    t = np.linspace(0,tmax,nsteps)

    X = np.zeros((nsteps,2))
    X[0] = X0
    yreduced = np.zeros((nsteps))
    # noisemag is the std deviation of noise added at each time interval
    if absnoise:
        width = 0.5
        noises = noisemag * np.random.lognormal(mean=0, sigma=width, size=nsteps) / np.exp(width**2/2)
    else:
        noises = np.random.normal(loc=0, scale=noisemag, size=nsteps)
    for i in range(1, nsteps):
        noises[i] = min(noises[i] * (1 - pow(X[i-1, 0], beta)), 
                        1-X[i-1, 0])
        Xstart = [X[i-1, 0] + noises[i],
                  X[i-1, 1]]
        sol=odeint(model_2D, Xstart, [t[i], 2*t[i]-t[i-1] if i==nsteps-1 else t[i+1]], args=(r,R,L,n0,ym,A,alpha,beta,gamma), atol=1e-5)
        #print(i,sol)
        X[i] = sol[-1]
        yreduced[i] = Xstart[0] - X[i,0]
        #print(noises[i], X[i-1,0], Xstart[0], X[i,0])
    
    #print('model_2D_noisy y = ',np.mean(X[-100:-1,0]))
    return X, t, yreduced, noises


def run_model_parameterchange_noisy(tchange, r, R, L, n0, ym, A, noisemag=0.005, alpha=1000, beta=1, gamma=1, absnoise=True):
    '''
    t, R, L, noisemag, A, should be arrays of the same length = number of changes of parameters
    '''

    X, t, yred, noises = np.array([[0,0]]), np.array([0]), np.array([0]), np.array([0])
    for ic in range(len(tchange)-1):
        Xseg, tseg, yredseg, noisesseg =  model_2D_noisy(X[-1], tchange[ic+1]-tchange[ic], r, R[ic], L[ic], n0, ym, A[ic], noisemag[ic], alpha, beta, gamma, absnoise)
        X = np.concatenate((X, Xseg[1:]))
        t = np.concatenate((t, t[-1]+tseg[1:]))
        yred = np.concatenate((yred, yredseg[1:]))
        noises = np.concatenate((noises, noisesseg[1:]))

    return X, t, yred, noises

def returntime(X0, r, R, L, n0, ym, A=0, alpha=1000, beta=1, gamma=1, tmax=1000):
    X = X0
    t_beg = 0
    t_int = 10
    t_step = 1
    sol = None
    t = None
    while (X[0] > ym or t_beg == 0) and t_beg<tmax:
        t = np.linspace(t_beg, t_beg+t_int, int(t_int/t_step)+1)
        sol=odeint(model_2D, X, t, args=(r,R,L,n0,ym,A,alpha,beta,gamma), atol=1e-3)
        X = sol[-1]
        t_beg += t_int

    # interpolate to find the value of t at which the solution crosses the value ym
    return np.interp(ym, sol[:,0][::-1], t[::-1])

def equil_value(X0, r, R, L, n0, ym, A=0, alpha=1000, beta=1, gamma=1, tmax=1000):
    tol_stability = 1e-5
    dX = 1
    t = 0
    t_int = 20
    X = X0
    while dX > X[0]*tol_stability and t<tmax:
        sol=odeint( model_2D , X, [t, t+t_int], args=(r,R,L,n0,ym,A,alpha,beta,gamma), atol=1e-5)
        dX = np.abs(X[0] - sol[-1,0])
        X = sol[-1]
        t += t_int
    return X, (t<tmax)

def returnrate(r, R, L, n0, ym, A=0, alpha=1000, beta=1,gamma=1,tmax=1000, yperturb=0.02):
    dt = 1 # the rate of restoration of y is per minute
    Xeq = equil_value([0,0], r, R, L, n0, ym, A, alpha, beta, gamma, tmax)[0]
    
    if Xeq[0]>0.9999:
        rate = 0
    else:
        Xstart = Xeq
        Xstart[0] += yperturb
        sol=odeint(model_2D, Xstart, [0, dt], args=(r,R,L,n0,ym,A,alpha,beta,gamma), atol=1e-6)
        Xend = sol[-1]
        #print(L,A, Xeq[0] , Xend[0])
        rate = (Xend[0] - Xstart[0]) / yperturb / dt

    return Xeq, rate

def variance(r, R, L, n0, ym, A=0, alpha=1000, beta=1,gamma=1, sigma=0.01):
    print(L,A)
    soleq = equil_value([0,0], r, R, L, n0, ym, A, alpha, beta, gamma, tmax=2000)[0]

    if soleq[0] > 0.95:
        return 0
    else:
        ntry = 15
        resvar = np.zeros(ntry)
        for tr in range(ntry):
            nnoise = 300
            noises = np.random.normal(loc=0, scale=sigma, size=nnoise)
            #noises[0:100] = 0

            X = np.zeros((nnoise,2))
            X[0] = soleq
            for i in range(1,nnoise):
                Xstart = [max(min(X[i-1, 0] + noises[i], 1), 0),
                        X[i-1, 1]]
                sol=odeint(model_2D, Xstart, [i,i+1], args=(r,R,L,n0,ym,A,alpha,beta,gamma), atol=1e-5)
                X[i] = sol[-1]

            #Xparts = np.array(np.split(X[-200:,0], 20))
            var = np.var(X[-200:,0]) #np.mean(np.var(Xparts, axis=1))
            #print(var)
            #plt.figure()
            #plt.plot(np.arange(nnoise), X[:,0])
            #plt.savefig('figs/variance/smallnoise_A'+str(A)+'_RonL'+str(R/L)+'.png')
            resvar[tr] = var

        return np.mean(resvar)
    

def plot2d(x,y,funz, name,  titlex='x', titley='y', titlez='z', logz=False, zmin=None, zmax=None, twoz=False):
    print(name)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros(X.shape)
    if twoz:
        Z2 = np.zeros(X.shape)
        Z3 = np.zeros(X.shape)
        Z4 = np.zeros(X.shape)
        Z5 = np.zeros(X.shape)

    for iX, iY in np.ndindex(X.shape):
        Ztmp = funz(X[iX,iY], Y[iX,iY])
        if twoz:
            Z[iX, iY] = Ztmp[0]
            Z2[iX, iY] = Ztmp[1]
            Z3[iX, iY] = Ztmp[2]
            Z4[iX, iY] = Ztmp[3]
            Z5[iX, iY] = Ztmp[4]
        else:
            Z[iX, iY] = Ztmp[0]
        #print(Z[iX, iY])

    for iz,z in enumerate([Z,Z2,Z3,Z4] if twoz else [Z]):
        fig, ax = plt.subplots(1, 1) 
        zm = np.partition(z.flatten(), 20)[20]
        zM = np.partition(z.flatten(), -20)[-20]
        zM = max(abs(zm), abs(zM))
        zm = -zM
        norm = LogNorm(vmin=(zm if zmin is None else zmin), 
                       vmax=(zM if zmax is None else zmax)) if logz else Normalize(vmin=(zm if zmin is None else zmin), 
                                                                                   vmax=(zM if zmax is None else zmax))
        c = ax.pcolormesh(X,Y,z, cmap='inferno', norm=norm)
        ax.axis([min(x), max(x),min(y),max(y)])
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_ylabel(titlez)
        ax.set_xlabel(titlex)
        ax.set_ylabel(titley)

        plt.contour(X, Y, Z5, [0.5])
        plt.savefig('figs/'+('returnrate_' if iz==0 else ('returnrateEq0_' if iz==1 else ('variance_' if iz==2 else 'varianceEq0_')))+name+'.png')




def plot_yvsn(X, t, yreduced, noises, title, RonL, noisemag, t_change, perturb=None, yeq=None, variance=None, vari_fit=None, returnr_fit=None,returnr0=None, returnr0_fit=None, vari0=None, vari0_fit=None, t_fit=None):
    RonL_expand = [0]
    noisemag_expand = [0]
    for ic in range(len(t_change)-1):
        RonL_expand = np.concatenate((RonL_expand, [RonL[ic]]*int(t_change[ic+1]-t_change[ic])))
        noisemag_expand = np.concatenate((noisemag_expand, [noisemag[ic]]*int(t_change[ic+1]-t_change[ic])))
    noiseav = np.zeros(len(yreduced))
    sw=20
    for i in range(0,sw):
        noiseav[i] = np.mean(noises[0:sw])
    for i in range(sw,len(noiseav)):
        noiseav[i] = np.mean(noises[i-sw:i+1])
    if perturb is not None:
        noiseav = perturb

    nplt = 7 if variance is None else (8 if vari0 is None else (9 if returnr0_fit is None else 10))
    fig, axs = plt.subplots(nplt, 1, sharex=True, dpi=300, figsize=(7, 7), clear=True, num=1)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(t, X[:,0], label='y')
    if yeq is not None:
        axs[0].plot(t, yeq, label='y_eq')
    axs[1].plot(t, X[:,1], label='n_users')
    axs[2].plot(t, noises, label='noise')
    axs[3].plot(t, yreduced, label='return')
    axs[4].plot(t, yreduced/noiseav, label='return rate')
    if returnr_fit is not None:
        axs[4].plot(t_fit, np.poly1d(returnr_fit)(t_fit), label='fit: slope '+"{:.5f}".format(returnr_fit[0]))
    axs[5].plot(t, RonL_expand, label='R/L')
    axs[6].plot(t, noisemag_expand, label='attack magnitude (A)')
    if variance is not None:
        axs[7].plot(t, variance, label='variance')
        if vari_fit is not None:
            axs[7].plot(t_fit, np.poly1d(vari_fit)(t_fit), label='fit: slope '+"{:.5f}".format(vari_fit[0]))
    if vari0 is not None:
        axs[8].plot(t, vari0, label='variance (eq=0)')
        if vari0_fit is not None:
            axs[8].plot(t_fit, np.poly1d(vari0_fit)(t_fit), label='fit: slope '+"{:.5f}".format(vari0_fit[0]))
    if returnr0 is not None:
        axs[9].plot(t, returnr0, label='return rate (eq=0)')
        if returnr0_fit is not None:
            axs[9].plot(t_fit, np.poly1d(returnr0_fit)(t_fit), label='fit: slope '+"{:.5f}".format(returnr0_fit[0]))


    axs[0].set_ylim([0, 1.1*max(X[:,0])])
    axs[1].set_ylim([0, 1.1*max(X[:,1])])
    if np.all(noises >= 0):
        axs[2].set_ylim([0, 1.1*max(noises)])
    axs[3].set_ylim([0, 1.1*max(yreduced)])
    axs[4].set_ylim([0, 1.1*max(yreduced[noiseav>0.01]/noiseav[noiseav>0.01])])
    axs[5].set_ylim([0, 1.1*max(RonL_expand)])
    axs[6].set_ylim([0, 1.1*max(noisemag_expand)])
    if variance is not None:
        axs[7].set_ylim([0, 1.1*max(variance)])
    if vari0 is not None:
        axs[8].set_ylim([0, 1.1*max(vari0)])
    if returnr0 is not None:
        axs[9].set_ylim([0, 1.1*max(returnr0)])
    for i in range(nplt):
        axs[i].legend()
    plt.xlabel('t [min]')
    plt.savefig('figs/yvst/'+title+'.png')

#sol_noisy=model_2D_noisy( X0, 1000,r,R,L,n0,ym, noisemag=0.005)
#plot_yvsn(*sol_noisy, 'yn_vst_gaussnoisemodel')


# parameter values
npix = 1000
# rate of pixel changes (in terms of fraction of the composition) per user
r = 0.5 / npix / 5. # half a pixel per 5 minutes
n0 = 10
ym = 0.01
R = 60 # number of users gained per minute when y=1 (and n=0)
L = 20 # number of users lost per minute when y=0

X0=[0.3, n0]
t = np.linspace(0,50,100)
sol=odeint(model_2D, X0, t, args=(r,R,L,n0,ym), atol=1e-3)

Rmin,Rmax = 10,300
RonLmin, RonLmax = 0.1, 10

A = 0.07
alpha=4
beta=10
gamma=1
noisemag=0.03

t_change = np.hstack(([0], np.linspace(300,600, 49)))
n_changes = len(t_change)
R_change = np.linspace(60,0, n_changes)
L_change = np.full(n_changes, L)
noisemag_change = np.full(n_changes, noisemag)
sol_noisy_paramchange = run_model_parameterchange_noisy(t_change, r, R_change, L_change, n0, ym, np.zeros(len(R_change)), noisemag_change, alpha=alpha, beta=beta, gamma=gamma)
#sol_noisy_paramchange2 = [sol_noisy_paramchange[i][-700:-1] for i in range(len(sol_noisy_paramchange))]
plot_yvsn(*sol_noisy_paramchange, 'yn_vst_gaussnoisemodel_gradualchange_R', R_change/L_change, noisemag_change, t_change)


R_change2 = np.full(n_changes, R)
#noisemag_change2 = np.concatenate((np.linspace(0.03,0.2,int(n_changes/2)), np.full(int(n_changes/2), 0.2)))
noisemag_change2 = np.linspace(0.02,0.12,n_changes)
sol_noisy_paramchange3 = run_model_parameterchange_noisy(t_change, r, R_change2, L_change, n0, ym, np.zeros(len(R_change)), noisemag_change2, alpha=alpha, beta=beta, gamma=gamma)
#sol_noisy_paramchange4 = [sol_noisy_paramchange3[i][0:400] for i in range(len(sol_noisy_paramchange))]
plot_yvsn(*sol_noisy_paramchange3, 'yn_vst_gaussnoisemodel_gradualchange_noisemag', R_change2/L_change, noisemag_change2, t_change)
#print(sol_noisy_paramchange3[3])


RonL = np.linspace(0.4,4, 150)
rate = np.zeros(RonL.shape)
yeq = np.zeros(RonL.shape)
""" for i in range(RonL.size):
    RateAndEquil = returnrate(r, R, R/RonL[i], n0, ym, A, alpha, tmax=1000, yperturb=0.02)
    rate[i] = RateAndEquil[1]
    yeq[i] = RateAndEquil[0][0]
    
plt.figure()
plt.plot(RonL, rate)
plt.plot(RonL, yeq)
plt.plot(RonL, yeq*yeq)
plt.ylim([0,1.1*max(rate)])
plt.savefig('figs/returnrate_vs_RonL.png')
 """

'''
plot2d(np.linspace(Rmin, Rmax, 50),
       np.linspace(RonLmin, RonLmax, 50),
       lambda X,Y: equil_value([0,0], r=r, R=X, L=X/Y, n0=n0, ym=ym, A=0, tmax=1000)[0][0],
       'equilvalue_vsRandRonL', 'R','R/L','y_eq',logz=False)

A=0.01
plot2d(np.linspace(Rmin, Rmax, 50),
       np.linspace(RonLmin, RonLmax, 50),
       lambda X,Y: equil_value([0,0], r=r, R=X, L=X/Y, n0=n0, ym=ym, A=A, tmax=1000)[0][0],
       'equilvalue_constAttack0p01_vsRandRonL', 'R','R/L','y_eq',logz=False) 

Amin,Amax=0,0.1
plot2d(np.linspace(Amin, Amax, 70),
       np.linspace(RonLmin,9, 70),
       lambda X,Y: equil_value([0,0], r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta,tmax=1000)[0][0],
       'equilvalue_constAttack_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','y_eq',logz=False) 

Amin,Amax=0,0.025
plot2d(np.linspace(Amin, Amax, 50),
       np.linspace(RonLmin, RonLmax, 50),
       lambda X,Y: equil_value([0,0], r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, tmax=1000)[0][0],
       'equilvalue_constAttack_vsAandRonL', 'A','R/L','y_eq',logz=False) 
'''

print(eigenvalues(False, r=r, R=R, L=R/3, n0=n0, ym=ym, A=0.02, alpha=alpha, beta=beta, gamma=gamma))

def returnrate_fromeig(X, Y, imag=False):
    eig, eq = eigenvalues(False, r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta, gamma=gamma)
    if eq[0] > 0.99999:
        return 0
    else:
        return np.max(np.imag(eig) if imag else np.real(eig))
    
def tip_time(y, thres, t_above_thres=10):
    tip_times = np.where(y > thres)[0]
    consec_seq = np.split(tip_times, np.where(np.diff(tip_times) != 1)[0]+1)
    tipt = len(y)
    for seq in consec_seq:
        if len(seq) >= t_above_thres:
            tipt = seq[0]
            break
    return(tipt) 

def tip_time_vseq(y, yeq, distfromeq):
    # last time when abs(y-yeq) < distfromeq
    ydiff = np.abs(y-yeq)
    ydiff = ydiff[::-1]
    return len(ydiff) - np.argmax(ydiff < distfromeq)

def mean_squaredev(y, ymean, sw=10):
    squares = pd.Series((y-ymean)**2)
    rollmean = np.array(squares.rolling(window=sw).mean())
    for i in range(sw-1):
        rollmean[i] = np.mean(squares[0:(i+1)])
    return rollmean

def ComputeSlope(AA,RonL,Rapproach,incr,noisy,L):
    '''
    slopes of variance and return rate for a single value of A or R/L to be tested (with ramping up to this value)
    '''

    t_before = 100
    t_after = 200

    tchange = np.array(np.hstack(([0] , np.linspace(t_before,t_before+incr, 30), [t_before+incr+t_after])), dtype=int) if incr>0 else np.array([0,t_before,t_before+t_after])
    nsteps = len(tchange)-1
    L = np.full(nsteps, L)

    R = L*np.linspace(12,RonL, nsteps) if Rapproach else np.full(nsteps, L*RonL)
    Achange = np.linspace(0.01, AA, nsteps)
    if noisy:
        A = np.zeros(nsteps)
        noisemag = np.full(nsteps, AA) if Rapproach else Achange
    else:
        noisemag = np.full(nsteps, 0.005)
        A = np.full(nsteps, AA) if Rapproach else Achange

    sol = run_model_parameterchange_noisy(tchange, r, R, L, n0, ym, A, noisemag, alpha, beta, gamma, absnoise=noisy)
    y = sol[0][:,0]

    y_eq_th = np.array([equil_value([0,1000], r, Rt, Lt, n0, ym, At, alpha,beta,gamma)[0] for (Rt,Lt,At) in zip(R,L,Achange)])
    y_eq_th_t = []
    Achange_t = []
    A_t = []
    for i in range(nsteps):
        y_eq_th_t += [y_eq_th[i]] * (tchange[i+1] - tchange[i])
        Achange_t += [Achange[i]] * (tchange[i+1] - tchange[i])
        A_t += [A[i]] * (tchange[i+1] - tchange[i])
    Achange_t.append(Achange_t[-1])
    Achange_t = np.array(Achange_t)
    A_t.append(A_t[-1])
    A_t = np.array(A_t)
    y_eq_th_t.append(y_eq_th_t[-1])
    y_eq_th_t = np.array(y_eq_th_t)
    y_eq_th_t_no1 = np.copy(y_eq_th_t)
    if np.any(y_eq_th_t_no1<0.99):
        y_eq_th_t_no1[y_eq_th_t_no1>0.99] = np.max(y_eq_th_t_no1[y_eq_th_t_no1<0.99])
    else:
        y_eq_th_t_no1 = 0
    vari = mean_squaredev(y, y_eq_th_t_no1[:,0])
    vari0 = mean_squaredev(y, 0)
    returnamount = np.hstack(([0], np.abs(y[:-1] - y[1:] + sol[3][1:] + A_t[1:]))) # difference between y in consecutive steps + noise and attack that was added to y at that step
    perturb = np.maximum(np.abs(y-y_eq_th_t_no1[:,0]), Achange_t) # size of perturbation is the difference with the current equilibrium at these parameters, with a lower threshold at the noise amplitude
    returnr = returnamount / perturb
    returnr0 = returnamount / np.maximum(y, Achange_t)

    # find tipping time
    tip_thres = 0.95
    tip_t = tip_time(y, tip_thres, 10)
    #tip_t = tip_time_vseq(y, yeq, 0.2)

    # maximum index for which to look at the variance
    istip = tip_t < len(y)
    tmax_look = tip_t if istip else t_before+10 + np.argmax(y[t_before+10:])
    tmin_look = min(t_before, tmax_look-20)
    inds = np.arange(tmin_look, tmax_look)

    # fit the slope of variance and return time until tmax_look 
    vari_pol = np.polyfit(inds, vari[inds], deg=1)
    vari0_pol = np.polyfit(inds, vari0[inds], deg=1)
    returnr_pol = np.polyfit(inds, returnr[inds], deg=1)
    returnr0_pol = np.polyfit(inds, returnr0[inds], deg=1)
    
    doplot = np.remainder(int(AA*10000)+int(RonL*10) , 10) == 9
    if doplot: # somewhat random but always the same in each run
        plot_yvsn(sol[0], sol[1], returnamount, sol[3], 'yvst_noisy'+str(int(noisy))+'_incr'+str(int(incr))+'_Rapproach'+str(int(Rapproach))+'_A'+str(AA)+'_RonL'+str(RonL),
                  R/L, (noisemag if noisy else A), tchange, 
                  perturb, y_eq_th_t[:,0], vari, vari_pol, returnr_pol, returnr0, returnr0_pol, vari0, vari0_pol, inds)

    return returnr_pol[0], returnr0_pol[0], vari_pol[0], vari0_pol[0], int(istip)


Amin,Amax=0.011,0.1
sigma=0.005

for noisy in [False, True]:
    for incr in [0,50,100,300]:
        for Rapproach in [False, True]:
            def slopes(AA, RRonLL):
                res = ComputeSlope(AA,RRonLL,Rapproach,incr,noisy,L)
                #print('A',AA,'R/L',RRonLL, res)
                return res
            plot2d(np.linspace(Amin, Amax, 35),
                   np.linspace(RonLmin if Rapproach else 2, 8 if Rapproach else 12, 35),
                   slopes,
                   'slope_noisy'+str(int(noisy))+'_incr'+str(int(incr))+'_Rapproach'+str(int(Rapproach)), 
                   'A','R/L','slope from fit',twoz=True)
                






plot2d(np.linspace(Amin, Amax, 35),
       np.linspace(RonLmin,12, 35),
        lambda X,Y: variance(r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta, gamma=gamma, sigma=sigma),
       'variance_sigma'+str(sigma)+'_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','variance (simulated small signed noise)',logz=False) 


plot2d(np.linspace(Amin, Amax, 70), 
       np.linspace(RonLmin,12, 70),
        lambda X,Y: returnrate(r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta, gamma=gamma)[1],
       'returnrateSimulated_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','return rate (simulated)',logz=False,zmax=0) 

plot2d(np.linspace(Amin, Amax, 70),
       np.linspace(RonLmin,12, 70),
       returnrate_fromeig,
       'largesteigenvalue_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','largest eigenvalue ( - return rate)',logz=False) 

plot2d(np.linspace(Amin, Amax, 70),
       np.linspace(RonLmin,12, 70),
       lambda X,Y: returnrate_fromeig(X,Y,True),
       'eigenvalueImaginarypart_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','eigenvalue imaginary part (oscillation frequency)',logz=False) 

plot2d(np.linspace(Amin, Amax, 70),
       np.linspace(RonLmin,12, 70),
       lambda X,Y: equil_value([0,0], r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta, gamma=gamma, tmax=1000)[0][0],
       'equilvalue_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','y_eq',logz=False) 
plot2d(np.linspace(Amin, Amax, 70),
       np.linspace(RonLmin,12, 70),
       lambda X,Y: equil_value([0,0], r=r, R=R, L=R/Y, n0=n0, ym=ym, A=X, alpha=alpha, beta=beta, gamma=gamma, tmax=1000)[0][1],
       'equilvalue_n_alpha'+str(int(alpha))+'_vsAandRonL', 'A','R/L','n_eq',logz=False) 
sys.exit()

nmin,nmax=0,1000
y0min,y0max=0,1
plot2d(np.linspace(y0min, y0max, 50),
       np.linspace(nmin, nmax, 50),
       lambda X,Y: equil_value([X,Y], r=r, R=R, L=L, n0=n0, ym=ym, A=A, tmax=1000, alpha=alpha,beta=beta)[0][0],
       'equilvalue_alpha'+str(int(alpha))+'_vsyAndnInit', 'y(t=0)','n(t=0)','y_eq',logz=False) 

#with noisy model
""" 
plot2d(np.linspace(Rmin, Rmax, 40),
       np.linspace(RonLmin, RonLmax, 40),
       lambda X,Y: np.mean(model_2D_noisy([0,0], 500, r=r, R=X, L=X/Y, n0=n0, ym=ym, A=0, noisemag=noisemag, alpha=alpha)[0][-100:-1, 0]),
       'equilvalue_noisy_noisemag0p'+"{:03d}".format(int(1000*noisemag))+'_vsRandRonL', 'R','R/L','y_eq',logz=False) 

noisemagmin,noisemagmax=0,0.5
plot2d(np.linspace(noisemagmin, noisemagmax, 40),
       np.concatenate((np.linspace(RonLmin,4, 30),np.linspace(4,9, 20))),
       lambda X,Y: np.mean(model_2D_noisy([0,0],600, r=r, R=R, L=R/Y, n0=n0, ym=ym, A=0, noisemag=X, alpha=alpha)[0][-100:-1, 0]),
       'equilvalue_noisy_alpha'+str(int(alpha))+'_vsnoisemagandRonL', 'attack noise standard deviation','R/L','y_eq',logz=False) 


for A in np.linspace(0,1,11):
    plot2d(np.linspace(0, 1, 50),
           np.linspace(RonLmin, RonLmax, 50),
           lambda X,Y: equil_value([X,0], r=r, R=R, L=R/Y, n0=n0, ym=ym, A=A, alpha=alpha, tmax=1000)[0][0],
           'equilvalue_constAttack'+str(A)+'_alpha'+str(int(alpha))+'_vsy0andRonL', 'y0','R/L','y_eq',logz=False) 
 """
""" 
Rmin,Rmax = 10,300
Lmin,Lmax = 2,50
plot2d(np.linspace(Rmin, Rmax, 50),
       np.linspace(Lmin, Lmax, 50),
       lambda X,Y: returntime([0.3,0], r=r, R=X, L=Y, n0=n0, ym=ym, tmax=1000),
       'returntime_vsRandL', 'R','L','return time',logz=True)

Rmin,Rmax = 10,300
RonLmin, RonLmax = 0.1, 12
plot2d(np.linspace(Rmin, Rmax, 50),
       np.linspace(RonLmin, RonLmax, 50),
       lambda X,Y: returntime([0.3,0], r=r, R=X, L=X/Y, n0=n0, ym=ym, tmax=1000),
       'returntime_vsRandRonL', 'R','R/L','return time',logz=True)

 """
if False:
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(t, sol[:,0], label='y')
    axs[1].plot(t,sol[:,1], label='n')
    axs[0].legend()
    axs[1].legend()
    plt.savefig('test.png')

