'''
Directionality Spectra

luis.montejo@upr.edu
alan.rivera@upr.edu

This module contains python functions that allow the computation of response 
directionality spectrum as described in Rivera-Figueroa and Montejo (2021). 
The functions included can also be used to load acceleration records in 
PEER/NGA format and compute RotD100/RotD50 response spectra.

Rivera-Figueroa, A., & Montejo, L.A. (2021). Spectral Matching RotD100 Target 
Spectra: Effect on records characteristics and seismic response. 
Earthquake Spectra (submitted for publication).

Functions:

    dfactor :               computes a single directionality factor 
    DFSpectra :             computes rotated response spectra and directionality 
                            response spectrum
    load_PEERNGA_record :   load record in .at2 format (PEER NGA Databases)   

'''

def dfactor(x,y, plot=1):
    
    '''
    
    dfactor - function to determine the directionality factor of a ground motion record
    INPUT:
        x       = x-coordinate of the response
        y       = y-coordinate of the response
        plots   = 1/0 (yes/no, whether plots are generated, default 1)
        
    RETURNS:
        hr, htheta =  polar coordinates of the envelope (convex hull)
        rmax       =  max response (radius)
        thetamax   =  angle for rmax
        df         =  directionality factor
    '''
    import numpy as np
    from scipy.spatial import ConvexHull
    
    n1 = np.size(x); n2 = np.size(y); npo = np.min((n1,n2))
    x = x[:npo]; y = y[:npo]
    
    # Change data to polar coordinates:   
    r = np.sqrt(x**2+y**2)      # radius
    theta = np.arctan2(y,x)     # angle in radians
    
    # Maximum radius and angle of occurence:
    rmax = np.amax(r)
    thetamax = theta[np.argmax(r)]
    
    # Determine the envelope of data points (convex hull):
    points  = np.column_stack((x,y))    # stack data in columns
    hull    = ConvexHull(points)        # convex hull 
    
    # Obtain coordinates of the envelope:
    xh = points[hull.vertices,0]        # x-coordinate of hull
    yh = points[hull.vertices,1]        # y-coorindate of hull
    
    # Change envelope coordinates to polar:
    hr = np.sqrt(xh**2+yh**2)      # radius
    htheta = np.arctan2(yh,xh)     # angle in radians
    
    # Determine area of hull and circle with max radius:
    hArea   = 0.5*np.abs(np.dot(xh,np.roll(yh,1)) 
                         - np.dot(yh,np.roll(xh,1)))    # hull area
    mArea   = np.pi*rmax**2                             # circle area

    # Calculate directionality factor                          
    df = mArea/hArea
    
# =============================================================================
#     PLOT:
# =============================================================================
    
    if plot:
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-talk')
        plt.figure(figsize=(5,4))
        plt.style.use('seaborn-talk')
        rlimit = rmax*1.05
        ax = plt.subplot(111, projection = 'polar')
        ax.plot([0, thetamax], [0, rmax] ,'-o', color='tab:orange', zorder = 3)
        ax.plot(theta, r, color='tab:blue', zorder = 2, linewidth = 1.5)
        ax.fill(htheta,hr, color='tab:blue',alpha = 0.5, linewidth=3, zorder=2)
        circle = plt.Circle((0, 0), rmax, color="tab:orange", 
                 transform=ax.transData._b, alpha=0.5, linewidth=3, zorder=1)
        ax.add_artist(circle)
        ax.set_facecolor('whitesmoke')
        ax.set_rlabel_position(0)
        ax.set_xlabel('DF = %.2f' %df, labelpad = -10, 
                        bbox=dict(boxstyle='square', fc='whitesmoke', ec = 'k'))
        lines, labels = plt.thetagrids(range(0,360,45),())
        lines, labels = ax.set_rgrids((rlimit*.25, rlimit*.5, rlimit*.75, 
                                            rlimit), fontsize = 8, fmt='%.2f')
        
        # To anotate the max resp and ocurrence angle:
        offset = (0,-30) if thetamax<0 else (0,30)
        if np.abs(thetamax+np.pi/2)<0.001: offset = (-30,0)
        hp = 'left' if thetamax<np.pi/2 and thetamax>-1*np.pi/2 else 'right'
        ax.annotate('(%.1f, %.0fº)'% (rmax, np.degrees(thetamax)), 
                      xy=(thetamax,rmax), 
                      textcoords='offset points', fontsize = 10,
                      xytext=offset, ha=hp, va='center',
                      arrowprops=dict(arrowstyle='->',  
                      connectionstyle="angle3,angleA=0,angleB=90"),
                      bbox = dict(boxstyle='round', fc='0.99'), zorder = 5)
        
        plt.suptitle('Directionality Factor') 
        plt.tight_layout(rect=(0,0,1,1))
    
    return(hr, htheta, rmax, np.degrees(thetamax), df)

def DFSpectra(T,s1,s2,z,dt,theta,plot=1):
    '''
    
    Rotated response spectra and directionality response spectra
    
    Input:
    T:          vector with periods (s)
    s1,s2:      accelerations time series
    z:          damping ratio
    dt:         time steps for s
    theta:      vector with the angles to calculate the spectra (deg)
    
    Returns:
    
    RotD100
    RotD50
    ratios RotD100/RotD50
    Directionality spectrum of accelerarion (DSA)

    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.fft import fft, ifft
    percentiles=[50,100]
    pi = np.pi
    theta = theta*pi/180
    
    ntheta = np.size(theta)
    
    n1 = np.size(s1); n2 = np.size(s2); npo = np.min((n1,n2))
    s1 = s1[:npo]; s2 = s2[:npo]
    
    
    nT  = np.size(T)
    
    SD  = np.zeros((ntheta,nT))
    
    nor = npo
    
    n = int(2**np.ceil(np.log2(npo+10*np.max(T)/dt)))  # add zeros to provide enough quiet time
    fs=1/dt;
    s1 = np.append(s1,np.zeros(n-npo))
    s2 = np.append(s2,np.zeros(n-npo))
    
    fres  = fs/n                            # frequency resolution
    nfrs  = int(np.ceil(n/2))               # number of frequencies
    freqs = fres*np.arange(0,nfrs+1,1)      # vector with frequencies
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts1 = fft(s1)        
    ffts2 = fft(s2) 
    
    DSA = np.zeros(nT)
    
    m = 1
    
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        H3 = -ww**2  / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Accelerance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        H3 = np.append(H3,np.conj(H3[n//2-1:0:-1]))
        H3[n//2] = np.real(H3[n//2])     # Transfer function (complete) - Accelerance
        
        
        CoFd1 = H1*ffts1   # frequency domain convolution
        d1 = ifft(CoFd1)   # go back to the time domain (displacement)
        d1 = np.real(d1[:nor])
        
        CoFd2 = H1*ffts2   # frequency domain convolution
        d2 = ifft(CoFd2)   # go back to the time domain (displacement)
        d2 = np.real(d2[:nor])
        
        Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
        Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
        drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta)
        
        SD[:,kk] = np.max(np.abs(drot),axis=1)
              
        CoFa1 = H3*ffts1   # frequency domain convolution
        a1 = np.real(ifft(CoFa1))   # go back to the time domain 
        a1 = a1 - s1
        a1 = a1[:nor]
        
        CoFa2 = H3*ffts2   # frequency domain convolution
        a2 = np.real(ifft(CoFa2))   # go back to the time domain 
        a2 = a2 - s2
        a2 = a2[:nor]
        
        _,_,_,_,DSA[kk] = dfactor(a1,a2, plot=0)
        
    
    PSA180 = (2*pi/T)**2 * SD
    n = len(percentiles)
    PSArotnn = np.zeros((nT,n))
    
    for k in range(n):
        PSArotnn[:,k] = np.percentile(PSA180,percentiles[k],axis=0)
    
    RotD50 = PSArotnn[:,0]
    RotD100 = PSArotnn[:,1]
    ratios = RotD100/RotD50
    
    if plot:
        
        plt.figure(figsize=(6.5,9))
        plt.subplot(311)
        plt.semilogx(T,RotD50,'-b')
        plt.semilogx(T,RotD100,'-g')
        plt.legend((': RotD50',': RotD100'))
        plt.ylabel('PSA [g]')
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
        plt.subplot(312)
        plt.semilogx(T,ratios,'-k')
        plt.ylabel('RotD100/RotD50')
        plt.ylim((0.9,1.1*np.max(ratios)))
        plt.xlabel('T [s]')
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
        plt.subplot(313)
        plt.semilogx(T,DSA,'-k')
        plt.xlabel('T [s]')
        plt.ylabel('Directionality Factor')
        plt.ylim((0.9,1.1*np.max(DSA)))
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
    
    return RotD100, RotD50, ratios, DSA


def load_PEERNGA_record(filepath):
    '''
    Load record in .at2 format (PEER NGA Databases)

    Input:
        filepath : file path for the file to be load
        
    Returns:
    
        acc : vector wit the acceleration time series
        dt : time step
        npts : number of points in record
        eqname : string with year_name_station_component info

    '''

    import numpy as np

    with open(filepath) as fp:
        line = next(fp)
        line = next(fp).split(',')
        year = (line[1].split('/'))[2]
        eqname = (year + '_' + line[0].strip() + '_' + 
                  line[2].strip() + '_comp_' + line[3].strip())
        line = next(fp)
        line = next(fp).split(',')
        npts = int(line[0].split('=')[1])
        dt = float(line[1].split('=')[1].split()[0])
        acc = np.array([p for l in fp for p in l.split()]).astype(float)
    
    return acc,dt,npts,eqname