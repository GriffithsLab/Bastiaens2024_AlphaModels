import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from fooof import FOOOF


# Jansen-Rit

# Sigmoid function
def sigm(nu_max,v0,r,v):
  action_potential = 2*nu_max/(1+math.exp(r*(v0-v)))
  return action_potential

# Parameter settings
A = 3.25       # mV
B = 22         # mV
C1 = 135       # na
C2 = 0.8*C1    # na
#C3 = 0.25*C1
#C4 = 0.25*C1
v0 = 6         # mV
a = 100        # s^-1
b = 50         # s^-1
nu_max = 2.5   # s^-1
r = 0.56       # mV^-1

C3_tot = np.arange(0.100,0.699, 0.01) # Varying C3
C4_tot = np.arange(0.100,0.699, 0.01) # Varying C4

# Simulation settings
start = 0.0
stim_time = 10
dt = 0.001
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)
noise = np.random.uniform(120,320,vec_len)
# We are using uniform noise but it is also possible to look at  normal noise with Euler-Maruyama method
#noise = A*a*math.sqrt(dt)*math.sqrt(22)*np.random.normal(0, 1, vec_len) + dt*220*A*a

# Output Initialization
y = np.zeros((6,vec_len))

final_WELCH = np.zeros((len(C3_tot), len(C4_tot)))

# Determine the frequency peak value for each combinations of C3 and C4 with all the other parameters set to resting state values
for ind2, C3_prop in enumerate(C3_tot):
  for ind, C4_prop in enumerate(C4_tot):
    C3 = C3_prop*C1
    C4 = C4_prop*C1
    for i in range (1,vec_len):
      y[0,i] = y[0,i-1] + y[3,i-1]*dt
      y[1,i] = y[1,i-1] + y[4,i-1]*dt
      y[2,i] = y[2,i-1] + y[5,i-1]*dt
      y[3,i] = y[3,i-1] + dt * (A*a*(sigm(nu_max,v0,r,(y[1,i-1]-y[2,i-1]))) - (2*a*y[3,i-1]) - (a**(2)*y[0,i-1]))
      y[4,i] = y[4,i-1] + dt * (A*a*(noise[i-1] + (C2*sigm(nu_max,v0,r,(C1*y[0,i-1])))) - (2*a*y[4,i-1]) - (a**(2)*y[1,i-1]))
      y[5,i] = y[5,i-1] + dt * (B*b*(C4*sigm(nu_max,v0,r,(C3*y[0,i-1]))) - (2*b*y[5,i-1]) - (b**(2)*y[2,i-1]))

    out = y[1,:]-y[2,:]

    if out[2000] == out[2005]:
      freqWelch = 0
    else:
      output = out[1000:] 
      # Calculate power spectrum
      freqsW,ps_vPN = welch(output,fs=1000, noverlap = 125, nperseg=1000)
      fm = FOOOF(max_n_peaks=2, min_peak_height=1, aperiodic_mode='knee')
      fm.fit(freqsW, ps_vPN, [1,50])
      cfs = fm.get_params('peak_params', 'CF')
      if np.isnan(cfs).any():
        final_WELCH[ind, ind2] = 0
      elif cfs.shape ==():
        final_WELCH[ind, ind2] = cfs
      else:
        pws = fm.get_params('peak_params', 'PW')
        final_WELCH[ind, ind2] = cfs[np.argmax(pws)]

data2 = pd.DataFrame(np.transpose(final_WELCH), index = np.round(C3_tot,3), columns=np.round(C4_tot,3))
ax = sns.heatmap(data2, vmax=11, cbar_kws={'label': 'Dominant Frequency (Hz)'} )
plt.xlabel('C4')
plt.ylabel('C3')
ax.invert_yaxis()


# Liley-Wright
def dynamics(p,X):
    dX = np.zeros((18,1))

    # Calculate synaptic reversal potentials
    psi_ee=(p.h_ee_eq-X[0])/abs(p.h_ee_eq-p.h_e_r)
    psi_ie=(p.h_ie_eq-X[0])/abs(p.h_ie_eq-p.h_e_r)
    psi_ei=(p.h_ei_eq-X[1])/abs(p.h_ei_eq-p.h_i_r)
    psi_ii=(p.h_ii_eq-X[1])/abs(p.h_ii_eq-p.h_i_r)

    # Calculate synaptic inputs A_jk 
    A_ee=p.N_ee_b*S_e(p,X[0])+X[10]+p.p_ee
    A_ei=p.N_ei_b*S_e(p,X[0])+X[12]+p.p_ei
    A_ie=p.N_ie_b*S_i(p,X[1])
    A_ii=p.N_ii_b*S_i(p,X[1])   

    # Calculate state vector
    dX[0] = (1/p.tau_e)*(p.h_e_r-X[0]+psi_ee*X[2]+psi_ie*X[6]) # V_e
    dX[1] = (1/p.tau_i)*(p.h_i_r-X[1]+psi_ei*X[4]+psi_ii*X[8]) # V_i
    dX[2] = X[3] # I_ee
    dX[3] = -(p.gamma_ee+p.gamma_ee_tilde)*X[3]     -p.gamma_ee*p.gamma_ee_tilde*X[2]   +p.gamma_ee_tilde*math.exp(p.gamma_ee/p.gamma_ee)*X[14]*A_ee# J_ee
    dX[4] = X[5] # I_ei
    dX[5] = -(p.gamma_ei+p.gamma_ei_tilde)*X[5]     -p.gamma_ei*p.gamma_ei_tilde*X[4]   +p.gamma_ei_tilde*math.exp(p.gamma_ei/p.gamma_ei)*X[15]*A_ei# J_ei
    dX[6] = X[7] # I_ie
    dX[7] = -(p.gamma_ie+p.gamma_ie_tilde)*X[7]     -p.gamma_ie*p.gamma_ie_tilde*X[6]   +p.gamma_ie_tilde*math.exp(p.gamma_ie/p.gamma_ie)*X[16]*A_ie# J_ie
    dX[8] = X[9] # I_ii
    dX[9] = -(p.gamma_ii+p.gamma_ii_tilde)*X[9]  -p.gamma_ii*p.gamma_ii_tilde*X[8]   +p.gamma_ii_tilde*math.exp(p.gamma_ii/p.gamma_ii)*X[17]*A_ii# J_ii% J_ii
    dX[10] = X[11] #theta_ee
    dX[11] = -2*p.nu_ee*p.Lambda_ee*X[11]           -p.nu_ee**(2)*p.Lambda_ee**(2)*X[10]     +p.nu_ee**(2)*p.Lambda_ee**(2)*p.N_ee_a*S_e(p,X[0])#d_theta_ee
    dX[12] = X[13]#theta_ei
    dX[13] = -2*p.nu_ei*p.Lambda_ei*X[13]           -p.nu_ei**(2)*p.Lambda_ei**(2)*X[12]+p.nu_ei**(2)*p.Lambda_ei**(2)*p.N_ei_a*S_e(p,X[0])#d_theta_ei   
    dX[14] = p.Gamma_ee-X[14]  # Gamma_ee
    dX[15] = p.Gamma_ei-X[15]  # Gamma_ei   
    dX[16] = p.Gamma_ie-X[16] # Gamma_ie 
    dX[17] = p.Gamma_ee-X[17]  # Gamma_ii   

    return dX
    
def S_e(t,v):   
    p = t
    spikerate = p.S_e_max/(1 + math.exp(-math.sqrt(2)*(v - p.mu_e)/p.sigma_e))
    return spikerate

def S_i(t,v):
    p=t
    spikerate = p.S_i_max/(1 + math.exp(-math.sqrt(2)*(v - p.mu_i)/p.sigma_i))
    return spikerate

# Parameter setting
class p:
  S_e_max = 0.5    # ms^-1
  S_i_max = 0.5    # ms^-1
  h_e_r = -7       # mV
  h_i_r = -70      # mV
  mu_e = -50       # mV
  mu_i = -50       # mV
  sigma_e = 5      # mV
  sigma_i = 5      # mV
  tau_e = 94       # ms
  tau_i = 42       # ms
  h_ee_eq = 45     # mV
  h_ei_eq = 45     # mV
  h_ie_eq = -90    # mV
  h_ii_eq = -90    # mV
  Gamma_ee = 0.71  # mV
  Gamma_ei = 0.71  # mV
  Gamma_ie = 0.71  # mV
  gamma_ee = 0.3   # ms^-1
  gamma_ei = 0.3   # ms^-1
  gamma_ie = 0.065 # ms^-1
  gamma_ii = 0.065 # ms^-1
  p_ee = 3.460     # ms^-1
  p_ee_sd = 1.000  # std
  p_ei = 5.070     # ms^-1
  p_ei_sd = 0      # std
  p_ie = 0         # ms^-1
  p_ii = 0         # ms^-1
  N_ei_b = 3000    # na
  N_ee_b = 3000    # na
  N_ie_b = 500     # na
  N_ii_b = 500     # na

  N_ei_tot = np.arange(1000,5100,1000) # Varying N_ei
  N_ie_tot = np.arange(100,1100, 500)  # Varying N_ie


# Simulation settings
sim_time=50           
dt = 1e-4
steps= 1/dt          
white_noise=1          

p.v_e_equil = p.h_e_r
p.v_i_equil = p.h_i_r
h = 1000/steps 
T = sim_time*10**3 
N = T/h-1

final_connection = np.zeros((len(p.N_ei_tot), len(p.N_ie_tot)))

# Determine frequency peak values for all combination of Nei and Nie
for ind2, p.N_ei_b in enumerate(p.N_ei_tot):
  for ind, p.N_ie_b in enumerate(p.N_ie_tot):
    X = np.zeros((18,int(N)))#      % initialization of state vector
    X[0,0] =   p.v_e_equil#  
    X[1,0] =   p.v_i_equil#    
    X[2,0] =   math.exp(1)/p.gamma_ee*p.Gamma_ee*(p.N_ee_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ee)#    
    X[4,0] =   math.exp(1)/p.gamma_ei*p.Gamma_ei*(p.N_ei_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ei)#
    X[6,0] =   math.exp(1)/p.gamma_ie*p.Gamma_ie*(p.N_ie_b*S_i(p,p.v_i_equil))#
    X[8,0] =   math.exp(1)/p.gamma_ii*p.Gamma_ii*(p.N_ii_b*S_i(p,p.v_i_equil))#
    
    for n in range (0,int(N-1)):
        noise = np.zeros((18,1))
        if (white_noise==1):
          noise[3]= p.gamma_ee*math.exp(p.gamma_ee/p.gamma_ee)*p.Gamma_ee*p.p_ee_sd*np.random.randn(1,1)
        X[:,n+1] = X[:,n] + ((h*dynamics(p,X[:,n])+math.sqrt(h)*noise).flatten())
    EEG=-X[0,:]	

    if np.round(EEG[2000],6) == np.round(EEG[2005],6):
      final_connection[ind2,ind] = 0
    else:
      output = EEG[1000:]
      # Welch method
      
      X = signal.resample(output, 5000)
      freqs_new,ps_vPN_new = welch(X,fs=100, noverlap = 125, nperseg=1000)
      fm = FOOOF(max_n_peaks=2, min_peak_height=1, aperiodic_mode='knee')
      fm.fit(freqs_new, ps_vPN_new, [1,50])
      if not fm.has_model:
        fm = FOOOF()
        fm.fit(freqs_new, ps_vPN_new, [1,50])
      cfs = fm.get_params('peak_params', 'CF')
      if np.isnan(cfs).any():
        final_connection[ind2, ind] = 0
      elif cfs.shape ==():
        final_connection[ind2, ind] = cfs
      else:
        pws = fm.get_params('peak_params', 'PW')
        final_connection[ind2, ind] = cfs[np.argmax(pws)]


data2 = pd.DataFrame(final_connection, index = p.N_ei_tot, columns=p.N_ie_tot)
ax = sns.heatmap(data2, vmax = 20)
ax.invert_yaxis()


# Robinson-Rennie-Wright

# Sigmoid functions
def sig(v,Qmax,theta,sigma):
 # sigmoid for voltage-rate relationship
 firing_rate = Qmax / (1 + math.exp(-(v-theta) / sigma))
 return firing_rate

def sigr(v,Qmax_r,modtheta,sigma_r):
 # sigmoid for voltage-rate relationship in reticular nucleus
 firing_rate = Qmax_r / (1 + math.exp(-(v-modtheta) / sigma_r));
 return firing_rate

gamma = 116      # s^-1
t0 = 80e-3       # s
Qmax = 340       # s^-1
theta = 12.92e-3 # V
sigma = 3.8e-3   # V
Qmax_r = 100     # s^-1
sigma_r = 3.8e-3 # V

phin_0 = 1.0     # s^-1
nu_ee = 0.00303  # Vs
nu_ei = -0.006   # Vs
nu_ie = 0.00303  # Vs
nu_ii = -0.006   # Vs
nu_is = 0.00206  # Vs
nu_re = 0.00033  # Vs
nu_sn = 0.00098  # Vs
alpha = 83.33    # s^-1
beta = 769.23    # s^-1
phin = 5e-4      # s^-1    
nu_es = 0.00206  # s^-1 
nu_se = 0.00218  # s^-1  


# Simulation settings 
start = 0.0
stop = 100.0 + t0/2
dt = 1e-4
k0 = int(t0/2/dt)
time_array = np.arange(start=start, stop=stop, step=dt)
vec_len = len(time_array)

# Noise input
np.random.seed(0)
noise = alpha*beta*nu_sn*math.sqrt(phin*dt)*np.random.randn(vec_len,1)


nu_rs_tot = np.round(np.arange(0.0, 0.00045, 0.00001),6)     # Varying nu_rs
nu_sr_tot = np.round(np.arange(-0.0009, -0.0005, 0.00001),6) # Varying nu_sr
iterables = [nu_rs_tot, nu_sr_tot]
index = pd.MultiIndex.from_product(iterables, names=['nu_rs', 'nu_sr'])
final_Rob = pd.DataFrame(index=index, columns = ['freq'], dtype='float64')
test_pws = pd.DataFrame(index=index, columns = ['values'], dtype='float64')

# Determine frequency peak for all combinations of nu_rs and nu_sr
for nu_rs in nu_rs_tot:
  print('next'+ str(nu_rs))  
  for nu_sr in nu_sr_tot:  
    # Outputs
    phie = np.zeros(vec_len)
    Ve = np.zeros(vec_len)
    Vr = np.zeros(vec_len)
    Vs = np.zeros(vec_len)
    phiedot = np.zeros(vec_len)
    Vedot = np.zeros(vec_len)
    Vrdot = np.zeros(vec_len)
    Vsdot = np.zeros(vec_len)
    # Initialize output
    phie[0:(k0+1)] = 3.175
    Ve[0:(k0+1)]  = 0.0006344; 
    Vr[0:(k0+1)]  = 0.005676;
    Vs[0:(k0+1)]  = -0.003234;

    for i in range ((k0+1),(int(stop/dt))):
        Ve[i] = Ve[i-1] + Vedot[i-1]*dt
        Vedot[i] = Vedot[i-1] + dt *( alpha*beta * ( nu_ee*phie[i-1] + nu_ei*sig(Ve[i-1],Qmax,theta,sigma) + nu_es*sig(Vs[i-1-k0],Qmax,theta,sigma) - (1/alpha + 1/beta)*Vedot[i-1] - Ve[i-1] ) )
          
        Vr[i] = Vr[i-1] + Vrdot[i-1]*dt
        Vrdot[i] = Vrdot[i-1] + dt*( alpha*beta * ( nu_re*phie[i-1-k0] + nu_rs*sig(Vs[i-1],Qmax,theta,sigma) - (1/alpha + 1/beta)*Vrdot[i-1] - Vr[i-1] ))
        
        Vs[i] = Vs[i-1] + Vsdot[i-1]*dt
        Vsdot[i] = Vsdot[i-1] + dt*( alpha*beta * ( nu_se*phie[i-1-k0] + nu_sr*sig(Vr[i-1],Qmax,theta,sigma) - (1/alpha + 1/beta)*Vsdot[i-1] - Vs[i-1] + nu_sn*phin_0 )) + noise[i-1]
        
        phie[i] = phie[i-1] + phiedot[i-1]*dt
        phiedot[i] = phiedot[i-1] + dt*( gamma**(2) * ( sig(Ve[i-1],Qmax,theta,sigma) - 2/gamma*phiedot[i-1] - phie[i-1] ))

    # Remove initial t0-length from outputs
    phie = phie[(k0+1):len(phie)]
    Ve = Ve[(k0+1):len(Ve)]
    Vr = Vr[(k0+1):len(Vr)]
    Vs = Vs[(k0+1):len(Vs)];

    X=signal.resample(phie,10000)
    output_Zhao = phie;
    freqs,ps_vPN = welch(X[1000:],fs=100, noverlap = 125, nperseg=1000)
    #X=signal.resample(phie,10000)
    #output_Zhao = phie[1000:vec_len-1];
    #freqs,ps_vPN = welch(output_Zhao,fs=1000, noverlap = 125, nperseg=1000)
    if np.round(phie[20000],6)  == np.round(phie[30000],6) :      # Case when time series is flat (this can be made more robust)
      final_Rob.loc[(nu_rs, nu_sr), 'freq'] = 0
    else:
      #fm = FOOOF()
      fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='knee')
      fm.fit(freqs,ps_vPN , [2, 70])
      cfs = fm.get_params('peak_params', 'CF')
      bw = fm.get_params('peak_params', 'BW')
      if np.isnan(cfs).any():
        final_Rob.loc[(nu_rs, nu_sr), 'freq'] = 0
      elif cfs.shape ==():
        pws = fm.get_params('peak_params', 'PW')
        if pws>0.5:
          final_Rob.loc[(nu_rs, nu_sr), 'freq'] = cfs
          test_pws.loc[(nu_rs, nu_sr), 'values'] = pws
        else:
          final_Rob.loc[(nu_rs, nu_sr), 'freq'] = 0
      else:
        pws = fm.get_params('peak_params', 'PW')
        if max(pws)>0.5:
          freqY = cfs[np.argmax(pws)]
          final_Rob.loc[(nu_rs, nu_sr), 'freq'] = freqY
          test_pws.loc[(nu_rs, nu_sr), 'values'] = np.max(pws)
        else:
          final_Rob.loc[(nu_rs, nu_sr), 'freq'] = 0
    
new_Rob = final_Rob.unstack(level='nu_sr')
new_Rob.columns=new_Rob.columns.droplevel()
data = pd.DataFrame(new_Rob)
ax = sns.heatmap(data)
ax.invert_yaxis()
ax.invert_xaxis()

rocket_colormap = sns.cm.rocket

# Define the turquoise color
turquoise_color = (0.0, 0.5, 1.0)  # RGB values for turquoise

# Create a custom colormap
n_bins = 256
rocket_part = rocket_colormap(range(0, int(n_bins * 12 / 25)))
turquoise_part = [turquoise_color] * (n_bins - len(rocket_part))

color_segments = list(rocket_part) + list(turquoise_part)
cmap_custom = LinearSegmentedColormap.from_list('custom_colormap', color_segments, N=n_bins)

# Create a heatmap with the custom colormap
ax = sns.heatmap(df, cmap=cmap_custom, cbar_kws={'label': 'Values'})

data.to_csv('Robinson_Heatmap_Data.csv')
df = pd.read_csv('Robinson_Heatmap_Data.csv')
print(df)

plt.plot(time_array[20000:40000],output_Zhao[20000:40000], color='black')

X=signal.resample(phie,10000)
output_Zhao = phie;
freqs_Robinson,ps_vPN_Robinson = welch(X[1000:],fs=100, noverlap = 125, nperseg=1000)
plt.plot(freqs_Robinson,ps_vPN_Robinson)
plt.xscale('log')
plt.yscale('log')





