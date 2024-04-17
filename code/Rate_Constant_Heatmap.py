import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import signal
from scipy.optimize import fsolve
from fooof import FOOOF

# Jansen-Rit

# Sigmoid function
def sigm(nu_max,v0,r,v):
  action_potential = 2*nu_max/(1+math.exp(r*(v0-v)))
  return action_potential

# Model Parameters
A = 3.25             # mV    
B = 22               # mV
C = 135              # na
C1 = 1*C             # na
C2 = 0.8*C           # na
C3 = 0.25*C          # na
C4 = 0.25*C          # na
v0 = 6               # mV

tau_e = np.arange(2,60,2)  # ms
tau_i = np.arange(2,60,2)  # ms
a_val = 1/(tau_e*0.001)    # s^-1
b_val = 1/(tau_i*0.001)    # s^-1


# Simulation settings
start = 0.0
stim_time = 100
dt = 1e-4
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)

noise = np.random.uniform(120,320,vec_len)
y = np.zeros((6,vec_len))
combination_number = []
New_final = np.zeros((len(b_val), len(a_val)))

# Determine frequency peak value for all combinations of tau_e and tau_i
for ind2, a in enumerate(a_val):
  for ind, b in enumerate(b_val):
    for i in range (1,vec_len):
      y[0,i] = y[0,i-1] + y[3,i-1]*dt
      y[1,i] = y[1,i-1] + y[4,i-1]*dt
      y[2,i] = y[2,i-1] + y[5,i-1]*dt
      y[3,i] = y[3,i-1] + dt * (A*a*(sigm(nu_max,v0,r,(y[1,i-1]-y[2,i-1]))) - (2*a*y[3,i-1]) - (a**(2)*y[0,i-1]))
      y[4,i] = y[4,i-1] + dt * (A*a*(noise[i-1] + (C2*sigm(nu_max,v0,r,(C1*y[0,i-1])))) - (2*a*y[4,i-1]) - (a**(2)*y[1,i-1]))
      y[5,i] = y[5,i-1] + dt * (B*b*(C4*sigm(nu_max,v0,r,(C3*y[0,i-1]))) - (2*b*y[5,i-1]) - (b**(2)*y[2,i-1]))

    out = y[1,:]-y[2,:]

    if out[20000] == out[20005]:
      freqWelch = 0
    else:
      output = out[1000:] 
      X = signal.resample(output, 10000)
      freqs_new,ps_vPN_new = welch(X,fs=100, noverlap = 125, nperseg=1000)
      fm = FOOOF(max_n_peaks=2, min_peak_height=1, aperiodic_mode='knee')
      fm.fit(freqs_new, ps_vPN_new, [1,50])
      cfs = fm.get_params('peak_params', 'CF')
      if np.isnan(cfs).any():
        New_final[ind, ind2] = 0
      elif cfs.shape ==():
        New_final[ind, ind2] = cfs
      else:
        pws = fm.get_params('peak_params', 'PW')
        New_final[ind, ind2] = cfs[np.argmax(pws)]
    combination_number.append([ind, ind2])

# Plotting
data = pd.DataFrame(New_final, index = tau_i, columns=tau_e)
plt.figure(figsize=(8,5), dpi=300)
plt.rcParams['font.size'] = '14'
ax = sns.heatmap(data,  cmap='viridis')
ax.set_xlabel(r"${\tau}_{e}$",fontsize=18)
ax.set_ylabel(r"${\tau}_{i}$",fontsize=18)
ax.invert_yaxis()
plt.tight_layout()
plt.show()


# Moran-David-Friston

# Sigmoid function
def sigm(x):
  return (1/(1+math.exp(-rho_1*(x - rho_2)))) - (1/(1+math.exp(rho_1*rho_2)))


H_e = 10                   # mV
H_i = 32                   # mV
tau_e = np.arange(2,60,1)  # ms
tau_i = np.arange(2,60,1)  # ms

kappa_e_val = (1/tau_e) * 1000 # s^-1
kappa_i_val = (1/tau_i) * 1000 # s^-1

gamma_1 = 128    # na
gamma_2 = 128    # na
gamma_3 = 64     # na
gamma_4 = 64     # na
gamma_5 = 4      # na
rho_1 = 2        # na
rho_2 = 1        # na
a= 0   

# Simulation settings
start = 0.0
stim_time = 10
dt = 0.001
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)

#Initialize input and output
x = np.zeros((12,vec_len))
final_rate= np.zeros((len(kappa_e_val), len(kappa_i_val)))

# Determine frequency peak for all combinations of tau_e and tau_i
for ind2, kappa_e in enumerate(kappa_e_val):
  for ind, kappa_i in enumerate(kappa_i_val):
    
    noise = kappa_e*H_e*math.sqrt(dt)*np.random.normal(0, 1, vec_len)
    for i in range (1,vec_len):

      #Inhibitory cells in agranular layers
      x[6,i] = x[6,i-1] + dt * x[7,i-1]
      x[7,i] = x[7,i-1] + dt * (kappa_e*H_e*(gamma_3)*sigm(x[8,i-1]) - 2*kappa_e*x[7,i-1] - kappa_e**(2)*x[6,i-1])
      x[9,i] = x[9,i-1] + dt * x[10,i-1]
      x[10,i] = x[10,i-1] + dt * (kappa_i*H_i*gamma_5*sigm(x[11,i-1]) - 2*kappa_i*x[10,i-1] - kappa_i**(2)*x[9,i-1])
      x[11,i] = x[11,i-1] + dt * (x[7,i-1] - x[10,i-1])

      #Excitatory spiny cells in granualar layer
      x[0,i] =  x[0,i-1] + dt * x[3,i-1]
      x[3,i] = x[3,i-1] + dt * (kappa_e*H_e*(gamma_1*sigm(x[8,i-1]-a)) - 2*kappa_e*x[3,i-1] - kappa_e**(2)*x[0,i-1]) + noise[i-1]    

      #Excitatory pyramidal cells in agranular layers
      x[1,i] = x[1,i-1] + dt * x[4,i-1]
      x[4,i] =  x[4,i-1] + dt * (kappa_e*H_e*(gamma_2*sigm(x[0,i-1])) - 2*kappa_e*x[4,i-1] - kappa_e**(2)*x[1,i-1])
      x[2,i] = x[2,i-1] + dt * x[5,i-1]
      x[5,i] = x[5,i-1] + dt * (kappa_i*H_i*gamma_4*sigm(x[11,i-1]) - 2*kappa_i*x[5,i-1] - kappa_i**(2)*x[2,i-1])
      x[8,i] = x[8,i-1] + dt * (x[4,i-1] - x[5,i-1])
    
    if out[2000] == out[2005]:
        final_rate[ind2, ind] = 0
    else: 
        out = x[8,0:]
        output = out[1000:] 
        freqsW,ps_vPN = welch(output,fs=1000, noverlap = 125, nperseg=1000)
        fm = FOOOF(max_n_peaks=2, min_peak_height=1, aperiodic_mode='knee')
        fm.fit(freqsW, ps_vPN, [1,100])
        cfs = fm.get_params('peak_params', 'CF')
        if np.isnan(cfs).any():
            final_rate[ind2, ind] = 0
        elif cfs.shape ==():
            final_rate[ind2, ind] = cfs
        else:
            pws = fm.get_params('peak_params', 'PW')
            final_rate[ind2, ind] = cfs[np.argmax(pws)]


# Plotting
data = pd.DataFrame(np.transpose(final_rate), index = tau_i, columns=tau_e)
plt.figure(figsize=(8,5), dpi=300)
plt.rcParams['font.size'] = '14'
ax = sns.heatmap(data,  cmap='viridis')
ax.set_xlabel(r"${\tau}_{e}$",fontsize=18)
ax.set_ylabel(r"${\tau}_{i}$",fontsize=18)
ax.invert_yaxis()
plt.tight_layout()
plt.show()


# Liley-Wright

# Function for the dynamics
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
    dX[3] = -2*p.gamma_ee*X[3]-p.gamma_ee**(2)*X[2]+p.gamma_ee*math.exp(1)*p.Gamma_ee*A_ee# J_ee
    dX[4] = X[5] # I_ei
    dX[5] = -2*p.gamma_ei*X[5]-p.gamma_ei**(2)*X[4]+p.gamma_ei*math.exp(1)*p.Gamma_ei*A_ei# J_ei
    dX[6] = X[7] # I_ie
    dX[7] = -2*p.gamma_ie*X[7]-p.gamma_ie**(2)*X[6]+p.gamma_ie*math.exp(1)*p.Gamma_ie*A_ie# J_ie
    dX[8] = X[9] # I_ii
    dX[9] = -2*p.gamma_ii*X[9]-p.gamma_ii**(2)*X[8]+p.gamma_ii*math.exp(1)*p.Gamma_ii*A_ii# J_ii% J_ii
    return dX

# Sigmoid for excitatory and inhibitory population
def S_e(t,v):   
    p = t
    spikerate = p.S_e_max/(1 + math.exp(-math.sqrt(2)*(v - p.mu_e)/p.sigma_e))
    return spikerate

def S_i(t,v):
    p=t
    spikerate = p.S_i_max/(1 + math.exp(-math.sqrt(2)*(v - p.mu_i)/p.sigma_i))
    return spikerate


# Parameter settings

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


  tau_ee = np.arange(1, 10, 0.2) # ms
  tau_ii = np.arange(10, 60, 1)  # ms
  gamma_ee_tot = 1/tau_ee        # ms^-1
  gamma_ii_tot = 1/tau_ii        # ms^-1


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
New_final = np.zeros((len(p.gamma_ii_tot), len(p.gamma_ee_tot)))

# Determine frequency peak for all combinations of tau_ee and tau_ii
for ind2, p.gamma_ee in enumerate(p.gamma_ee_tot):
 for ind, p.gamma_ii in enumerate(p.gamma_ii_tot):
    p.gamma_ei = p.gamma_ee
    p.gamma_ie = p.gamma_ii

    X = np.zeros((18,int(N)))#      % initialization of state vector
    X[0,0] = p.v_e_equil#  
    X[1,0] = p.v_i_equil#    
    X[2,0] = math.exp(1)/p.gamma_ee*p.Gamma_ee*(p.N_ee_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ee)#    
    X[4,0] = math.exp(1)/p.gamma_ei*p.Gamma_ei*(p.N_ei_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ei)#
    X[6,0] = math.exp(1)/p.gamma_ie*p.Gamma_ie*(p.N_ie_b*S_i(p,p.v_i_equil))#
    X[8,0] = math.exp(1)/p.gamma_ii*p.Gamma_ii*(p.N_ii_b*S_i(p,p.v_i_equil))
    
    for n in range (0,int(N-1)):
        noise = np.zeros((18,1))
        if (white_noise==1):
          noise[3]= p.gamma_ee*math.exp(p.gamma_ee/p.gamma_ee)*p.Gamma_ee*p.p_ee_sd*np.random.randn(1,1)
        X[:,n+1] = X[:,n] + ((h*dynamics(p,X[:,n])+math.sqrt(h)*noise).flatten())
    EEG=-X[0,:]	

    if np.round(EEG[2000],6) == np.round(EEG[2005],6):
      freqWelch = 0
    else:
      output = EEG[1000:]
      X = signal.resample(output, 5000)
      freqs_new,ps_vPN_new = welch(X,fs=100, noverlap = 125, nperseg=1000)
      fm = FOOOF(max_n_peaks=2, min_peak_height=1, aperiodic_mode='knee')
      fm.fit(freqs_new, ps_vPN_new, [1,50])
      cfs = fm.get_params('peak_params', 'CF')
      if np.isnan(cfs).any():
        New_final[ind, ind2] = 0
      elif cfs.shape ==():
        New_final[ind, ind2] = cfs
      else:
        pws = fm.get_params('peak_params', 'PW')
        New_final[ind, ind2] = cfs[np.argmax(pws)]


# Plotting
data = pd.DataFrame(New_final, index = p.tau_ii, columns=p.tau_ee)
plt.figure(figsize=(8,5), dpi=300)
plt.rcParams['font.size'] = '14'
ax = sns.heatmap(data,  cmap='viridis')
ax.set_xlabel(r"${\tau}_{e}$",fontsize=18)
ax.set_ylabel(r"${\tau}_{i}$",fontsize=18)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

