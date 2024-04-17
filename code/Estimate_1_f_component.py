import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from fooof import FOOOF
from scipy.optimize import fsolve
from scipy import signal
from scipy.signal import welch


# Jansen-Rit

# Sigmoid function
def sigm(nu_max,v0,r,v):
  action_potential = (2*nu_max)/(1+math.exp(r*(v0-v)))
  return action_potential

# Parameter settings
A = 3.25            # mV 
B = 22              # mV 
C = 135             # na
C1 = 1*C            # na
C2 = 0.8*C          # na 
C3 = 0.25*C         # na 
C4 = 0.25*C         # na
v0 = 6              # mV
tau_e = 10          # ms
tau_i = 20          # ms
a = (1/tau_e)*1000  # s^-1
b = (1/tau_i)*1000  # s^-1
nu_max = 2.5        # s^-1
r = 0.56            # mV^-1

# Simulation settings
start = 0.0
stim_time = 100
dt = 1e-4
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)

# Input
noise = np.random.uniform(120,320,vec_len)
# Output Initialization
y = np.zeros((6,vec_len))

# Euler integration method to solve JR differential equations
for i in range (1,vec_len):
  y[0,i] = y[0,i-1] + y[3,i-1]*dt
  y[1,i] = y[1,i-1] + y[4,i-1]*dt
  y[2,i] = y[2,i-1] + y[5,i-1]*dt
  y[3,i] = y[3,i-1] + dt * (A*a*(sigm(nu_max,v0,r,(y[1,i-1]-y[2,i-1]))) - (2*a*y[3,i-1]) - (a**(2)*y[0,i-1]))
  y[4,i] = y[4,i-1] + dt * (A*a*(noise[i-1] + (C2*sigm(nu_max,v0,r,(C1*y[0,i-1])))) - (2*a*y[4,i-1]) - (a**(2)*y[1,i-1]))
  y[5,i] = y[5,i-1] + dt * (B*b*(C4*sigm(nu_max,v0,r,(C3*y[0,i-1]))) - (2*b*y[5,i-1]) - (b**(2)*y[2,i-1]))

output = y[1,:]-y[2,:]
X = signal.resample(output, 10000)
freqs_Jansen,ps_vPN_Jansen = welch(X,fs=100, noverlap = 125, nperseg=1000)


# Estimating 1/f components using line fitting 
X = freqs_Jansen
Y = ps_vPN_Jansen
# 1/f pre peak
pre_X = X[4:40]
pre_Y = Y[4:40]

# 1/f post peak
post_X = X[110:]
post_Y = Y[110:]

# full 1/f
full_X = X[10:]
full_Y = Y[10:]

# Pre peak 1/f line fitting
p1=np.polyfit(np.log(pre_X),np.log(pre_Y),1);
pre_y =np.polyval(p1,np.log(pre_X));

# Post peak 1/f line fitting
p2=np.polyfit(np.log(post_X),np.log(post_Y),1);
post_y =np.polyval(p2,np.log(post_X));

# Full 1/f line fitting
p3=np.polyfit(np.log(full_X),np.log(full_Y),1);
full_y =np.polyval(p3,np.log(full_X));

# Plotting
plt.plot(X,Y)
plt.plot(pre_X,np.exp(pre_y), color="red")
plt.plot(post_X,np.exp(post_y), color="red")
plt.plot(full_X,np.exp(full_y), color="orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency")
plt.ylabel("Power")


# Slope and intercept values
slope_pre, intercept = np.polyfit(np.log(pre_X),np.log(pre_Y),1);
print(slope_pre)
slope_post, intercept=np.polyfit(np.log(post_X),np.log(post_Y),1);
print(slope_post)
full_slope, intercept=np.polyfit(np.log(full_X),np.log(full_Y),1);
print(full_slope)

# Estimating 1/f with FOOOF toolbox
fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='knee')
fm.report(X,Y , [1, 50], plt_log=True)


# Moran-David-Friston

# Sigmoid function
def sigm(x):
  return (1/(1+math.exp(-rho_1*(x - rho_2)))) - (1/(1+math.exp(rho_1*rho_2)))



# Parameter settings
H_e = 10                # mV
H_i = 22                # mV
kappa_e = (1/4) * 1000  # s
kappa_i = (1/16) * 1000 # s
gamma_1 = 128           # na
gamma_2 =128            # na
gamma_3 = 64            # na
gamma_4 = 64            # na
gamma_5 = 1             # na
rho_1 = 2               # na
rho_2 = 1               # na
a = 0     

# Simulation settings
start = 0.0
stim_time = 100
dt = 1e-4
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)

#Initialize input and output
x = np.zeros((12,vec_len))
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

output_Moran=x[1,:]-x[2,:]
X_Moran = signal.resample(output_Moran, 10000)
freqs_Moran,ps_vPN_Moran = welch(X_Moran,fs=100, noverlap = 125, nperseg=1000)


# Estimate 1/f components using line fitting
X = freqs_Moran
Y = ps_vPN_Moran

# Pre peak 1/f
pre_X = X[4:40]
pre_Y = Y[4:40]
# Post peak 1/f
post_X = X[110:]
post_Y = Y[110:]
# Full 1/f
full_X = X[10:]
full_Y = Y[10:]

# Pre peak 1/f line fitting
p1=np.polyfit(np.log(pre_X),np.log(pre_Y),1);
pre_y =np.polyval(p1,np.log(pre_X));

# Post peak 1/f line fitting
p2=np.polyfit(np.log(post_X),np.log(post_Y),1);
post_y =np.polyval(p2,np.log(post_X));

# Full 1/f line fitting
p3=np.polyfit(np.log(full_X),np.log(full_Y),1);
full_y =np.polyval(p3,np.log(full_X));

# Plotting
plt.plot(X,Y)
plt.plot(pre_X,np.exp(pre_y), color="red")
plt.plot(post_X,np.exp(post_y), color="red")
plt.plot(full_X,np.exp(full_y), color="orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency")
plt.ylabel("Power")


# Slope and intercept values
slope_pre, intercept = np.polyfit(np.log(pre_X),np.log(pre_Y),1);
print(slope_pre)
slope_post, intercept=np.polyfit(np.log(post_X),np.log(post_Y),1);
print(slope_post)
full_slope, intercept=np.polyfit(np.log(full_X),np.log(full_Y),1);
print(full_slope)


# Estimating 1/f components using FOOOF toolbox
fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='knee')
fm.report(X,Y , [1, 50], plt_log=True)


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
    
# Sigmoid function for excitatory and inhibitory populations respectively    
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


# Simulation settings
sim_time= 100        
dt = 1e-4
steps=  1/dt           
white_noise=1         

# Initialization
p.v_e_equil = p.h_e_r
p.v_i_equil = p.h_i_r
h = 1000/steps        
T = sim_time*10**3      
N = T/h-1
X = np.zeros((18,int(N)))
X[0,0] = p.v_e_equil
X[1,0] = p.v_i_equil    
X[2,0] = math.exp(1)/p.gamma_ee*p.Gamma_ee*(p.N_ee_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ee)   
X[4,0] = math.exp(1)/p.gamma_ei*p.Gamma_ei*(p.N_ei_b*S_e(p,p.v_e_equil) + 0*S_e(p,p.v_e_equil)+p.p_ei)
X[6,0] = math.exp(1)/p.gamma_ie*p.Gamma_ie*(p.N_ie_b*S_i(p,p.v_i_equil))
X[8,0] = math.exp(1)/p.gamma_ii*p.Gamma_ii*(p.N_ii_b*S_i(p,p.v_i_equil))

for n in range (0,int(N-1)):
    noise = np.zeros((18,1))
    if (white_noise==1):
      noise[3]= p.gamma_ee*math.exp(p.gamma_ee/p.gamma_ee)*p.Gamma_ee*p.p_ee_sd*np.random.randn(1,1)
    X[:,n+1] = X[:,n] + ((h*dynamics(p,X[:,n])+math.sqrt(h)*noise).flatten())
EEG=-X[0,:]

new_EEG =signal.resample(EEG,10000)
freqs_Liley,ps_vPN_Liley = welch(new_EEG,fs=100, noverlap = 125, nperseg=1000)


# Estimating 1/f with line fitting
X = freqs_Liley
Y = ps_vPN_Liley

# Pre peak 1/f
pre_X = X[10:90]
pre_Y = Y[10:90]
# Post peak 1/f
post_X = X[140:]
post_Y = Y[140:]
# Full 1/f
full_X = X[10:]
full_Y = Y[10:]

# Pre peak 1/f line fitting
p1=np.polyfit(np.log(pre_X),np.log(pre_Y),1);
pre_y =np.polyval(p1,np.log(pre_X));

# Post peak 1/f line fitting
p2=np.polyfit(np.log(post_X),np.log(post_Y),1);
post_y =np.polyval(p2,np.log(post_X));

# Full 1/f line fitting
p3=np.polyfit(np.log(full_X),np.log(full_Y),1);
full_y =np.polyval(p3,np.log(full_X));

# PLotting
plt.plot(X,Y)
plt.plot(pre_X,np.exp(pre_y), color="red")
plt.plot(post_X,np.exp(post_y), color="red")
plt.plot(full_X,np.exp(full_y), color="orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency")
plt.ylabel("Power")


# Slope and intercept values
slope_pre, intercept = np.polyfit(np.log(pre_X),np.log(pre_Y),1);
print(slope_pre)
slope_post, intercept=np.polyfit(np.log(post_X),np.log(post_Y),1);
print(slope_post)
full_slope, intercept=np.polyfit(np.log(full_X),np.log(full_Y),1);
print(full_slope)


# Estimating 1/f components using FOOOF toolbox
fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='knee')
fm.report(X,Y , [1, 50], plt_log=True)


# Robinson-Rennie-Wright

# Sigmoid function
def sig(v,Qmax,theta,sigma):
 # sigmoid for voltage-rate relationship
 firing_rate = Qmax / (1 + math.exp(-(v-theta) / sigma))
 return firing_rate

def sigr(v,Qmax_r,modtheta,sigma_r):
 # sigmoid for voltage-rate relationship in reticular nucleus
 firing_rate = Qmax_r / (1 + math.exp(-(v-modtheta) / sigma_r));
 return firing_rate

# Parameter settings
gamma = 116          # s^-1
t0 = 80e-3           # s
Qmax = 340           # s^-1
theta = 12.92e-3     # V
sigma = 3.8e-3       # V
Qmax_r = 100         # s^-1
sigma_r = 3.8e-3     # V
phin_0 = 1.0         # s^-1

nu_ee = 0.00303      # Vs
nu_ei = -0.006       # Vs
nu_es = 0.00206      # Vs
nu_ie = 0.00303      # Vs
nu_ii = -0.006       # Vs
nu_is = 0.00206      # Vs
nu_re = 0.00033      # Vs
nu_rs =  0.00003     # Vs
nu_se = 0.00218      # Vs
nu_sr = -0.00083     # Vs
nu_sn = 0.00098      # Vs
alpha = 83.33        # s^-1
beta = 769.23        # s^-1
phin = 5e-4          # s^-1

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

# Initialization
phie = np.zeros(vec_len)
Ve = np.zeros(vec_len)
Vr = np.zeros(vec_len)
Vs = np.zeros(vec_len)
phiedot = np.zeros(vec_len)
Vedot = np.zeros(vec_len)
Vrdot = np.zeros(vec_len)
Vsdot = np.zeros(vec_len)

phie[0:(k0+1)] = 3.175
Ve[0:(k0+1)]  = 0.0006344; 
Vr[0:(k0+1)]  = 0.005676;
Vs[0:(k0+1)]  = -0.003234;

# Euler-Maruyama method to solve differential equations
for i in range ((k0+1),(int(stop/dt))):
    Ve[i] = Ve[i-1] + Vedot[i-1]*dt
    Vedot[i] = Vedot[i-1] + dt *( alpha*beta * ( nu_ee*phie[i-1] + nu_ei*sig(Ve[i-1],Qmax,theta,sigma) + nu_es*sig(Vs[i-1-k0],Qmax,theta,sigma) - (1/alpha + 1/beta)*Vedot[i-1] - Ve[i-1] ) )
      
    Vr[i] = Vr[i-1] + Vrdot[i-1]*dt
    Vrdot[i] = Vrdot[i-1] + dt*( alpha*beta * ( nu_re*phie[i-1-k0] + nu_rs*sig(Vs[i-1],Qmax,theta,sigma) - (1/alpha + 1/beta)*Vrdot[i-1] - Vr[i-1] ))
    
    Vs[i] = Vs[i-1] + Vsdot[i-1]*dt
    Vsdot[i] = Vsdot[i-1] + dt*( alpha*beta * ( nu_se*phie[i-1-k0] + nu_sr*sigr(Vr[i-1],Qmax,theta,sigma_r) - (1/alpha + 1/beta)*Vsdot[i-1] - Vs[i-1] + nu_sn*phin_0 )) + noise[i-1]
    
    phie[i] = phie[i-1] + phiedot[i-1]*dt
    phiedot[i] = phiedot[i-1] + dt*( gamma**(2) * ( sig(Ve[i-1],Qmax,theta,sigma) - 2/gamma*phiedot[i-1] - phie[i-1] ))

phie = phie[(k0+1):len(phie)]
Ve = Ve[(k0+1):len(Ve)]
Vr = Vr[(k0+1):len(Vr)]
Vs = Vs[(k0+1):len(Vs)];
time_array = time_array[(k0+1):len(time_array)];

X=signal.resample(phie,10000)
output_Zhao = phie;
freqs_Robinson,ps_vPN_Robinson = welch(X[1000:],fs=100, noverlap = 125, nperseg=1000)


# Estimating 1/f components using line fitting
X = freqs_Robinson
Y = ps_vPN_Robinson
# Pre peak 1/f
pre_X = X[10:60]
pre_Y = Y[10:60]
# Post peak 1/f
post_X = X[60:]
post_Y = Y[60:]
# Full 1/f
full_X = X[10:]
full_Y = Y[10:]

# Pre peak 1/f line fitting
p1=np.polyfit(np.log(pre_X),np.log(pre_Y),1);
pre_y =np.polyval(p1,np.log(pre_X));
# Post peak 1/f line fitting
p2=np.polyfit(np.log(post_X),np.log(post_Y),1);
post_y =np.polyval(p2,np.log(post_X));
# Full 1/f line fitting
p3=np.polyfit(np.log(full_X),np.log(full_Y),1);
full_y =np.polyval(p3,np.log(full_X));

# Plotting
plt.plot(X,Y)
plt.plot(pre_X,np.exp(pre_y), color="red")
plt.plot(post_X,np.exp(post_y), color="red")
plt.plot(full_X,np.exp(full_y), color="orange")
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.xscale("log")
plt.yscale("log")


# Slope and intercept values
slope_pre, intercept = np.polyfit(np.log(pre_X),np.log(pre_Y),1);
print(slope_pre)
slope_post, intercept=np.polyfit(np.log(post_X),np.log(post_Y),1);
print(slope_post)
full_slope, intercept=np.polyfit(np.log(full_X),np.log(full_Y),1);
print(full_slope)

# Estimating 1/f components using FOOOF toolbox
fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='knee')
fm.report(X,Y , [1, 50], plt_log=True)