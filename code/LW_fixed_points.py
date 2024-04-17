import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from shapely.geometry import LineString

# Sigmoid and voltage potential functions
def sigm(v):
  spikerate = S_max/(1 + np.exp(-np.sqrt(2)*(v - mu)/sigma))
  return spikerate

def psi_ee(ve):
  return (h_ee_eq-ve)/(abs(h_ee_eq-h_e_r))

def psi_ie(ve): 
  return (h_ie_eq-ve)/(abs(h_ie_eq-h_e_r))

def psi_ei(ve): 
  return (h_ei_eq-ve)/(abs(h_ei_eq-h_i_r))

def psi_ii(ve): 
  return (h_ii_eq-ve)/(abs(h_ii_eq-h_i_r))

# Derivative of sigmoig
def diff_sigm_e(h_e):
  sprime = (S_max*math.sqrt(2)*math.exp((-math.sqrt(2)*(h_e-mu))/sigma)) / (sigma*(1 + math.exp((-math.sqrt(2)*(h_e-mu))/sigma)) **2)
  return sprime

def diff_sigm_i(h_i):
  sprime = (S_max*math.sqrt(2)*math.exp((-math.sqrt(2)*(h_i-mu))/sigma)) / (sigma*(1 + math.exp((-math.sqrt(2)*(h_i-mu))/sigma)) **2)
  return sprime


# Parameter settings
h_e_r = -70
h_i_r = -70
h_ee_eq = 45
h_ie_eq = -90
h_ei_eq = 45
h_ii_eq = -90
Gamma_e = 0.71
Gamma_i = 0.71
gamma_e = 0.3
gamma_i = 0.065
N_ee_b = 3000
N_ie_b =  200
N_ei_b = 2200
N_ii_b = 500
S_max = 0.5
mu = -50
sigma = 5
p_ee = 3.460
p_ei = 5.070
tau_e = 94
tau_i = 42

all_fixed_y0 = []
all_fixed_y1 = []

N_ie_b = np.arange(300,700, 1) #
N_ei_b = np.arange(2000, 4000,5) #
x = np.linspace(-100, 10, 1000)
x1 = np.linspace(-100, 10, 1000)
X, Y = np.meshgrid(x,x1)

# Determine the fixed points
for i in range(0, len(N_ei_b)):
    F1 = -X + (h_e_r + psi_ee(X) * ((Gamma_e/gamma_e)*np.exp(1)*(N_ee_b*sigm(X)+p_ee)) + psi_ie(X) * ((Gamma_i/gamma_i)*np.exp(1)*(N_ie_b[i]*sigm(Y))))
    F2 = - Y + (h_i_r + psi_ei(Y) * ((Gamma_e/gamma_e)*np.exp(1)*(N_ei_b[i]*sigm(X)+p_ei)) + psi_ii(Y) * ((Gamma_i/gamma_i)*np.exp(1)*(N_ii_b*sigm(Y))))
    contour1 = plt.contour(X,Y,F1,[0])
    contour2 = plt.contour(X,Y,F2,[0])

    v1 = contour1.collections[0].get_paths()[0].vertices
    v2 = contour2.collections[0].get_paths()[0].vertices
    ls1 = LineString(v1)
    ls2 = LineString(v2)
    points = ls1.intersection(ls2)
    xnew, ynew = points.x, points.y
    
    all_fixed_y0.append(xnew)
    all_fixed_y1.append(ynew)
#plt.plot(xnew, ynew, 'ro')


# Determine the stability of the fixed points
un = 0
stability = []
for i in range(0,len(all_fixed_y0)):
    # Jacobian matrix test
    J = np.zeros((10,10))
    ve_fixed_point = all_fixed_y0[i]
    vi_fixed_point = all_fixed_y1[i]
    # See appendix Hartoyo as the details
    I_ee_fixed = (Gamma_e/gamma_e)*math.exp(1)*(N_ee_b*sigm(ve_fixed_point)+p_ee)
    I_ei_fixed = (Gamma_e/gamma_e)*math.exp(1)*(N_ei_b[i]*sigm(ve_fixed_point)+p_ei)
    I_ie_fixed = (Gamma_i/gamma_i)*math.exp(1)*(N_ie_b[i]*sigm(vi_fixed_point))
    I_ii_fixed = (Gamma_i/gamma_i)*math.exp(1)*(N_ii_b*sigm(vi_fixed_point))

    J[0,0] = (1/tau_e)*(-1 - (I_ee_fixed/(abs(h_ee_eq-h_e_r)))  - (I_ie_fixed/(abs(h_ie_eq-h_e_r)))  )
    J[0,2] = (psi_ee(ve_fixed_point))/(tau_e)  #not sure if diff or normal sigmoid (considered a constant)
    J[0,6] = (psi_ie(ve_fixed_point))/(tau_e)
    J[1,1] = (1/tau_i)*(-1 - (I_ei_fixed/(abs(h_ei_eq-h_i_r)))  - (I_ii_fixed/(abs(h_ii_eq-h_i_r)))  )
    J[1,4] = (psi_ei(vi_fixed_point))/(tau_i)
    J[1,8] = (psi_ii(vi_fixed_point))/(tau_i)
    J[2,3] = 1 
    J[3,0] = Gamma_e*gamma_e*math.exp(1)*N_ee_b*diff_sigm_e(ve_fixed_point)
    J[3,2] = -gamma_e**2
    J[3,3] = -2*gamma_e
    J[4,5] = 1
    J[5,0] = Gamma_e*gamma_e*math.exp(1)*N_ei_b[i]*diff_sigm_e(ve_fixed_point)
    J[5,4] = -gamma_e**2
    J[5,5] = -2*gamma_e
    J[6,7] = 1
    J[7,1] = Gamma_i*gamma_i*math.exp(1)*N_ie_b[i]*diff_sigm_i(vi_fixed_point)
    J[7,6] = -gamma_i**2
    J[7,7] = -2*gamma_i
    J[8,9] = 1
    J[9,1] = Gamma_i*gamma_i*math.exp(1)*N_ii_b*diff_sigm_i(vi_fixed_point)
    J[9,8] = -gamma_i**2
    J[9,9] = -2*gamma_i

    evals = np.linalg.eigvals(J)
    evals
    stability_per = np.zeros(len(evals))
    for j in range(0,len(evals)):
      real_part = np.real(evals[j])
      if real_part > 0:
        un = 1
        stability_per[j] = un
      else:
        un = 0
        stability_per[j] = un
    if stability_per.any()==1:
      value = 1
    else:
      value = 0
    stability.append(value)



val = [2450, 2650, 2850, 3050, 3250]
pos = []
for i in val:
    pos.append(np.where(np.round(N_ei_b,3)==i)[0][0])
index_unstable = np.where(np.array(stability)==1)
array_unstable = np.array(index_unstable).flatten()

# Plotting
plt.figure(figsize=(4, 3), dpi =1000)
all_fixed_array = np.array(all_fixed_y0)
fig,ax1=plt.subplots(figsize=(4, 3), dpi =1000)
ax1.scatter(N_ei_b, all_fixed_y0, color=plt.cm.Set1(1), label='stable')
ax1.scatter(N_ei_b[array_unstable], all_fixed_array[array_unstable],color = plt.cm.Set1(0), label='unstable')
ax1.set_xlabel('$N_{ei}$')
ax2=plt.twiny(ax1)
ax2.scatter(N_ie_b, all_fixed_y0, color=plt.cm.Set1(1), alpha=0)
ax2.set_xlabel('$N_{ie}$')
markers = ["s", "v", "x", "*", "+"]
x = val
y = all_fixed_array[pos]
for xp, yp, m in zip(x, y, markers):
   ax1.scatter(xp, yp, marker=m, color="black",s=100)

ax1.set_ylabel("Fixed point ($V_{e}$: mV)")
ax1.legend()