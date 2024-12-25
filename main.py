import argparse	
from argparse import RawTextHelpFormatter
import numpy as np   
import matplotlib.pyplot as plt  
import integrators # Stepper functions and integrator driver
import signals # Signal functions
from mpl_toolkits import mplot3d # Import toolkit for 3D plots
from systemparameters import SystemParameters
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

'''
Calculates RHS for Lorenz System sender

Sender is responsible for calling the receiver in the event receiver values are specified
'''
def dydx_sender(t, values, dx):
    # Retrieve constants (class variables of SystemParameters)
    sigma = SystemParameters.sigma
    b = SystemParameters.b
    r = SystemParameters.r

    # Unpackage values  (constant for case of sender and receiver)
    x = values[0]
    y = values[1]
    z = values[2]

    # Define RHS for sender system
    dydx = np.zeros(3)
    dydx[0] = sigma*(y-x)
    dydx[1] = r*x - y - x*z
    dydx[2] = x*y - b*z
    
    # Check if receiver values are set in this time step
    if (len(values) > 3):
        receiver = dydx_receiver(t, values, dx)
        dydx = np.append(dydx, receiver, axis=0)
    
    return dydx

'''Calculates RHS for Lorenz System receiver'''
def dydx_receiver(t, values, dx):
    # Retrieve constants (class variables of SystemParameters)
    sigma = SystemParameters.sigma
    b = SystemParameters.b
    r = SystemParameters.r

    # Unpackage values
    x = values[0] # Using x as the previous value and not the value calculated above
    u = values[3]
    v = values[4]
    w = values[5]

    # Define RHS for receiver system
    dydx = np.zeros(3)
    if (SystemParameters.perturbation is not None):
        # Add pertubration if present
        perturbation = SystemParameters.perturbation
        index = SystemParameters.perturbation_index
        x_perturbed = x + perturbation[index]
        
        dydx[0] = sigma*(v-u)
        dydx[1] = r*x_perturbed - v - x_perturbed*w
        dydx[2] = x_perturbed*v - b*w
    else:
        dydx[0] = sigma*(v-u)
        dydx[1] = r*x - v - x*w
        dydx[2] = x*v - b*w

    return dydx


''''Manually finding local maxima in plot of z'''
def find_max(z):
    z_left = z[:-2]
    z_center = z[1:-1]
    z_right = z[2:]
    is_max = (z_center > z_left) & (z_center > z_right)
    return z_center[is_max]


'''Create necessary plots and parameterize signals as needed for perturbation'''
def main():
    # Expect integrator and r value inputs
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper", type=str, default='euler', help="Stepping function")
    parser.add_argument("r", type=float, default=10, help="Steepness of temperature gradient")
    parser.add_argument("--receiver", action='store_true', help="Flag to include receiver ODEs in system") # Optional arguement for included additional ODEs
    parser.add_argument("--perturbation", type=str, default=None, help="Optional kind of perturbation to include in the system")  # Optional text argument
    args   = parser.parse_args()
    
    # Determine Stepper
    if (args.stepper == "euler"):
        fORD = integrators.euler
    elif (args.stepper == "rk2"):
        fORD = integrators.rk2
    elif (args.stepper == "rk4"):
        fORD = integrators.rk4
    elif (args.stepper == "rk45"):
        fORD = integrators.rk45
    elif (args.stepper == "backeuler"):
        fORD = integrators.backeuler
    else:
        raise Exception("invalid stepper %s" % (args.stepper))
    
    # Initialization (assuming Lorenz system is always an initial-value problem)
    nstep = 10000
    t0 = 0
    t1 = 100
    
    x0 = 10
    y0 = 10
    z0 = 10
    values0 = [x0, y0, z0]
    
    # Determine presence of receiver system (in addition to sender)
    if args.receiver:
        u0 = 100
        v0 = 3.7
        w0 = -5
        values0.extend([u0, v0, w0])
    
    values0 = np.array(values0)
    
    # Determine presence of perturbation (on the receiver system)
    if (args.perturbation == "binary"):
        SystemParameters.perturbation = signals.load_digital_signal(nstep+1, 1, 1)
    elif (args.perturbation == "recording"):
        SystemParameters.perturbation = signals.load_audio_signal(length=nstep+1)
    elif (args.perturbation == "constant"):
        SystemParameters.perturbation = signals.load_constant_signal(nstep+1, 1)

    # Needed constants (using standard values for the Lorenz System for sigma and and passing in steepness of temperature gradient r as arguement)
    sigma = 10
    b = 8.0/3
    
    SystemParameters.sigma = sigma
    SystemParameters.b = b
    SystemParameters.r = args.r

    # Set driver and RHS ODE definitions
    fINT = integrators.ode_ivp
    fRHS = dydx_sender

    # Solve for values for each of the dependent variables (X, Y, Z) and for independent variable (t)
    t, values, iterations = fINT(fRHS, fORD, t0, t1, values0, nstep)
    x = values[0]
    y = values[1]
    z = values[2]
    
    if (len(values) > 3):
        u = values[3]
        v = values[4]
        w = values[5]

    # Plots for all values against t
    plt.figure(num=1,figsize=(5,6),dpi=100,facecolor='white')
    
    # X vs. T
    plt.subplot(311)
    plt.plot(t, x, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('x vs. t')
    
    # Y vs. T
    plt.subplot(312)
    plt.plot(t, y, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('y vs. t')
    
    # Z vs. T
    plt.subplot(313)
    plt.plot(t, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('t')
    plt.ylabel('z(t)')
    plt.title('z vs. t')

    plt.tight_layout()
    plt.show()
    
    # Plots for all pairwise plots
    plt.figure(num=2,figsize=(5,6),dpi=100,facecolor='white')
    
    plt.subplot(311)
    plt.plot(x, y, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.title('y vs. x')
    
    plt.subplot(312)
    plt.plot(x, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('x(t)')
    plt.ylabel('z(t)')
    plt.title('z vs. x')
    
    plt.subplot(313)
    plt.plot(y, z, linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('y(t)')
    plt.ylabel('z(t)')
    plt.title('z vs. y')
    
    plt.tight_layout()
    plt.show()
    
    # Plot for 3D graph of all variables
    plt.figure(num=3,figsize=(5,6),dpi=100,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, color='black')
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title('3D plot of x(t), y(t), z(t)')
    plt.show()
    
    # FFT
    plt.figure(num=4,figsize=(5,6),dpi=100,facecolor='white')
    z_norm = z - np.mean(z)
    plt.plot(fftfreq(nstep, t[1]-t[0])[:nstep//2], 2.0/nstep * np.abs(fft(z_norm)[0:nstep//2]), linestyle='-', color='black', linewidth=1.0)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.title('FFT for z(t)')
    
    # Lorenz Map
    plt.figure(num=5,figsize=(6,8),dpi=100,facecolor='white')
    z_n = find_peaks(z)[0]
    plt.scatter(z[z_n[:-1]], z[z_n[1:]], color='black')
    plt.xlabel('z_{n} (Current Peak)')
    plt.ylabel('z_{n+1} (Next Peak)')
    plt.title('Lorenz Map')
    plt.show()
    
    # Plot differences in variables with receiver system is present
    if (len(values) > 3):
        plt.figure(num=6,figsize=(5,6),dpi=100,facecolor='white')
        
        # u-x vs. T
        plt.subplot(311)
        plt.plot(t, np.abs(u-x), linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('|u-x|')
        plt.title('Difference Between U and X')
        
        # v-y vs. T
        plt.subplot(312)
        plt.plot(t, np.abs(v-y), linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('|v-y|')
        plt.title('Difference Between V and Y')
        
        # w-z vs. T
        plt.subplot(313)
        plt.plot(t, np.abs(w-z), linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('|w-z|')
        plt.title('Difference Between W and Z')

        plt.tight_layout()
        plt.show()
        
    
    # Plot perturbation siganal if present
    if (SystemParameters.perturbation is not None):
        # Plot for original signal and recovered signal
        plt.figure(num=7,figsize=(10,10),dpi=100,facecolor='white')
        plt.subplot(221)
        plt.plot(t, SystemParameters.perturbation, linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('s(t)')
        plt.title('Original Signal vs. t')
        
        plt.subplot(223)
        x_perturbed = x + SystemParameters.perturbation
        recovered_signal = x_perturbed - u
        plt.plot(t, recovered_signal, linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('X(t) - u')
        plt.title('Recovered Signal vs. t')
        plt.show()
        
        # Log difference
        plt.figure(num=7,figsize=(5,8),dpi=100,facecolor='white')
        plt.plot(t, np.log10(np.abs(recovered_signal - SystemParameters.perturbation)), linestyle='-', color='black', linewidth=1.0)
        plt.xlabel('t')
        plt.ylabel('log Difference')
        plt.title('Log Difference in Signals')
        plt.show()
    

if __name__ == "__main__":
    main()