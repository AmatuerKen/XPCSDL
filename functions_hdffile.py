import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def print_hdf_structure(g, indent=0):
    """Recursively prints the file structure."""
    for key in g.keys():
        item = g[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"[Group]  {key}/")
            print_hdf_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * indent + f"[Dataset] {key}  shape={item.shape} dtype={item.dtype}")

def load_g2_tau(f, qindex):
    entry = "exchange"                 # group name
    dataset_name = "norm-0-g2"                # dataset name inside the group

    g2 = np.array(f[f"{entry}/{dataset_name}"])[:, qindex]

    g2err = np.array(f[f"{entry}/norm-0-stderr"])[:, qindex]
    baseline = np.array(f[f"{entry}/baselineFIT1"])[0, qindex]

    t0 = np.array(f["measurement/instrument/detector/exposure_period"])[0]
    tau = np.array(f[f"{entry}/tau"]).reshape(-1, )*t0
    q = np.array(f["xpcs/dqlist"])[0,qindex]*10
    return g2, g2err, baseline, tau, q



def fun_g2_diff(A, D, q, tau):
    return A * np.exp(-2*D*q**2*tau)

def plot_fit_diffusion_procedure(g2_quie, g2err_quie, baseline_quie, tau_quie, q_quie, H):
    
    def g2_fit_step1(tau, A, D, q):
        return fun_g2_diff(A, D, q, tau)

    # Initial guesses for A and D

    p0 = [0.1, 0.001]    # [A, D]

    # Fit (A, D) using the measured g2, tau, g2err
    sigma = g2err_quie if g2err_quie is not None else None

    popt, pcov = curve_fit(
        lambda tau, A, D: g2_fit_step1(tau, A, D, q_quie),
        tau_quie,
        g2_quie - baseline_quie,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True if sigma is not None else False
    )

    A_fit, D_fit = popt
    A_err, D_err = np.sqrt(np.diag(pcov))

    print(f"A = {A_fit:.5g} ± {A_err:.5g}")
    print(f"D = {D_fit:.5g} ± {D_err:.5g}")

    tau_list = np.logspace(-3.5, 1, 50)
    g2_diff_fit = fun_g2_diff(A_fit, D_fit, q_quie, tau_list)

    fig = plt.figure(figsize = (4, 3))
    plt.errorbar(tau_quie, g2_quie - baseline_quie, yerr = g2err_quie, label = f"{q_quie}", fmt = 'o')
    plt.plot(tau_list, g2_diff_fit)
    plt.xscale('log')
    plt.xlabel(r'$\tau$', fontsize = 16)
    plt.ylabel(r'$g_2$($\tau$) - 1', fontsize = 14)
    plt.legend(loc = 'lower left')
    plt.show()

    return A_fit, D_fit, A_err, D_err


def fun_g2_shear(A, gammadot, D, q, H, tau):
    a = q * gammadot * H * tau / 2
    #return A * np.exp(-2*D*q**2*tau) * (np.sin(a)/(a))**2
    return A * np.exp(-2*D*q**2*tau * (1 + (gammadot * tau)**2)) * (np.sin(a)/(a))**2

def plot_fit_shear_procedure(g2, g2err, baseline, tau, q, H, indexcut, D_fixed, A_fixed):

    def g2_fit_step2(tau, A, gammadot, q, H, D_fixed):
        return fun_g2_shear(A, gammadot, D_fixed, q, H, tau)

    # Initial guesses for Step 2 fit (use Step 1 A_fit)
    p0_step2 = [A_fixed, 0.01]   # [A, gammadot]

    index = indexcut

    sigma = g2err[0:index] if g2err is not None else None

    popt2, pcov2 = curve_fit(
        lambda tau, A, gammadot: g2_fit_step2(tau, A, gammadot, q, H, D_fixed),
        tau[0:index],
        g2[0:index]-baseline,
        p0=p0_step2,
        sigma=sigma,
        absolute_sigma=True if sigma is not None else False
    )

    A2_fit, gammadot_fit = popt2
    A2_err, gammadot_err = np.sqrt(np.diag(pcov2))

    print("===== Fit Results (D fixed) =====")
    print(f"A        = {A2_fit:.5g} ± {A2_err:.5g}")
    print(f"gammadot = {gammadot_fit:.5g} ± {gammadot_err:.5g}")  

    tau_list = np.logspace(-3.5, 1, 100)
    g2_shear_fit = fun_g2(A2_fit, gammadot_fit, D_fixed, q, H, tau_list)

    fig = plt.figure(figsize = (4, 3))
    plt.errorbar(tau, g2 - baseline, yerr = g2err, label = r'$\dot{\gamma}$ = 0.1/s', fmt = 'o')
    plt.plot(tau_list, g2_shear_fit)
    plt.xscale('log')
    plt.xlabel(r'$\tau$', fontsize = 16)
    plt.ylabel(r'$g_2$($\tau$) - 1', fontsize = 14)
    plt.legend()
    plt.show()

    return A2_fit, gammadot_fit, A2_err, gammadot_err
    