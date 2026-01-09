import numpy as np
from scipy.integrate import quad

class BoltzmannSolver:
    """
    A simplified BTE solver using the Callaway Model for Silicon Nanowires.
    """
    
    def __init__(self):
        # Constants for Silicon
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.hbar = 1.0545718e-34 # Reduced Planck constant
        self.v_s = 8433.0         # Sound velocity (m/s)
        self.Theta_D = 645.0      # Debye temperature (K)
        self.A = 1.32e-45         # Impurity scattering constant
        self.B = 1.73e-24         # Umklapp scattering constant

    def run_simulation(self, params: dict) -> dict:
        """
        Calculates Thermal Conductivity (k) given Temperature (T) and Diameter (D).
        
        Expected params:
        - T: Temperature in Kelvin (float)
        - D: Diameter in nanometers (float)
        """
        try:
            T = float(params.get("T", 300))
            D_nm = float(params.get("D", 100))
            D = D_nm * 1e-9 # Convert nm to meters

            # Define the scattering rate term tau^-1
            # Rate = Boundary + Impurity + Umklapp
            def scattering_rate(x):
                omega = x * (self.k_B * T) / self.hbar
                
                # 1. Boundary Scattering (The limit for Nanowires)
                tau_b_inv = self.v_s / D 
                
                # 2. Impurity Scattering (Rayleigh)
                tau_i_inv = self.A * (omega ** 4)
                
                # 3. Umklapp Scattering (Phonon-Phonon)
                tau_u_inv = self.B * (omega ** 2) * T * np.exp(-self.Theta_D / (3 * T))
                
                total_rate_inv = tau_b_inv + tau_i_inv + tau_u_inv
                return 1.0 / total_rate_inv

            # Define the integrand for Thermal Conductivity (Callaway Model)
            def integrand(x):
                tau = scattering_rate(x)
                coeff = (x**4 * np.exp(x)) / ((np.exp(x) - 1)**2)
                return coeff * tau

            # Integrate from 0 to Debye limit (Theta_D/T)
            x_max = self.Theta_D / T
            constant_prefix = (self.k_B / (2 * np.pi**2 * self.v_s)) * ((self.k_B * T / self.hbar)**3)
            
            integral_val, _ = quad(integrand, 0.1, x_max) # Start 0.1 to avoid div by zero
            k_value = constant_prefix * integral_val

            return {
                "status": "success",
                "k_wmk": round(k_value, 2),
                "T_K": T,
                "D_nm": D_nm,
                "note": "Calculated using Callaway BTE Model"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}