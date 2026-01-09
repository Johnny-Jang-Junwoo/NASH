import numpy as np
from scipy.integrate import quad

class BoltzmannSolver:
    """Universal BTE solver that accepts material properties at runtime."""

    def __init__(self):
        self.k_B = 1.380649e-23
        self.hbar = 1.0545718e-34

    def run_simulation(self, params: dict) -> dict:
        """Calculate thermal conductivity using the Callaway BTE formulation."""

        try:
            T = float(params.get("T", 300))
            D_nm = float(params.get("D", 100))
            D = D_nm * 1e-9

            mat = params.get("material_props", {})
            v_s = float(mat.get("v_s", 8433.0))
            Theta_D = float(mat.get("Theta_D", 645.0))
            A = float(mat.get("A", 1.32e-45))
            B = float(mat.get("B", 1.73e-24))
            name = mat.get("name", "Silicon (Default)")

            def scattering_rate(x):
                omega = x * (self.k_B * T) / self.hbar
                tau_b_inv = v_s / D
                tau_i_inv = A * (omega**4)
                tau_u_inv = B * (omega**2) * T * np.exp(-Theta_D / (3 * T))
                return 1.0 / (tau_b_inv + tau_i_inv + tau_u_inv)

            def integrand(x):
                tau = scattering_rate(x)
                coeff = (x**4 * np.exp(x)) / ((np.exp(x) - 1) ** 2)
                return coeff * tau

            x_max = Theta_D / T
            constant_prefix = (self.k_B / (2 * np.pi**2 * v_s)) * ((self.k_B * T / self.hbar) ** 3)
            integral_val, _ = quad(integrand, 0.1, x_max)
            k_value = constant_prefix * integral_val

            return {
                "status": "success",
                "k_wmk": round(k_value, 2),
                "T_K": T,
                "D_nm": D_nm,
                "material": name,
                "note": f"Simulated {name} with v_s={v_s} m/s",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
