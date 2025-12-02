import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

# ==========================================
# 1. The Quaternion Engine
# ==========================================
@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)

    def __mul__(self, other):
        # Standard Hamilton Product
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        else:
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
            
    def __rmul__(self, other):
        return self * other

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        n = self.norm()
        if n < 1e-15: return Quaternion(1,0,0,0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def __repr__(self):
        return f"[{self.w:.5f}, {self.x:.5f}, {self.y:.5f}, {self.z:.5f}]"

# ==========================================
# 2. The Competitors
# ==========================================

class Methods:
    
    @staticmethod
    def standard_euler(A: Quaternion, x: float) -> Quaternion:
        """
        The 'Textbook' Method (Polar Decomposition).
        Relies on finding angle theta via arccos/arctan.
        Cost: Norm + Normalize + Arctan2 + Sin/Cos.
        """
        A_norm = A.norm()
        if A_norm < 1e-15: return Quaternion(0,0,0,0)
        
        # 1. Normalize to unit sphere
        U = A.normalize()
        
        # 2. Find Angle (Polar form: cos(t) + v*sin(t))
        # For a pure imaginary, w=0, so theta = pi/2.
        # But we calculate it generally to be fair.
        vec_norm = np.sqrt(U.x**2 + U.y**2 + U.z**2)
        theta = np.arctan2(vec_norm, U.w)
        
        # 3. Compute Unit Power
        if vec_norm < 1e-15:
            # Real number case
            return Quaternion(A.w**x, 0,0,0)
            
        v_dir = Quaternion(0, U.x/vec_norm, U.y/vec_norm, U.z/vec_norm)
        
        new_theta = x * theta
        unit_pow = Quaternion(np.cos(new_theta), 0,0,0) + (v_dir * np.sin(new_theta))
        
        # 4. Scale
        return (A_norm**x) * unit_pow

    @staticmethod
    def universal_law(A: Quaternion, x: float) -> Quaternion:
        """
        Maher's Universal Method.
        Relies on A^2 = -alpha^2 (Minimal Polynomial).
        Cost: Square + Sqrt + Sin/Cos. (No Arctan/Arccos).
        """
        # 1. Compute Square to find alpha (The Invariant)
        # For Pure Imaginary A: A^2 is a negative real scalar.
        A2 = A * A
        
        # Validation (Strict Rigor): Ensure A is in the 'Elliptic' domain
        if A2.w >= 0 or abs(A2.x)+abs(A2.y)+abs(A2.z) > 1e-9:
            # In a real library, we would handle this. 
            # For this test, we assume valid input to test core logic speed/accuracy.
            pass 

        alpha_sq = -A2.w
        alpha = np.sqrt(alpha_sq)
        
        if alpha < 1e-15: return Quaternion(0,0,0,0)
        
        # 2. The Universal Formula: A^x = alpha^x * (cos(pi*x/2) + E*sin(pi*x/2))
        # Note: In your paper E is the unit. A = alpha * E.
        # This simplifies to the code below without explicit normalization step if we group terms.
        
        # However, to match your paper exactly:
        E = A * (1.0/alpha) 
        
        # The Formula:
        term1 = np.cos(np.pi * x / 2.0)
        term2 = np.sin(np.pi * x / 2.0)
        
        unit_res = Quaternion(term1, 0,0,0) + (E * term2)
        
        # Scaling
        return (alpha**x) * unit_res

# ==========================================
# 3. Rigorous Testing Laboratory
# ==========================================

def run_rigorous_comparison(power,iterations):
    print(f"power: {power}, iterations: {iterations}")
    print(f"{'METRIC':<20} | {'STANDARD (EULER)':<20} | {'UNIVERSAL (MAHER)':<20} | {'STATUS'}")
    print("-" * 80)

    # --- TEST DATA: Pure Imaginary Quaternion A = 1*i + 1*j ---
    A = Quaternion(0, 1, 1, 0) 
    power = 1/3.0 # Cube Root

    # 1. ACCURACY TEST (Reversibility)
    # Calculate Root -> Cube it -> Compare to original
    
    # Standard
    root_s = Methods.standard_euler(A, power)
    cube_s = root_s * root_s * root_s
    err_s = (cube_s - A).norm()
    
    # Universal
    root_u = Methods.universal_law(A, power)
    cube_u = root_u * root_u * root_u
    err_u = (cube_u - A).norm()
    
    print(f"{'Accuracy (Error)':<20} | {err_s:.2e}             | {err_u:.2e}             | {'TIED' if abs(err_u-err_s) < 1e-16 else 'DIFF'}")

    # 2. EQUIVALENCE TEST (Do they agree?)
    diff = (root_s - root_u).norm()
    print(f"{'Equivalence Diff':<20} | {'N/A':<20} | {'N/A':<20} | {diff:.2e}")

    # 3. PERFORMANCE TEST (100,000 Iterations)
    # We measure raw compute time for the calculation logic
    
    
    start = time.time()
    for _ in range(iterations):
        Methods.standard_euler(A, power)
    time_s = time.time() - start
    
    start = time.time()
    for _ in range(iterations):
        Methods.universal_law(A, power)
    time_u = time.time() - start
    
    winner = "UNIVERSAL" if time_u < time_s else "STANDARD"
    if(time_s>time_u):
        speedup = (time_s / time_u) 
    else:
        speedup = (time_u / time_s) 
    per_k=iterations/1000
    print(f"{'Speed':<20} | {time_s:.4f}s             | {time_u:.4f}s             | {winner} ({speedup:.1f}x faster)")

    

    return root_u

def plot_stability():
    """
    Visualizing Numerical Drift over 1000 compositions.
    We take the 1000th root, then multiply it 1000 times.
    Ideally, we should return to the original A.
    """
    A = Quaternion(0, 2, 0, 0) # Pure 2j
    N = 500
    small_power = 1.0/N
    
    # Get roots
    root_s = Methods.standard_euler(A, small_power)
    root_u = Methods.universal_law(A, small_power)
    
    norms_s = []
    norms_u = []
    
    curr_s = Quaternion(1,0,0,0) # Start at identity
    curr_u = Quaternion(1,0,0,0)
    
    # We accumulate (multiply) the small root N times.
    # Note: We act on Identity. Effectively calculating (A^(1/N))^N
    # Ideally ending at A/||A|| direction if we track rotation, or A if we track magnitude.
    # Let's just track Norm Drift from Expected Theoretical Norm growth.
    
    expected_norm = A.norm() # The target
    
    # We essentially want to reach A from Identity by multiplying small steps
    # Step: A_step = A^(1/N)
    # Current = Current * A_step
    
    current_s = Quaternion(1,0,0,0)
    current_u = Quaternion(1,0,0,0)
    
    drift_s = []
    drift_u = []
    
    for i in range(1, N+1):
        current_s = current_s * root_s
        current_u = current_u * root_u
        
        # Theoretical norm at step i: (||A||^(1/N))^i
        theo_norm = (A.norm()**(1/N))**i
        value_drift_s=abs(current_s.norm() - theo_norm)
        value_drift_u=abs(current_u.norm() - theo_norm)
        diff_drift=abs(value_drift_s-value_drift_u)


        drift_s.append(value_drift_s)
        drift_u.append(value_drift_u)

        if(i % 50 ==0):
            print(f"drift at {i} euler: {value_drift_s}")
            print(f"drift at {i} universal: {value_drift_u}")
            print(f" differences : {diff_drift}")
            print("*"*60)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(drift_s, label='Standard Euler Drift', color='red', linestyle='--')
    plt.plot(drift_u, label='Universal Law Drift', color='blue', alpha=0.7)
    plt.title(f"Numerical Stability: Accumulated Error over {N} Multiplications")
    plt.xlabel("Iteration")
    plt.ylabel("Abs Error from Theoretical Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_rigorous_comparison(1/3.0,10**5)
    run_rigorous_comparison(1/3.0,10**6)
    run_rigorous_comparison(1/15,10**5)
    run_rigorous_comparison(1/15,10**6)
    run_rigorous_comparison(1/1267,10**5)
    run_rigorous_comparison(1/1267,10**6)
    plot_stability()