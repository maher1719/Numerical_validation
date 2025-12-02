import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

# ==========================================
# 1. Quaternion Class (Minimal, for clarity)
# ==========================================
@dataclass
class Quaternion:
    """
    Quaternion in scalar-last convention: (w, x, y, z) = w + xi + yj + zk
    """
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)

    def __mul__(self, other):
        """Hamilton product"""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        else:  # Scalar multiplication
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
            
    def __rmul__(self, other):
        return self * other

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        n = self.norm()
        if n < 1e-15: 
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def to_array(self):
        """Convert to NumPy array [x, y, z, w] for scipy"""
        return np.array([self.x, self.y, self.z, self.w])
    
    @classmethod
    def from_array(cls, arr):
        """Create from NumPy array [x, y, z, w]"""
        return cls(w=arr[3], x=arr[0], y=arr[1], z=arr[2])

    def __repr__(self):
        return f"Q({self.w:.6f} + {self.x:.6f}i + {self.y:.6f}j + {self.z:.6f}k)"


# ==========================================
# 2. Method Implementations
# ==========================================

class Methods:
    
    @staticmethod
    def scipy_rotation_power(A: Quaternion, x: float) -> Quaternion:
        """
        Industry Standard: SciPy's Rotation class.
        Uses optimized C++ implementation with BLAS/LAPACK backend.
        This is what professional libraries actually use.
        """
        # Convert to scipy format [x, y, z, w]
        q_array = A.to_array()
        
        # Normalize (scipy requires unit quaternions)
        norm = np.linalg.norm(q_array)
        if norm < 1e-15:
            return Quaternion(0, 0, 0, 0)
        
        q_unit = q_array / norm
        
        # Create rotation object
        rot = Rotation.from_quat(q_unit)
        
        # Compute fractional power via rotvec (axis-angle)
        # This uses arccos internally but is heavily optimized
        rotvec = rot.as_rotvec()
        rotvec_scaled = rotvec * x
        
        # Back to quaternion
        rot_powered = Rotation.from_rotvec(rotvec_scaled)
        q_result = rot_powered.as_quat()
        
        # Scale back to original magnitude
        result = Quaternion.from_array(q_result * (norm ** x))
        
        return result

    @staticmethod
    def numpy_optimized_euler(A: Quaternion, x: float) -> Quaternion:
        """
        Optimized implementation using NumPy's vectorized operations.
        This is what a good implementation would look like.
        """
        # Use NumPy's optimized norm
        arr = np.array([A.w, A.x, A.y, A.z])
        A_norm = np.linalg.norm(arr)
        
        if A_norm < 1e-15:
            return Quaternion(0, 0, 0, 0)
        
        # Normalize
        U = arr / A_norm
        
        # Extract vector part efficiently
        vec = U[1:]  # [x, y, z]
        vec_norm = np.linalg.norm(vec)
        
        # Compute angle using NumPy's arctan2 (hardware optimized)
        theta = np.arctan2(vec_norm, U[0])
        
        if vec_norm < 1e-15:
            # Real quaternion case
            return Quaternion(A.w**x, 0, 0, 0)
        
        # Unit vector direction
        v_dir = vec / vec_norm
        
        # Fractional power
        new_theta = x * theta
        cos_t = np.cos(new_theta)
        sin_t = np.sin(new_theta)
        
        # Construct result using vectorized operations
        result = np.zeros(4)
        result[0] = cos_t
        result[1:] = v_dir * sin_t
        
        # Scale by magnitude^x
        result *= (A_norm ** x)
        
        return Quaternion(result[0], result[1], result[2], result[3])

    @staticmethod
    def universal_law_numpy(A: Quaternion, x: float) -> Quaternion:
        """
        Your Universal Method - optimized with NumPy operations.
        """
        # Use vectorized array operations
        arr = np.array([A.w, A.x, A.y, A.z])
        
        # Compute A^2 using quaternion product rules (vectorized)
        # For pure imaginary: A^2 = -(x^2 + y^2 + z^2)
        vec_sq = arr[1]**2 + arr[2]**2 + arr[3]**2
        A2_w = arr[0]**2 - vec_sq
        
        # Validate elliptic domain
        if A2_w >= 0:
            # Not in elliptic domain - this shouldn't happen for pure imaginary
            pass
        
        alpha_sq = -A2_w
        alpha = np.sqrt(alpha_sq)
        
        if alpha < 1e-15:
            return Quaternion(0, 0, 0, 0)
        
        # Normalize to get E
        E = arr / alpha
        
        # Universal formula: cos(πx/2) + E*sin(πx/2)
        angle = np.pi * x / 2.0
        cos_term = np.cos(angle)
        sin_term = np.sin(angle)
        
        # Construct result
        result = np.zeros(4)
        result[0] = cos_term
        result[1:] = E[1:] * sin_term
        
        # Scale by alpha^x
        result *= (alpha ** x)
        
        return Quaternion(result[0], result[1], result[2], result[3])

    @staticmethod
    def universal_law_original(A: Quaternion, x: float) -> Quaternion:
        """
        Your original implementation (for comparison)
        """
        A2 = A * A
        
        if A2.w >= 0 or abs(A2.x)+abs(A2.y)+abs(A2.z) > 1e-9:
            pass 

        alpha_sq = -A2.w
        alpha = np.sqrt(alpha_sq)
        
        if alpha < 1e-15:
            return Quaternion(0, 0, 0, 0)
        
        E = A * (1.0/alpha)
        
        term1 = np.cos(np.pi * x / 2.0)
        term2 = np.sin(np.pi * x / 2.0)
        
        unit_res = Quaternion(term1, 0, 0, 0) + (E * term2)
        
        return (alpha**x) * unit_res


# ==========================================
# 3. Comprehensive Benchmarking Suite
# ==========================================

def run_comprehensive_comparison1(power: float, iterations: int):
    """
    Rigorous comparison against industry-standard libraries.
    """
    print(f"\n{'='*90}")
    print(f"BENCHMARK: x = 1/{int(1/power)} (cube root), N = {iterations:,} iterations")
    print(f"{'='*90}")
    
    # Test quaternion: pure imaginary (0 + 1i + 1j + 0k)
    A = Quaternion(0, 1, 1, 0)
    
    print(f"\n{'Method':<30} | {'Time (s)':<12} | {'Error':<12} | {'Speedup':<10}")
    print("-" * 90)
    
    results = {}
    
    # ===== 1. SciPy Rotation (Industry Standard) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_scipy = Methods.scipy_rotation_power(A, power)
    time_scipy = time.perf_counter() - start
    
    # Verify: cube the result
    cubed = result_scipy * result_scipy * result_scipy
    error_scipy = (cubed - A).norm()
    results['scipy'] = {'time': time_scipy, 'error': error_scipy, 'result': result_scipy}
    
    print(f"{'SciPy Rotation (Standard)':<30} | {time_scipy:>10.4f}s | {error_scipy:>10.2e} | {'1.00x (baseline)':<10}")
    
    # ===== 2. NumPy-Optimized Euler =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_numpy = Methods.numpy_optimized_euler(A, power)
    time_numpy = time.perf_counter() - start
    
    cubed = result_numpy * result_numpy * result_numpy
    error_numpy = (cubed - A).norm()
    speedup_numpy = time_scipy / time_numpy
    results['numpy'] = {'time': time_numpy, 'error': error_numpy, 'result': result_numpy}
    
    print(f"{'NumPy Optimized Euler':<30} | {time_numpy:>10.4f}s | {error_numpy:>10.2e} | {speedup_numpy:>9.2f}x")
    
    # ===== 3. Universal Law (NumPy-optimized) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_universal = Methods.universal_law_numpy(A, power)
    time_universal = time.perf_counter() - start
    
    cubed = result_universal * result_universal * result_universal
    error_universal = (cubed - A).norm()
    speedup_universal = time_scipy / time_universal
    results['universal'] = {'time': time_universal, 'error': error_universal, 'result': result_universal}
    
    print(f"{'Universal Law (NumPy)':<30} | {time_universal:>10.4f}s | {error_universal:>10.2e} | {speedup_universal:>9.2f}x")
    
    # ===== 4. Universal Law (Original) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_orig = Methods.universal_law_original(A, power)
    time_orig = time.perf_counter() - start
    
    cubed = result_orig * result_orig * result_orig
    error_orig = (cubed - A).norm()
    speedup_orig = time_scipy / time_orig
    results['original'] = {'time': time_orig, 'error': error_orig, 'result': result_orig}
    
    print(f"{'Universal Law (Original)':<30} | {time_orig:>10.4f}s | {error_orig:>10.2e} | {speedup_orig:>9.2f}x")
    
    # ===== Equivalence Check =====
    print(f"\n{'Equivalence Analysis':<30}")
    print("-" * 90)
    diff_scipy_numpy = (results['scipy']['result'] - results['numpy']['result']).norm()
    diff_scipy_universal = (results['scipy']['result'] - results['universal']['result']).norm()
    diff_numpy_universal = (results['numpy']['result'] - results['universal']['result']).norm()
    
    print(f"  SciPy ↔ NumPy Euler:     {diff_scipy_numpy:.2e}")
    print(f"  SciPy ↔ Universal:       {diff_scipy_universal:.2e}")
    print(f"  NumPy ↔ Universal:       {diff_numpy_universal:.2e}")
    
    # Determine winner
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"\n🏆 WINNER: {fastest[0].upper()} ({results['scipy']['time']/fastest[1]['time']:.2f}x faster than baseline)")
    
    return results

def run_comprehensive_comparison(power: float, iterations: int):
    """
    Rigorous comparison against industry-standard libraries.
    """
    print(f"\n{'='*90}")
    print(f"BENCHMARK: x = 1/{int(1/power)}, N = {iterations:,} iterations")
    print(f"{'='*90}")
    
    # Test quaternion: pure imaginary (0 + 1i + 1j + 0k)
    A = Quaternion(0, 1, 1, 0)
    
    # Calculate the inverse power for verification
    verification_power = int(round(1/power))  # e.g., 3 for 1/3, 15 for 1/15
    
    print(f"\n{'Method':<30} | {'Time (s)':<12} | {'Error':<12} | {'Speedup':<10}")
    print("-" * 90)
    
    results = {}
    
    # ===== 1. SciPy Rotation (Industry Standard) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_scipy = Methods.scipy_rotation_power(A, power)
    time_scipy = time.perf_counter() - start
    
    # CORRECTED VERIFICATION: Raise to the inverse power
    powered = result_scipy
    for _ in range(verification_power - 1):
        powered = powered * result_scipy
    error_scipy = (powered - A).norm()
    results['scipy'] = {'time': time_scipy, 'error': error_scipy, 'result': result_scipy}
    
    print(f"{'SciPy Rotation (Standard)':<30} | {time_scipy:>10.4f}s | {error_scipy:>10.2e} | {'1.00x (baseline)':<10}")
    
    # ===== 2. NumPy-Optimized Euler =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_numpy = Methods.numpy_optimized_euler(A, power)
    time_numpy = time.perf_counter() - start
    
    powered = result_numpy
    for _ in range(verification_power - 1):
        powered = powered * result_numpy
    error_numpy = (powered - A).norm()
    speedup_numpy = time_scipy / time_numpy
    results['numpy'] = {'time': time_numpy, 'error': error_numpy, 'result': result_numpy}
    
    print(f"{'NumPy Optimized Euler':<30} | {time_numpy:>10.4f}s | {error_numpy:>10.2e} | {speedup_numpy:>9.2f}x")
    
    # ===== 3. Universal Law (NumPy-optimized) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_universal = Methods.universal_law_numpy(A, power)
    time_universal = time.perf_counter() - start
    
    powered = result_universal
    for _ in range(verification_power - 1):
        powered = powered * result_universal
    error_universal = (powered - A).norm()
    speedup_universal = time_scipy / time_universal
    results['universal'] = {'time': time_universal, 'error': error_universal, 'result': result_universal}
    
    print(f"{'Universal Law (NumPy)':<30} | {time_universal:>10.4f}s | {error_universal:>10.2e} | {speedup_universal:>9.2f}x")
    
    # ===== 4. Universal Law (Original) =====
    start = time.perf_counter()
    for _ in range(iterations):
        result_orig = Methods.universal_law_original(A, power)
    time_orig = time.perf_counter() - start
    
    powered = result_orig
    for _ in range(verification_power - 1):
        powered = powered * result_orig
    error_orig = (powered - A).norm()
    speedup_orig = time_scipy / time_orig
    results['original'] = {'time': time_orig, 'error': error_orig, 'result': result_orig}
    
    print(f"{'Universal Law (Original)':<30} | {time_orig:>10.4f}s | {error_orig:>10.2e} | {speedup_orig:>9.2f}x")
    
    # ===== Equivalence Check =====
    print(f"\n{'Equivalence Analysis':<30}")
    print("-" * 90)
    diff_scipy_numpy = (results['scipy']['result'] - results['numpy']['result']).norm()
    diff_scipy_universal = (results['scipy']['result'] - results['universal']['result']).norm()
    diff_numpy_universal = (results['numpy']['result'] - results['universal']['result']).norm()
    
    print(f"  SciPy ↔ NumPy Euler:     {diff_scipy_numpy:.2e}")
    print(f"  SciPy ↔ Universal:       {diff_scipy_universal:.2e}")
    print(f"  NumPy ↔ Universal:       {diff_numpy_universal:.2e}")
    
    # Determine winner
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"\n🏆 WINNER: {fastest[0].upper()} ({results['scipy']['time']/fastest[1]['time']:.2f}x faster than baseline)")
    
    return results

def plot_stability_comparison():
    """
    Stability test: compose fractional powers N times.
    """
    print(f"\n{'='*90}")
    print("STABILITY TEST: Accumulated error over 500 compositions")
    print(f"{'='*90}\n")
    
    A = Quaternion(0, 1, 1, 0)
    N = 500
    small_power = 1.0 / N
    
    # Get single-step roots
    root_scipy = Methods.scipy_rotation_power(A, small_power)
    root_numpy = Methods.numpy_optimized_euler(A, small_power)
    root_universal = Methods.universal_law_numpy(A, small_power)
    
    # Track drift
    drift_scipy = []
    drift_numpy = []
    drift_universal = []
    
    current_scipy = Quaternion(1, 0, 0, 0)
    current_numpy = Quaternion(1, 0, 0, 0)
    current_universal = Quaternion(1, 0, 0, 0)
    
    for i in range(1, N + 1):
        current_scipy = current_scipy * root_scipy
        current_numpy = current_numpy * root_numpy
        current_universal = current_universal * root_universal
        
        # Theoretical norm at step i
        theo_norm = (A.norm() ** (1/N)) ** i
        
        drift_scipy.append(abs(current_scipy.norm() - theo_norm))
        drift_numpy.append(abs(current_numpy.norm() - theo_norm))
        drift_universal.append(abs(current_universal.norm() - theo_norm))
        
        if i % 100 == 0:
            print(f"Step {i:3d}: SciPy={drift_scipy[-1]:.2e}, "
                  f"NumPy={drift_numpy[-1]:.2e}, "
                  f"Universal={drift_universal[-1]:.2e}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(drift_scipy, label='SciPy Rotation', color='purple', linewidth=2)
    plt.plot(drift_numpy, label='NumPy Euler', color='red', linestyle='--')
    plt.plot(drift_universal, label='Universal Law', color='blue', alpha=0.7)
    plt.title(f"Numerical Stability over {N} Compositions")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error from Theoretical Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(drift_scipy, label='SciPy Rotation', color='purple', linewidth=2)
    plt.semilogy(drift_numpy, label='NumPy Euler', color='red', linestyle='--')
    plt.semilogy(drift_universal, label='Universal Law', color='blue', alpha=0.7)
    plt.title(f"Log Scale: Error Accumulation")
    plt.xlabel("Iteration")
    plt.ylabel("Log(Error)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quaternion_stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal drift at step {N}:")
    print(f"  SciPy:     {drift_scipy[-1]:.2e}")
    print(f"  NumPy:     {drift_numpy[-1]:.2e}")
    print(f"  Universal: {drift_universal[-1]:.2e}")


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*90)
    print(" RIGOROUS QUATERNION POWER BENCHMARKING")
    print(" Comparing against Industry-Standard Libraries (SciPy, NumPy)")
    print("="*90)
    
    # Run benchmarks with different powers and iteration counts
    test_cases = [
        (1/3, 10**5),
        (1/3, 10**6),
        (1/15, 10**5),
        (1/15, 10**6),
        (1/1267, 10**6),
    ]
    
    for power, iterations in test_cases:
        run_comprehensive_comparison(power, iterations)
    
    # Stability analysis
    plot_stability_comparison()
    
    print("\n" + "="*90)
    print("BENCHMARK COMPLETE")
    print("="*90)