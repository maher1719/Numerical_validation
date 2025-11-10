import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Quaternion:
    """Quaternion class with w + xi + yj + zk representation"""
    w: float
    x: float
    y: float
    z: float
    
    def __mul__(self, other):
        """Quaternion multiplication"""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        else:  # scalar multiplication
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
    
    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
    
    def __sub__(self, other):
        return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)
    
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def norm_sq(self):
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def normalize(self):
        n = self.norm()
        if n < 1e-15:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def conj(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def __repr__(self):
        return f"{self.w:.6f} + {self.x:.6f}i + {self.y:.6f}j + {self.z:.6f}k"
    
    def vector_part(self):
        return np.array([self.x, self.y, self.z])
    
    def vector_norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


class QuaternionMethods:
    """Comparison of different methods for computing quaternion fractional powers"""
    
    @staticmethod
    def universal_method(A: Quaternion, x: float) -> Quaternion:
        """
        Universal exponential: E^x = cos(πx/2) + E sin(πx/2)
        where E = A/√(-A²)
        """
        # Compute A²
        A_squared = A * A
        
        # Check if A² is negative (real and < 0)
        if abs(A_squared.x) > 1e-10 or abs(A_squared.y) > 1e-10 or abs(A_squared.z) > 1e-10:
            raise ValueError("A² must be real (negative scalar)")
        
        if A_squared.w >= 0:
            raise ValueError("A² must be negative")
        
        # Normalize: E = A/√(-A²)
        alpha = np.sqrt(-A_squared.w)
        E = Quaternion(A.w/alpha, A.x/alpha, A.y/alpha, A.z/alpha)
        
        # Apply universal formula
        theta = np.pi * x / 2
        result = Quaternion(np.cos(theta), 0, 0, 0) + E * np.sin(theta)
        
        return result, E, alpha
    
    @staticmethod
    def naive_euler(A: Quaternion, x: float) -> Quaternion:
        """
        Naive Euler: cos(πx/2) + A sin(πx/2)
        (No normalization - WRONG)
        """
        theta = np.pi * x / 2
        result = Quaternion(np.cos(theta), 0, 0, 0) + A * np.sin(theta)
        return result
    
    @staticmethod
    def normalized_euler(A: Quaternion, x: float) -> Quaternion:
        """
        Normalized Euler: q^x using exp(x·log(q))
        Standard method in quaternion libraries
        """
        # Normalize A first
        A_norm = A.norm()
        A_unit = A.normalize()
        
        # For unit quaternion q = cos(θ) + v̂ sin(θ)
        # We have log(q) = v̂ θ
        w = A_unit.w
        vec = A_unit.vector_part()
        vec_norm = np.sqrt(np.sum(vec**2))
        
        if vec_norm < 1e-10:
            # A is real, return A^x
            return Quaternion(A.w**x, 0, 0, 0)
        
        # Compute angle θ
        theta = np.arctan2(vec_norm, w)
        
        # Unit vector in imaginary direction
        v_hat = vec / vec_norm
        
        # Fractional power: q^x = exp(x·log(q)) = exp(x·v̂·θ)
        # = A_norm^x · [cos(xθ) + v̂ sin(xθ)]
        new_angle = x * theta
        cos_new = np.cos(new_angle)
        sin_new = np.sin(new_angle)
        
        scale = A_norm ** x
        
        result = Quaternion(
            scale * cos_new,
            scale * sin_new * v_hat[0],
            scale * sin_new * v_hat[1],
            scale * sin_new * v_hat[2]
        )
        
        return result


def example_1_single_power():
    """Example 1: Compute (i+j)^(1/3)"""
    print("="*80)
    print("EXAMPLE 1: Single Fractional Power (i+j)^(1/3)")
    print("="*80)
    
    A = Quaternion(0, 1, 1, 0)  # i + j
    x = 1/3
    
    print(f"\nInput: A = {A}")
    print(f"Power: x = {x}")
    
    # Method 1: Universal
    print("\n" + "-"*80)
    print("METHOD 1: UNIVERSAL ( Method)")
    print("-"*80)
    universal_result, E, alpha = QuaternionMethods.universal_method(A, x)
    print(f"A² = {(A*A).w:.6f} (should be -2)")
    print(f"α = √(-A²) = {alpha:.6f}")
    print(f"E = A/α = {E}")
    print(f"E² = {(E*E).w:.6f} (should be -1)")
    print(f"\nResult: E^(1/3) = {universal_result}")
    print(f"||E^(1/3)|| = {universal_result.norm():.15f}")
    
    # Verify by cubing
    cubed = universal_result * universal_result * universal_result
    print(f"Verification: (E^(1/3))³ = {cubed}")
    print(f"Expected: E = {E}")
    error_universal = (cubed - E).norm()
    print(f"Error: {error_universal:.3e}")
    
    # Method 2: Naive Euler
    print("\n" + "-"*80)
    print("METHOD 2: NAIVE EULER (No Normalization)")
    print("-"*80)
    naive_result = QuaternionMethods.naive_euler(A, x)
    print(f"Result: {naive_result}")
    print(f"||Result|| = {naive_result.norm():.15f}")
    
    cubed_naive = naive_result * naive_result * naive_result
    print(f"Verification: (Result)³ = {cubed_naive}")
    print(f"Expected: A = {A}")
    error_naive = (cubed_naive - A).norm()
    print(f"Error: {error_naive:.3e}")
    
    # Method 3: Normalized Euler (standard library method)
    print("\n" + "-"*80)
    print("METHOD 3: NORMALIZED EULER (Standard Library Method)")
    print("-"*80)
    normalized_result = QuaternionMethods.normalized_euler(A, x)
    print(f"Result: {normalized_result}")
    print(f"||Result|| = {normalized_result.norm():.15f}")
    
    cubed_normalized = normalized_result * normalized_result * normalized_result
    print(f"Verification: (Result)³ = {cubed_normalized}")
    print(f"Expected: A = {A}")
    error_normalized = (cubed_normalized - A).norm()
    print(f"Error: {error_normalized:.3e}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'||Result||':<20} {'Cubing Error':<15} {'Status'}")
    print("-"*80)
    print(f"{'Universal':<25} {universal_result.norm():<20.15f} {error_universal:<15.3e} {'✓ EXACT'}")
    print(f"{'Naive Euler':<25} {naive_result.norm():<20.15f} {error_naive:<15.3e} {'✗ WRONG'}")
    print(f"{'Normalized Euler':<25} {normalized_result.norm():<20.15f} {error_normalized:<15.3e} {'✓ GOOD'}")
    
    # Check if universal and normalized give same result
    diff = (universal_result - normalized_result).norm()
    print(f"\nDifference between Universal and Normalized Euler: {diff:.3e}")
    
    if diff < 1e-10:
        print("✓ Both methods give IDENTICAL results!")
    else:
        print(f"⚠ Methods differ by {diff:.3e}")
        print(f"  Universal: {universal_result}")
        print(f"  Normalized: {normalized_result}")
    
    return {
        'universal': universal_result,
        'naive': naive_result,
        'normalized': normalized_result,
        'error_universal': error_universal,
        'error_naive': error_naive,
        'error_normalized': error_normalized
    }


def example_2_iterative_stability():
    """Example 2: Test stability under 100 iterations"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Iterative Stability - 100 Compositions")
    print("="*80)
    
    A = Quaternion(0, 1, 1, 0)  # i + j
    n_iterations = 100
    x_step = 4 / n_iterations  # E^4 = 1, so 100 steps of E^(4/100)
    
    print(f"\nSetup: Compose E^(4/{n_iterations}) {n_iterations} times")
    print(f"Expected result: E^4 = 1 (identity)")
    
    # Method 1: Universal
    print("\n" + "-"*80)
    print("METHOD 1: UNIVERSAL")
    print("-"*80)
    E_step_universal, E, alpha = QuaternionMethods.universal_method(A, x_step)
    result_universal = Quaternion(1, 0, 0, 0)
    norms_universal = [1.0]
    
    for i in range(n_iterations):
        result_universal = E_step_universal * result_universal
        norms_universal.append(result_universal.norm())
    
    print(f"Final result: {result_universal}")
    print(f"||Result|| = {result_universal.norm():.15f}")
    error_universal = abs(result_universal.norm() - 1.0)
    print(f"Norm drift: {error_universal:.3e}")
    
    # Method 2: Naive
    print("\n" + "-"*80)
    print("METHOD 2: NAIVE EULER")
    print("-"*80)
    E_step_naive = QuaternionMethods.naive_euler(A, x_step)
    result_naive = Quaternion(1, 0, 0, 0)
    norms_naive = [1.0]
    
    for i in range(n_iterations):
        result_naive = E_step_naive * result_naive
        norms_naive.append(result_naive.norm())
    
    print(f"Final result: {result_naive}")
    print(f"||Result|| = {result_naive.norm():.15f}")
    error_naive = abs(result_naive.norm() - 1.0)
    print(f"Norm drift: {error_naive:.3e}")
    
    # Method 3: Normalized Euler
    print("\n" + "-"*80)
    print("METHOD 3: NORMALIZED EULER")
    print("-"*80)
    E_step_normalized = QuaternionMethods.normalized_euler(A, x_step)
    result_normalized = Quaternion(1, 0, 0, 0)
    norms_normalized = [1.0]
    
    for i in range(n_iterations):
        result_normalized = E_step_normalized * result_normalized
        norms_normalized.append(result_normalized.norm())
    
    print(f"Final result: {result_normalized}")
    print(f"||Result|| = {result_normalized.norm():.15f}")
    error_normalized = abs(result_normalized.norm() - 1.0)
    print(f"Norm drift: {error_normalized:.3e}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'Final ||·||':<20} {'Drift':<15} {'Improvement'}")
    print("-"*80)
    print(f"{'Universal':<25} {result_universal.norm():<20.15f} {error_universal:<15.3e} {'1.0×'}")
    print(f"{'Naive Euler':<25} {result_naive.norm():<20.15f} {error_naive:<15.3e} {f'{error_naive/error_universal:.1e}× WORSE'}")
    print(f"{'Normalized Euler':<25} {result_normalized.norm():<20.15f} {error_normalized:<15.3e} {f'{error_normalized/error_universal:.1f}×'}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    iterations = list(range(n_iterations + 1))
    plt.plot(iterations, norms_universal, 'g-', linewidth=2, label='Universal')
    plt.plot(iterations, norms_naive, 'r--', linewidth=2, label='Naive Euler')
    plt.plot(iterations, norms_normalized, 'b:', linewidth=2, label='Normalized Euler')
    plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.3, label='Expected')
    plt.xlabel('Iteration')
    plt.ylabel('||Result||')
    plt.title('Norm Stability Under Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    drift_universal = [abs(n - 1.0) for n in norms_universal]
    drift_naive = [abs(n - 1.0) for n in norms_naive]
    drift_normalized = [abs(n - 1.0) for n in norms_normalized]
    
    plt.semilogy(iterations, drift_universal, 'g-', linewidth=2, label='Universal')
    plt.semilogy(iterations, drift_naive, 'r--', linewidth=2, label='Naive Euler')
    plt.semilogy(iterations, drift_normalized, 'b:', linewidth=2, label='Normalized Euler')
    plt.xlabel('Iteration')
    plt.ylabel('|Norm - 1.0| (log scale)')
    plt.title('Norm Drift (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quaternion_stability_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'quaternion_stability_comparison.png'")
    
    return {
        'norms_universal': norms_universal,
        'norms_naive': norms_naive,
        'norms_normalized': norms_normalized,
        'error_universal': error_universal,
        'error_naive': error_naive,
        'error_normalized': error_normalized
    }


def example_3_rotation_accuracy():
    """Example 3: Rotation accuracy test"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Rotation Accuracy - 45° Rotation")
    print("="*80)
    
    E = Quaternion(0, 0, 1, 0)  # j direction
    v = Quaternion(0, 1, 0, 0)  # i vector
    x = 1/2  # E^(1/2) = 45° rotation
    
    print(f"\nSetup: Rotate v = i by 45° around j-axis using E^(1/2)")
    print(f"E = j, v = i")
    
    # Expected analytical result
    expected = Quaternion(0, np.cos(np.pi/4), 0, np.sin(np.pi/4))
    print(f"Expected: {expected}")
    
    # Method 1: Universal
    print("\n" + "-"*80)
    print("METHOD 1: UNIVERSAL")
    print("-"*80)
    E_half_universal, _, _ = QuaternionMethods.universal_method(E, x)
    rotated_universal = E_half_universal * v
    print(f"E^(1/2) = {E_half_universal}")
    print(f"Rotated = {rotated_universal}")
    print(f"||Rotated|| = {rotated_universal.norm():.15f}")
    error_universal = (rotated_universal - expected).norm()
    print(f"Error vs expected: {error_universal:.3e}")
    
    # Method 2: Normalized Euler
    print("\n" + "-"*80)
    print("METHOD 2: NORMALIZED EULER")
    print("-"*80)
    E_half_normalized = QuaternionMethods.normalized_euler(E, x)
    rotated_normalized = E_half_normalized * v
    print(f"E^(1/2) = {E_half_normalized}")
    print(f"Rotated = {rotated_normalized}")
    print(f"||Rotated|| = {rotated_normalized.norm():.15f}")
    error_normalized = (rotated_normalized - expected).norm()
    print(f"Error vs expected: {error_normalized:.3e}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    diff = (E_half_universal - E_half_normalized).norm()
    print(f"Difference between methods: {diff:.3e}")
    
    if diff < 1e-10:
        print("✓ Both methods give IDENTICAL results for normalized inputs!")
    
    return {
        'universal': rotated_universal,
        'normalized': rotated_normalized,
        'expected': expected,
        'error_universal': error_universal,
        'error_normalized': error_normalized
    }


def comprehensive_summary():
    """Generate comprehensive summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n1. SINGLE FRACTIONAL POWER (i+j)^(1/3)")
    print("-" * 80)
    results1 = example_1_single_power()
    
    print("\n2. ITERATIVE STABILITY (100 compositions)")
    print("-" * 80)
    results2 = example_2_iterative_stability()
    
    print("\n3. ROTATION ACCURACY")
    print("-" * 80)
    results3 = example_3_rotation_accuracy()
    



def generalized_example_1_single_power(Q,b):
        """Example 1: Compute (i+j)^(1/3)"""
        x = 1/b
        print("="*80)
        print(f"EXAMPLE : Single Fractional Power (i+j)^({x})")
        print("="*80)



        A = Q  # i + j
        
        naive_result = QuaternionMethods.naive_euler(A, x)
        normalized_result = QuaternionMethods.normalized_euler(A, x)
        print(f"\nInput: A = {A}")
        print(f"Power: x = {x}")
        
        # Method 1: Universal
        print("\n" + "-"*80)
        print("METHOD 1: UNIVERSAL ( Method)")
        print("-"*80)
        universal_result, E, alpha = QuaternionMethods.universal_method(A, x)
        print(f"A² = {(A*A).w:.6f} (should be -2)")
        print(f"α = √(-A²) = {alpha:.6f}")
        print(f"E = A/α = {E}")
        print(f"E² = {(E*E).w:.6f} (should be -1)")
        print(f"\nResult: E^({x}) = {universal_result}")
        print(f"||E^({x})|| = {universal_result.norm():.15f}")
        
        # Verify by cubing
        cubed = universal_result
        cubed_naive = naive_result
        cubed_normalized = normalized_result 
        for i in range(1,b):
            cubed = cubed * universal_result
            cubed_naive= cubed_naive * naive_result
            cubed_normalized = cubed_normalized *normalized_result
        print(f"Verification: (E^({x}))³ = {cubed}")
        print(f"Expected: E = {E}")
        error_universal = (cubed - E).norm()
        print(f"Error: {error_universal:.3e}")
        
        # Method 2: Naive Euler
        print("\n" + "-"*80)
        print("METHOD 2: NAIVE EULER (No Normalization)")
        print("-"*80)
        naive_result = QuaternionMethods.naive_euler(A, x)
        print(f"Result: {naive_result}")
        print(f"||Result|| = {naive_result.norm():.15f}")
        
        

        print(f"Verification: (Result)³ = {cubed_naive}")
        print(f"Expected: A = {A}")
        error_naive = (cubed_naive - A).norm()
        print(f"Error: {error_naive:.3e}")
        
        # Method 3: Normalized Euler (standard library method)
        print("\n" + "-"*80)
        print("METHOD 3: NORMALIZED EULER (Standard Library Method)")
        print("-"*80)
        normalized_result = QuaternionMethods.normalized_euler(A, x)
        print(f"Result: {normalized_result}")
        print(f"||Result|| = {normalized_result.norm():.15f}")
        

        print(f"Verification: (Result)^(1/{b}) = {cubed_normalized}")
        print(f"Expected: A = {A}")
        error_normalized = (cubed_normalized - A).norm()
        print(f"Error: {error_normalized:.3e}")
        
        # Comparison
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"{'Method':<25} {'||Result||':<20} {'exponentation Error':<15} {'Status'}")
        print("-"*80)
        print(f"{'Universal ()':<25} {universal_result.norm():<20.15f} {error_universal:<15.3e} {'✓ EXACT'}")
        print(f"{'Naive Euler':<25} {naive_result.norm():<20.15f} {error_naive:<15.3e} {'✗ WRONG'}")
        print(f"{'Normalized Euler':<25} {normalized_result.norm():<20.15f} {error_normalized:<15.3e} {'✓ GOOD'}")
        
        # Check if universal and normalized give same result
        diff = (universal_result - normalized_result).norm()
        print(f"\nDifference between Universal and Normalized Euler: {diff:.3e}")
        
        if diff < 1e-10:
            print("✓ Both methods give IDENTICAL results!")
        else:
            print(f"⚠ Methods differ by {diff:.3e}")
            print(f"  Universal: {universal_result}")
            print(f"  Normalized: {normalized_result}")
        
        return {
            'universal': universal_result,
            'naive': naive_result,
            'normalized': normalized_result,
            'error_universal': error_universal,
            'error_naive': error_naive,
            'error_normalized': error_normalized
        }

    

if __name__ == "__main__":
    comprehensive_summary()
    print("*"*50)
    # for demo purposese we assume 1/x always
    generalized_example_1_single_power(Quaternion(0,0,1,1),7)
    generalized_example_1_sigeneralized_example_1_single_powerngle_power(Quaternion(0,0,1,1),12)
    generalized_example_1_single_power(Quaternion(0,0,1,1),57)

    plt.show()