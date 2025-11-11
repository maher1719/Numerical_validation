
import numpy as np
from math import pi, cosh, sinh, isclose

# Helper: matrix exponential via eigendecomposition (works when diagonalizable)
def matrix_expm(M):
    vals, vecs = np.linalg.eig(M)
    # construct exp(D) and compute V exp(D) V^{-1}
    expD = np.diag(np.exp(vals))
    V = vecs
    Vinv = np.linalg.inv(V)
    return V @ expD @ Vinv

# Define hyperbolic unit H = [[0,1],[1,0]] so H^2 = I
H = np.array([[0., 1.],
              [1., 0.]])
I2 = np.eye(2)

# Principal logarithm choice: Log(H) = H * (pi/2) (valid because H^2 = I and minimal polynomial)
def principal_log_H(H):
    return H * (pi/2)

def hyperbolic_formula(H, x):
    t = pi * x / 2
    return cosh(t) * I2 + sinh(t) * H

def test_values(xs):
    results = []
    LogH = principal_log_H(H)
    for x in xs:
        # compute formula matrix
        M_formula = hyperbolic_formula(H, x)
        # compute via matrix exponential: H^x := exp(x Log(H))
        M_expm = matrix_expm(x * LogH)
        # compute H^{H x} = exp(H x Log(H))
        M_collapsed = matrix_expm(H * x @ LogH)  # careful multiplication: H @ (x*LogH) but H and LogH commute here
        # However since LogH = (pi/2)*H and H@H = I, H * (x*LogH) = H @ (x*LogH) = x*(pi/2) * H@H = x*(pi/2) * I
        # compute expected collapsed scalar
        scalar_expected = np.exp((np.trace(H @ H) / 2) * x * pi/2)  # not used; we compute directly below
        # More directly, since H^2 = I, H @ H = I -> collapsed = exp(x*pi/2 * I) = exp(x*pi/2) * I
        collapsed_expected = np.exp(x * pi/2) * I2
        
        # numerical errors (Frobenius norm)
        err_formula = np.linalg.norm(M_formula - M_expm)
        err_collapse = np.linalg.norm(M_collapsed - collapsed_expected)
        
        results.append({
            'x': x,
            'M_formula': M_formula,
            'M_expm': M_expm,
            'err_formula': err_formula,
            'M_collapsed': M_collapsed,
            'collapsed_expected': collapsed_expected,
            'err_collapse': err_collapse
        })
    return results

# Choose test x values
xs = [0.5, 1.0, 2.0, -1.0, 1/13]

results = test_values(xs)

# Print results succinctly
for r in results:
    x = r['x']
    print(f"\n=== x = {x} ===")
    print("Hyperbolic formula H^x = cosh(t)*I + sinh(t)*H:")
    print(r['M_formula'])
    print("Matrix exponential exp(x Log(H)):")
    print(np.round(r['M_expm'], 12))
    print(f"Frobenius norm error between formula and exp: {r['err_formula']:.3e}")
    print("\nCollapsed matrix H^{H x} via exp(H x Log(H)):")
    print(np.round(r['M_collapsed'], 12))
    print("Expected collapsed scalar matrix exp(x*pi/2)*I:")
    print(np.round(r['collapsed_expected'], 12))
    print(f"Frobenius norm error for collapse identity: {r['err_collapse']:.3e}")

# Summary check: all errors should be near numerical zero
max_err_formula = max(r['err_formula'] for r in results)
max_err_collapse = max(r['err_collapse'] for r in results)
print("\nSUMMARY:")
print(f"Max formula error across tests: {max_err_formula:.3e}")
print(f"Max collapse error across tests: {max_err_collapse:.3e}")
