#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Gaussian Elimination with partial pivoting
VectorXd gaussianElimination(MatrixXd A, VectorXd b) {
    int n = A.rows();
    
    // Build augmented matrix [A | b]
    MatrixXd aug(n, n + 1);
    aug << A, b;
    
    // Forward elimination with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find pivot row
        int maxRow = k;
        double maxVal = abs(aug(k, k));
        for (int i = k + 1; i < n; i++) {
            if (abs(aug(i, k)) > maxVal) {
                maxVal = abs(aug(i, k));
                maxRow = i;
            }
        }
        
        // Swap rows if needed
        if (maxRow != k) {
            aug.row(k).swap(aug.row(maxRow));
        }
        
        // Check for singular matrix
        if (abs(aug(k, k)) < 1e-12) {
            throw runtime_error("Matrix is singular or nearly singular.");
        }
        
        // Eliminate below
        for (int i = k + 1; i < n; i++) {
            double factor = aug(i, k) / aug(k, k);
            for (int j = k; j <= n; j++) {
                aug(i, j) -= factor * aug(k, j);
            }
        }
    }
    
    // Back substitution
    VectorXd x(n);
    for (int i = n - 1; i >= 0; i--) {
        x(i) = aug(i, n);
        for (int j = i + 1; j < n; j++) {
            x(i) -= aug(i, j) * x(j);
        }
        x(i) /= aug(i, i);
    }
    
    return x;
}

// Matrix inversion method
VectorXd matrixInversionMethod(MatrixXd A, VectorXd b) {
    MatrixXd A_inv = A.inverse();
    return A_inv * b;
}

// LU decomposition method (using Eigen)
VectorXd luDecompositionMethod(MatrixXd A, VectorXd b) {
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

// Generate Hilbert matrix
MatrixXd hilbertMatrix(int n) {
    MatrixXd H(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H(i, j) = 1.0 / (i + j + 1);
        }
    }
    return H;
}

// Test accuracy of methods
void testMethods() {
    cout << "=== Accuracy Test of Three Methods ===" << endl;
    
    // Test case 1: Well-conditioned system
    MatrixXd A1(3, 3);
    A1 << 2, 3, 4,
          3, 2, 5,
          4, 5, 3;
    VectorXd b1(3);
    b1 << 1, 2, 3;
    
    cout << "\nTest Case 1 - Well-conditioned system:" << endl;
    cout << "A = \n" << A1 << endl;
    cout << "b = " << b1.transpose() << endl;
    
    // Reference solution using Eigen's QR solver
    VectorXd exactSolution = A1.colPivHouseholderQr().solve(b1);
    cout << "Reference solution: " << exactSolution.transpose() << endl;
    
    try {
        VectorXd solGauss = gaussianElimination(A1, b1);
        cout << "Gaussian elimination: " << solGauss.transpose() << endl;
        cout << "Error (Gaussian): " << (solGauss - exactSolution).norm() << endl;
    } catch (const exception& e) {
        cout << "Gaussian elimination failed: " << e.what() << endl;
    }
    
    try {
        VectorXd solInv = matrixInversionMethod(A1, b1);
        cout << "Matrix inversion:    " << solInv.transpose() << endl;
        cout << "Error (Inversion):   " << (solInv - exactSolution).norm() << endl;
    } catch (const exception& e) {
        cout << "Matrix inversion failed: " << e.what() << endl;
    }
    
    try {
        VectorXd solLU = luDecompositionMethod(A1, b1);
        cout << "LU decomposition:    " << solLU.transpose() << endl;
        cout << "Error (LU):          " << (solLU - exactSolution).norm() << endl;
    } catch (const exception& e) {
        cout << "LU decomposition failed: " << e.what() << endl;
    }
    
    // Test case 2: Ill-conditioned Hilbert matrix
    int n = 5;
    MatrixXd A2 = hilbertMatrix(n);
    VectorXd xTrue = VectorXd::Ones(n);
    VectorXd b2 = A2 * xTrue;
    // Compute condition number using SVD
    JacobiSVD<MatrixXd> svd(A2);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(svd.singularValues().size() - 1);
    double condNum = sigma_max / sigma_min; 
    
    cout << "\nTest Case 2 - " << n << "x" << n << " Hilbert matrix:" << endl;
    cout << "Condition number: " << scientific << setprecision(2) << condNum << endl;
    cout << "True solution: " << xTrue.transpose() << endl;
    
    try {
        VectorXd solGauss = gaussianElimination(A2, b2);
        cout << "Gaussian elimination: " << solGauss.transpose() << endl;
        cout << "Error (Gaussian): " << (solGauss - xTrue).norm() << endl;
    } catch (const exception& e) {
        cout << "Gaussian elimination failed: " << e.what() << endl;
    }
    
    try {
        VectorXd solInv = matrixInversionMethod(A2, b2);
        cout << "Matrix inversion:    " << solInv.transpose() << endl;
        cout << "Error (Inversion):   " << (solInv - xTrue).norm() << endl;
    } catch (const exception& e) {
        cout << "Matrix inversion failed: " << e.what() << endl;
    }
    
    try {
        VectorXd solLU = luDecompositionMethod(A2, b2);
        cout << "LU decomposition:    " << solLU.transpose() << endl;
        cout << "Error (LU):          " << (solLU - xTrue).norm() << endl;
    } catch (const exception& e) {
        cout << "LU decomposition failed: " << e.what() << endl;
    }

    // Test Case 3: Nearly singular matrix
    cout << "\nTest Case 3 - Nearly Singular Matrix:" << endl;
    MatrixXd A3(3, 3);
    A3 << 1, 2, 3,
          4, 5, 6,
          7, 8, 9.0000001; // tiny perturbation
    VectorXd b3(3);
    b3 << 1, 2, 3;

    cout << "A = \n" << A3 << endl;

    // Compute condition number manually via SVD (safe for all Eigen versions)
    JacobiSVD<MatrixXd> svd3(A3);
    double sigma_max3 = svd3.singularValues()(0);
    double sigma_min3 = svd3.singularValues()(svd3.singularValues().size() - 1);
    double cond3 = sigma_max3 / sigma_min3;
    cout << "Condition number (via SVD): " << scientific << setprecision(2) << cond3 << endl;

    try {
        VectorXd x_inv = matrixInversionMethod(A3, b3);
        cout << "Matrix inversion solution: " << x_inv.transpose() << endl;
    } catch (const exception& e) {
        cout << "Matrix inversion failed: " << e.what() << endl;
    }

    try {
        VectorXd x_lu = luDecompositionMethod(A3, b3);
        cout << "LU decomposition solution: " << x_lu.transpose() << endl;
    } catch (const exception& e) {
        cout << "LU decomposition failed: " << e.what() << endl;
    }
}

// Performance test
void performanceTest() {
    cout << "\n=== Performance Test (Large Scale & Multiple RHS) ===" << endl;
    
    int n = 500; // large matrix
    int nrhs = 10; // multiple right-hand sides
    
    // Generate well-conditioned matrix
    MatrixXd A = MatrixXd::Random(n, n);
    A = A * A.transpose() + n * MatrixXd::Identity(n, n); // SPD and well-conditioned
    vector<VectorXd> B;
    for (int k = 0; k < nrhs; k++) {
        B.push_back(VectorXd::Random(n));
    }
    
    cout << "Matrix size: " << n << "x" << n << ", RHS count: " << nrhs << endl;
    
    // Test Gaussian (inefficient for multiple RHS)
    auto start = chrono::high_resolution_clock::now();
    for (int k = 0; k < nrhs; k++) {
        gaussianElimination(A, B[k]);
    }
    auto end = chrono::high_resolution_clock::now();
    auto gauss_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "  Gaussian (repeated): " << gauss_ms << " ms" << endl;
    
    // Test Matrix Inversion (compute once, apply many times)
    start = chrono::high_resolution_clock::now();
    MatrixXd A_inv = A.inverse();
    for (int k = 0; k < nrhs; k++) {
        VectorXd x = A_inv * B[k];
    }
    end = chrono::high_resolution_clock::now();
    auto inv_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "  Matrix inversion:    " << inv_ms << " ms" << endl;
    
    // Test LU (decompose once, solve many times)
    start = chrono::high_resolution_clock::now();
    PartialPivLU<MatrixXd> lu(A);
    for (int k = 0; k < nrhs; k++) {
        VectorXd x = lu.solve(B[k]);
    }
    end = chrono::high_resolution_clock::now();
    auto lu_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "  LU decomposition:    " << lu_ms << " ms" << endl;
}

// Stability test
void stabilityTest() {
    cout << "\n=== Numerical Stability Test (Extreme Cases) ===" << endl;
    
    vector<double> condNumbers = {1e10, 1e12, 1e14, 1e16};
    
    for (double targetCond : condNumbers) {
        cout << "\nTarget condition number: " << scientific << setprecision(0) << targetCond << endl;
        
        int n = 12; // larger size amplifies instability
        
        // Construct matrix with singular values: [1, 1/κ^(1/(n-1)), ..., 1/κ]
        VectorXd sigma(n);
        sigma(0) = 1.0;
        double ratio = pow(targetCond, -1.0 / (n - 1));
        for (int i = 1; i < n; ++i) {
            sigma(i) = sigma(i-1) * ratio;
        }
        
        // Random orthogonal matrices
        MatrixXd U = MatrixXd::Random(n, n).householderQr().householderQ();
        MatrixXd V = MatrixXd::Random(n, n).householderQr().householderQ();
        MatrixXd A = U * sigma.asDiagonal() * V.transpose();
        
        VectorXd xTrue = VectorXd::Random(n); // non-unit vector to avoid lucky cancellation
        VectorXd b = A * xTrue;
        
        try {
            VectorXd xGauss = gaussianElimination(A, b);
            double errGauss = (xGauss - xTrue).norm() / xTrue.norm();
            cout << "  Gaussian error: " << scientific << setprecision(2) << errGauss << endl;
        } catch (...) {
            cout << "  Gaussian: failed" << endl;
        }
        
        try {
            VectorXd xInv = matrixInversionMethod(A, b);
            double errInv = (xInv - xTrue).norm() / xTrue.norm();
            cout << "  Inversion error: " << scientific << setprecision(2) << errInv << endl;
        } catch (...) {
            cout << "  Inversion: failed" << endl;
        }
        
        try {
            VectorXd xLU = luDecompositionMethod(A, b);
            double errLU = (xLU - xTrue).norm() / xTrue.norm();
            cout << "  LU error:       " << scientific << setprecision(2) << errLU << endl;
        } catch (...) {
            cout << "  LU: failed" << endl;
        }
    }
}

int main() {
    cout << "Linear System Solver: Implementation and Performance Comparison" << endl;
    cout << "================================================================" << endl;
    
    testMethods();
    performanceTest();
    stabilityTest();
    
    cout << "\n=== Summary of Method Characteristics ===" << endl;
    cout << "1. Gaussian Elimination:" << endl;
    cout << "   + Simple implementation, good numerical stability (with pivoting)" << endl;
    cout << "   - Not efficient for multiple RHS with same A" << endl;
    
    cout << "\n2. Matrix Inversion:" << endl;
    cout << "   + Conceptually simple" << endl;
    cout << "   - Poor numerical stability, inefficient, not recommended" << endl;
    
    cout << "\n3. LU Decomposition:" << endl;
    cout << "   + Excellent stability, efficient for multiple RHS" << endl;
    cout << "   - Requires decomposition step (but worth it)" << endl;
    
    return 0;
}