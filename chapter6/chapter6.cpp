#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace Eigen;

// Cramer's Rule (for small square systems only)
VectorXd cramerSolve(const MatrixXd& A, const VectorXd& b) {
    int n = A.rows();
    double detA = A.determinant();
    if (abs(detA) < 1e-12) {
        throw runtime_error("Matrix is singular (det ~ 0), Cramer's rule fails.");
    }
    VectorXd x(n);
    for (int i = 0; i < n; ++i) {
        MatrixXd Ai = A;
        Ai.col(i) = b;
        x(i) = Ai.determinant() / detA;
    }
    return x;
}

// Jacobi Iteration
VectorXd jacobiSolve(const MatrixXd& A, const VectorXd& b, int maxIter = 1000, double tol = 1e-8) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd x_new(n);

    for (int i = 0; i < n; ++i) {
        if (abs(A(i, i)) < 1e-12) {
            throw runtime_error("Jacobi: Zero diagonal at (" + to_string(i) + "," + to_string(i) + ")");
        }
    }

    for (int iter = 0; iter < maxIter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) sum += A(i, j) * x(j);
            }
            x_new(i) = (b(i) - sum) / A(i, i);
        }
        if ((x_new - x).norm() < tol) {
            return x_new;
        }
        x = x_new;
    }
    throw runtime_error("Jacobi did not converge within " + to_string(maxIter) + " iterations.");
}

// Normal Equations (Least Squares)
VectorXd normalEquations(const MatrixXd& A, const VectorXd& b) {
    MatrixXd AtA = A.transpose() * A;
    VectorXd Atb = A.transpose() * b;
    return AtA.fullPivLu().solve(Atb);
}

// QR-based Least Squares
VectorXd qrLeastSquares(const MatrixXd& A, const VectorXd& b) {
    return A.householderQr().solve(b);
}

// SVD-based Least Squares
VectorXd svdLeastSquares(const MatrixXd& A, const VectorXd& b) {
    return A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
}

// Main function
int main() {
    // Use fixed-point notation throughout
    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(6);

    cout << "=== Chapter 6: Special Case Solvers ===" << endl;
    cout << "Comparing with Ch4 (Gaussian/LU) and Ch5 (QR/SVD)" << endl;
    cout << "=========================================" << endl;

    // --------------------------------------------------
    // TEST 1: Overdetermined system (3 equations, 2 unknowns)
    // Equations:
    //   x1 + x2 = 1
    //  2x1 + x2 = 2
    //   x1 + 2x2 = 3
    // True least-squares solution: [3/11, 14/11] = [0.272727, 1.272727]
    // --------------------------------------------------
    cout << "\n[TEST 1] Overdetermined System (m=3, n=2)" << endl;
    MatrixXd A1(3, 2);
    A1 << 1, 1,
          2, 1,
          1, 2;
    VectorXd b1(3); b1 << 1, 2, 3;

    cout << "A =\n" << A1 << "\nb = " << b1.transpose() << endl;

    VectorXd x_qr  = qrLeastSquares(A1, b1);
    VectorXd x_svd = svdLeastSquares(A1, b1);
    VectorXd x_ne  = normalEquations(A1, b1);
    VectorXd x_true(2); x_true << 3.0/11.0, 14.0/11.0;

    cout << "QR solution:     " << x_qr.transpose() << endl;
    cout << "SVD solution:    " << x_svd.transpose() << endl;
    cout << "Normal Eq:       " << x_ne.transpose() << endl;
    cout << "True solution:   " << x_true.transpose() << endl;
    cout << "Error (QR):      " << (x_qr - x_true).norm() << endl;
    cout << "Error (SVD):     " << (x_svd - x_true).norm() << endl;
    cout << "Error (Normal):  " << (x_ne - x_true).norm() << endl;

    // --------------------------------------------------
    // TEST 2: Ill-conditioned system (eps = 1e-8)
    // A = [[1, 1], [1, 1+eps]], b = [1, 1]
    // True solution: [1, 0]
    // --------------------------------------------------
    cout << "\n[TEST 2] Ill-conditioned System (eps = 1e-8)" << endl;
    double eps = 1e-8;
    MatrixXd A2(2, 2);
    A2 << 1.0, 1.0,
          1.0, 1.0 + eps;
    VectorXd b2(2); b2 << 1.0, 1.0;

    cout << "A =\n" << A2 << endl;

    // Safe singular value check (no Greek letters!)
    JacobiSVD<MatrixXd> svd_test(A2);
    VectorXd s_vals = svd_test.singularValues();
    double s_max = s_vals(0);
    double s_min = s_vals(s_vals.size() - 1);

    cout << "s_max = " << s_max << ", s_min = " << s_min << endl;

    if (s_min < 1e-15) {
        cout << "Matrix is numerically singular (s_min ~ 0)." << endl;
    } else {
        double condA = s_max / s_min;
        if (condA > 1e12) {
            cout << "Condition number is very large (> 1e12) -> ill-conditioned." << endl;
        } else {
            cout << "Condition number ~ " << condA << endl;
        }
    }

    VectorXd x_qr2  = qrLeastSquares(A2, b2);
    VectorXd x_svd2 = svdLeastSquares(A2, b2);
    VectorXd x_ne2  = normalEquations(A2, b2);
    VectorXd x_true2(2); x_true2 << 1.0, 0.0;

    cout << "QR solution:     " << x_qr2.transpose() << endl;
    cout << "SVD solution:    " << x_svd2.transpose() << endl;
    cout << "Normal Eq:       " << x_ne2.transpose() << endl;
    cout << "True solution:   " << x_true2.transpose() << endl;
    cout << "Error (QR):      " << (x_qr2 - x_true2).norm() << endl;
    cout << "Error (SVD):     " << (x_svd2 - x_true2).norm() << endl;
    cout << "Error (Normal):  " << (x_ne2 - x_true2).norm() << endl;

    // --------------------------------------------------
    // TEST 3: Cramer's Rule (2x2)
    // 2x1 + 3x2 = 1
    // 3x1 + 2x2 = 2
    // Solution: [0.8, -0.2]
    // --------------------------------------------------
    cout << "\n[TEST 3] Cramer's Rule (2x2 system)" << endl;
    MatrixXd A3(2, 2);
    A3 << 2, 3,
          3, 2;
    VectorXd b3(2); b3 << 1, 2;

    try {
        auto start = chrono::high_resolution_clock::now();
        VectorXd x_cramer = cramerSolve(A3, b3);
        auto end = chrono::high_resolution_clock::now();
        double time_us = chrono::duration_cast<chrono::microseconds>(end - start).count();

        cout << "Cramer solution: " << x_cramer.transpose()
             << " (time: " << time_us << " us)" << endl;

        VectorXd x_gauss = A3.fullPivLu().solve(b3);
        cout << "Gaussian sol:    " << x_gauss.transpose() << endl;
        cout << "Error vs Gaussian: " << (x_cramer - x_gauss).norm() << endl;
    } catch (const exception& e) {
        cout << "Cramer failed: " << e.what() << endl;
    }

    // --------------------------------------------------
    // TEST 4: Cramer on singular matrix
    // --------------------------------------------------
    cout << "\n[TEST 4] Cramer on Singular 4x4 Matrix" << endl;
    MatrixXd A4(4, 4);
    A4 << 1, 2, 3, 4,
          5, 6, 7, 8,
          9,10,11,12,
         13,14,15,16;
    VectorXd b4(4); b4 << 1, 2, 3, 4;
    try {
        VectorXd x4 = cramerSolve(A4, b4);
        cout << "Unexpected success!" << endl;
    } catch (const exception& e) {
        cout << "Cramer correctly failed: " << e.what() << endl;
    }

    // --------------------------------------------------
    // TEST 5: Jacobi (convergent case)
    // 4x1 - x2 = 3
    // -x1 + 4x2 = 3
    // Solution: [1, 1]
    // --------------------------------------------------
    cout << "\n[TEST 5] Jacobi Iteration (Convergent Case)" << endl;
    MatrixXd A5(2, 2);
    A5 << 4, -1,
         -1,  4;
    VectorXd b5(2); b5 << 3, 3;

    try {
        VectorXd x_jacobi = jacobiSolve(A5, b5);
        cout << "Jacobi solution: " << x_jacobi.transpose() << endl;
        VectorXd x_exact = A5.inverse() * b5;
        cout << "Exact solution:  " << x_exact.transpose() << endl;
        cout << "Error:           " << (x_jacobi - x_exact).norm() << endl;
    } catch (const exception& e) {
        cout << "Jacobi failed: " << e.what() << endl;
    }

    // --------------------------------------------------
    // TEST 6: Jacobi (non-convergent)
    // --------------------------------------------------
    cout << "\n[TEST 6] Jacobi on Non-Diagonally Dominant Matrix" << endl;
    MatrixXd A6(2, 2);
    A6 << 1, 2,
          3, 4;
    VectorXd b6(2); b6 << 1, 1;
    try {
        VectorXd x6 = jacobiSolve(A6, b6, 100);
        cout << "Jacobi gave: " << x6.transpose() << " (may be inaccurate!)" << endl;
        VectorXd x_direct = A6.fullPivLu().solve(b6);
        cout << "Direct solve: " << x_direct.transpose() << endl;
    } catch (const exception& e) {
        cout << "Jacobi correctly failed to converge." << endl;
    }

    // --------------------------------------------------
    // Summary
    // --------------------------------------------------
    cout << "\n=== Summary: When to Use Which Method? ===" << endl;
    cout << "Direct methods (Ch4): Small/medium square systems." << endl;
    cout << "QR/SVD (Ch5): Stable for least squares (recommended)." << endl;
    cout << "Normal equations (Ch6.1.2): Avoid in ill-conditioned cases." << endl;
    cout << "Cramer (Ch6.2): Only for tiny theoretical examples." << endl;
    cout << "Jacobi (Ch6.3): Only if diagonally dominant; slow." << endl;

    return 0;
}