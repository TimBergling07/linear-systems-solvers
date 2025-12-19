#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// 高斯消元法（使用第四章的实现，带列主元）
VectorXd gaussianElimination(MatrixXd A, VectorXd b) {
    int n = A.rows();
    MatrixXd aug(n, n + 1);
    aug << A, b;

    for (int k = 0; k < n; k++) {
        int maxRow = k;
        double maxVal = abs(aug(k, k));
        for (int i = k + 1; i < n; i++) {
            if (abs(aug(i, k)) > maxVal) {
                maxVal = abs(aug(i, k));
                maxRow = i;
            }
        }
        if (maxRow != k) aug.row(k).swap(aug.row(maxRow));
        if (abs(aug(k, k)) < 1e-12) throw runtime_error("Matrix is singular or nearly singular.");

        for (int i = k + 1; i < n; i++) {
            double factor = aug(i, k) / aug(k, k);
            for (int j = k; j <= n; j++) {
                aug(i, j) -= factor * aug(k, j);
            }
        }
    }

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

// QR 分解法
VectorXd qrDecompositionMethod(MatrixXd A, VectorXd b) {
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// SVD 分解法（最小二乘解）
VectorXd svdDecompositionMethod(MatrixXd A, VectorXd b) {
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    return svd.solve(b);
}

// 生成Hilbert矩阵
MatrixXd hilbertMatrix(int n) {
    MatrixXd H(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H(i, j) = 1.0 / (i + j + 1);
        }
    }
    return H;
}

// 测试三种方法的准确性
void testMethods() {
    cout << "=== Accuracy Test of Three Methods (QR vs SVD vs Gaussian) ===" << endl;

    // Test Case 1: Well-conditioned system
    MatrixXd A1(3, 3);
    A1 << 2, 3, 4,
          3, 2, 5,
          4, 5, 3;
    VectorXd b1(3);
    b1 << 1, 2, 3;

    cout << "\nTest Case 1 - Well-conditioned system:" << endl;
    cout << "A = \n" << A1 << endl;
    cout << "b = " << b1.transpose() << endl;

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
        VectorXd solQR = qrDecompositionMethod(A1, b1);
        cout << "QR decomposition:     " << solQR.transpose() << endl;
        cout << "Error (QR):           " << (solQR - exactSolution).norm() << endl;
    } catch (const exception& e) {
        cout << "QR decomposition failed: " << e.what() << endl;
    }

    try {
        VectorXd solSVD = svdDecompositionMethod(A1, b1);
        cout << "SVD decomposition:    " << solSVD.transpose() << endl;
        cout << "Error (SVD):          " << (solSVD - exactSolution).norm() << endl;
    } catch (const exception& e) {
        cout << "SVD decomposition failed: " << e.what() << endl;
    }

    // Test Case 2: Ill-conditioned Hilbert matrix
    int n = 6;
    MatrixXd A2 = hilbertMatrix(n);
    VectorXd xTrue = VectorXd::Ones(n);
    VectorXd b2 = A2 * xTrue;

    // Manual condition number
    JacobiSVD<MatrixXd> svd2(A2);
    double condNum = svd2.singularValues()(0) / svd2.singularValues()(svd2.singularValues().size() - 1);

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
        VectorXd solQR = qrDecompositionMethod(A2, b2);
        cout << "QR decomposition:     " << solQR.transpose() << endl;
        cout << "Error (QR):           " << (solQR - xTrue).norm() << endl;
    } catch (const exception& e) {
        cout << "QR decomposition failed: " << e.what() << endl;
    }

    try {
        VectorXd solSVD = svdDecompositionMethod(A2, b2);
        cout << "SVD decomposition:    " << solSVD.transpose() << endl;
        cout << "Error (SVD):          " << (solSVD - xTrue).norm() << endl;
    } catch (const exception& e) {
        cout << "SVD decomposition failed: " << e.what() << endl;
    }
}

// 性能测试
void performanceTest() {
    cout << "\n=== Performance Test (Time Complexity) ===" << endl;

    vector<int> sizes = {50, 100, 200, 300};
    vector<string> methods = {"gaussian", "qr", "svd"};

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int n : sizes) {
        cout << "\nMatrix size: " << n << "x" << n << endl;

        MatrixXd A(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A(i, j) = dis(gen);
            }
        }
        A += n * MatrixXd::Identity(n, n);
        VectorXd b = VectorXd::Random(n);

        for (const string& method : methods) {
            auto start = chrono::high_resolution_clock::now();

            if (method == "gaussian") {
                gaussianElimination(A, b);
            } else if (method == "qr") {
                qrDecompositionMethod(A, b);
            } else if (method == "svd") {
                svdDecompositionMethod(A, b);
            }

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            cout << "  " << setw(10) << left << method << ": "
                 << fixed << setprecision(2) << duration.count() / 1000.0 << " ms" << endl;
        }
    }
}

// 数值稳定性测试（极端病态）
void stabilityTest() {
    cout << "\n=== Numerical Stability Test (Extreme Ill-Conditioning) ===" << endl;

    vector<double> condNumbers = {1e10, 1e12, 1e14};

    for (double targetCond : condNumbers) {
        cout << "\nTarget condition number: " << scientific << setprecision(0) << targetCond << endl;

        int n = 12;
        VectorXd sigma(n);
        sigma(0) = 1.0;
        double ratio = pow(targetCond, -1.0 / (n - 1));
        for (int i = 1; i < n; ++i) {
            sigma(i) = sigma(i-1) * ratio;
        }

        MatrixXd U = MatrixXd::Random(n, n).householderQr().householderQ();
        MatrixXd V = MatrixXd::Random(n, n).householderQr().householderQ();
        MatrixXd A = U * sigma.asDiagonal() * V.transpose();

        VectorXd xTrue = VectorXd::Random(n);
        VectorXd b = A * xTrue;

        try {
            VectorXd xGauss = gaussianElimination(A, b);
            double errGauss = (xGauss - xTrue).norm() / xTrue.norm();
            cout << "  Gaussian error: " << scientific << setprecision(2) << errGauss << endl;
        } catch (...) {
            cout << "  Gaussian: failed" << endl;
        }

        try {
            VectorXd xQR = qrDecompositionMethod(A, b);
            double errQR = (xQR - xTrue).norm() / xTrue.norm();
            cout << "  QR error:       " << scientific << setprecision(2) << errQR << endl;
        } catch (...) {
            cout << "  QR: failed" << endl;
        }

        try {
            VectorXd xSVD = svdDecompositionMethod(A, b);
            double errSVD = (xSVD - xTrue).norm() / xTrue.norm();
            cout << "  SVD error:      " << scientific << setprecision(2) << errSVD << endl;
        } catch (...) {
            cout << "  SVD: failed" << endl;
        }
    }
}

// 超定方程组测试
void overdeterminedTest() {
    cout << "\n=== Overdetermined System Test (QR vs SVD) ===" << endl;

    int m = 10, n = 4; // m > n: overdetermined
    MatrixXd A = MatrixXd::Random(m, n);
    VectorXd xTrue = VectorXd::Random(n);
    VectorXd b = A * xTrue + 0.01 * VectorXd::Random(m); // add small noise

    cout << "Overdetermined system: " << m << "x" << n << endl;
    cout << "True solution: " << xTrue.transpose() << endl;

    try {
        VectorXd xQR = qrDecompositionMethod(A, b);
        cout << "QR solution: " << xQR.transpose() << endl;
        cout << "QR residual: " << (A * xQR - b).norm() << endl;
    } catch (const exception& e) {
        cout << "QR failed: " << e.what() << endl;
    }

    try {
        VectorXd xSVD = svdDecompositionMethod(A, b);
        cout << "SVD solution: " << xSVD.transpose() << endl;
        cout << "SVD residual: " << (A * xSVD - b).norm() << endl;
    } catch (const exception& e) {
        cout << "SVD failed: " << e.what() << endl;
    }
}

int main() {
    cout << "QR vs SVD vs Gaussian: Numerical Methods Comparison" << endl;
    cout << "===================================================" << endl;

    testMethods();
    performanceTest();
    stabilityTest();
    overdeterminedTest();

    cout << "\n=== Summary of Method Characteristics ===" << endl;
    cout << "1. Gaussian Elimination:" << endl;
    cout << "   + O(n^3), efficient for well-conditioned systems" << endl;
    cout << "   - Suffers from numerical instability for ill-conditioned matrices" << endl;

    cout << "\n2. QR Decomposition:" << endl;
    cout << "   + Excellent stability, handles overdetermined systems via least squares" << endl;
    cout << "   - Slightly more expensive than Gaussian (but more robust)" << endl;

    cout << "\n3. SVD Decomposition:" << endl;
    cout << "   + Most stable, handles rank-deficient and overdetermined systems" << endl;
    cout << "   - Most computationally expensive, but most reliable for ill-conditioned problems" << endl;

    return 0;
}