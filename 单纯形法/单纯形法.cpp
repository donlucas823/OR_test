#include <iostream>
#include <vector>
#include <algorithm>
#include <mkl.h>

using namespace std;

const double EPS = 0.001;
const int MAX_ITER = 1000;

// 矩阵-矩阵乘法
void matmat(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, bool trans_A, bool trans_B) {
    int m = trans_A ? A[0].size() : A.size();
    int k = trans_A ? A.size() : A[0].size();
    int n = trans_B ? B.size() : B[0].size();

    CBLAS_TRANSPOSE TransA = trans_A ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE TransB = trans_B ? CblasTrans : CblasNoTrans;

    cblas_dgemm(CblasRowMajor, TransA, TransB, m, n, k, 1.0, A[0].data(), k, B[0].data(), n, 0.0, C[0].data(), n);
}

// 矩阵-向量乘法
void matmat(const vector<vector<double>>& A, const vector<double>& x, vector<double>& b, bool trans_A, bool trans_x = 0) {
    int n = trans_A ? A.size() : A[0].size();
    int m = trans_A ? A[0].size() : A.size();


    CBLAS_TRANSPOSE TransA = trans_A ? CblasTrans : CblasNoTrans;

    // 调用 MKL 进行矩阵-向量乘法
    cblas_dgemv(CblasRowMajor, TransA, m, n, 1.0, A[0].data(), n, x.data(), 1, 0.0, b.data(), 1);
}

// 向量-矩阵乘法
void matmat(const vector<double>& x, const vector<vector<double>>& A, vector<double>& b, bool trans_x, bool trans_A) {
    int m = trans_A ? A.size() : A[0].size();
    int n = trans_A ? A[0].size() : A.size();

    CBLAS_TRANSPOSE TransA = trans_A ? CblasTrans : CblasNoTrans;

    // 调用 MKL 进行向量-矩阵乘法
    cblas_dgemv(CblasRowMajor, TransA, n, m, 1.0, A[0].data(), m, x.data(), 1, 0.0, b.data(), 1);
}   

// 向量-向量乘法（点积）
double matmat(const vector<double>& x, const vector<double>& y, bool trans_x = 0, bool trans_y = 0) {
    int n = x.size();

    // 调用 MKL 进行向量-向量点积
    double result = cblas_ddot(n, x.data(), 1, y.data(), 1);

    return result;
}


// 矩阵求逆
void invertMatrix(const vector<vector<double>>& A, vector<vector<double>>& invA) {
    int n = A.size();
    if (n != A[0].size()) {
        cerr << "Matrix is not square: " << "A: " << n << "x" << A[0].size() << endl;
        exit(1);
    }
    invA = A;
    vector<int> ipiv(n);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, invA[0].data(), n, ipiv.data());
    if (info != 0) {
        cerr << "Error in LU decomposition: " << info << endl;
        exit(1);
    }
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, invA[0].data(), n, ipiv.data());
    if (info != 0) {
        cerr << "Error in matrix inversion: " << info << endl;
        exit(1);
    }
}


// 打印矩阵A的函数
void printMatrix(const vector<vector<double>>& A) {
    for (const auto& row : A) {
        for (double element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
}


int main() {
    // 初始化参数
    vector<int> Basic = { 2, 3, 4 };
    vector<int> Nonbasic = { 0, 1 };

    vector<double> c = { 2, 3, 0, 0, 0 };
    vector<double> c_B = { 0, 0, 0 };
    vector<double> c_N = { 2, 3 };

    vector<vector<double>> A = {
        {1, 2, 1, 0, 0},
        {4, 0, 0, 1, 0},
        {0, 4, 0, 0, 1}
    };

    vector<double> b = { 8, 16, 12 };
    vector<vector<double>> B_inv = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    vector<double> x_opt(5, 0.0);
    double z_opt = 0;

    string solutionStatus;

    int row_num = A.size();
    int column_num = A[0].size();

    vector<double> reducedCost(c_N.size());
    vector<double> temp(row_num);

    // 计算初始reduced cost
    vector<vector<double>> A_N(row_num, vector<double>(Nonbasic.size()));
    for (int i = 0; i < row_num; ++i) {
        for (int j = 0; j < Nonbasic.size(); ++j) {
            A_N[i][j] = A[i][Nonbasic[j]];
        }
    };
    matmat(c_B, B_inv, temp, 0,0);//计算 c_B^trans * B_inv，结果存储在 temp 中
    matmat(temp, A_N, reducedCost, 0, 0); // 计算 c_B^trans * B_inv * A_N，结果存储在 reducedCost 中

    for (int i = 0; i < reducedCost.size(); ++i) {
        double tmp_redu = c_N[i] - reducedCost[i];
        reducedCost[i] = tmp_redu;
    }

    double max_sigma = *max_element(reducedCost.begin(), reducedCost.end());

    int iterNum = 1;
    while (max_sigma >= EPS && iterNum <= MAX_ITER) {
        // 确定入基变量
        int enter_var_index = Nonbasic[max_element(reducedCost.begin(), reducedCost.end()) - reducedCost.begin()];
        cout << "enter_var_index: " << enter_var_index << endl;

        // 确定出基变量
        double min_ratio = 1e20;
        int leave_var_index = -1;
        for (int i = 0; i < row_num; ++i) {
            if (A[i][enter_var_index] > 0) {
                double ratio = b[i] / A[i][enter_var_index];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    leave_var_index = i;
                }
            }
        }
        if (leave_var_index == -1) {
            cerr << "Problem is unbounded." << endl;
            exit(1);
        }

        // 更新基变量
        int leave_var = Basic[leave_var_index];
        cout << "leave_var_index: " << leave_var << endl;
        Basic[leave_var_index] = enter_var_index;
        Nonbasic.erase(remove(Nonbasic.begin(), Nonbasic.end(), enter_var_index), Nonbasic.end());
        Nonbasic.push_back(leave_var);
        sort(Nonbasic.begin(), Nonbasic.end());

        // 更新入基变量所在行
        double pivot = A[leave_var_index][enter_var_index];
        for (int col = 0; col < column_num; ++col) {
            A[leave_var_index][col] /= pivot;
        }
        b[leave_var_index] /= pivot;
       
      

        // 更新其他行
        for (int i = 0; i < row_num; ++i) {
            if (i != leave_var_index) {
                double factor = A[i][enter_var_index];
                for (int col = 0; col < column_num; ++col) {
                    A[i][col] -= factor * A[leave_var_index][col];
                }
                b[i] -= factor * b[leave_var_index];
            }
        }

        // 重新计算B_inv, c_B, c_N
        vector<vector<double>> A_B(row_num, vector<double>(Basic.size()));
        for (int i = 0; i < row_num; ++i) {
            for (int j = 0; j < Basic.size(); ++j) {
                A_B[i][j] = A[i][Basic[j]];
            }
        }


        for (int i = 0; i < Basic.size(); ++i) {
            c_B[i] = c[Basic[i]];
        }
        for (int i = 0; i < Nonbasic.size(); ++i) {
            c_N[i] = c[Nonbasic[i]];
        }

        // 更新A_N
        for (int i = 0; i < row_num; ++i) {
            for (int j = 0; j < Nonbasic.size(); ++j) {
                A_N[i][j] = A[i][Nonbasic[j]];
            }
        }
        

        // 计算reduced cost
        matmat(c_B, B_inv, temp, 0, 0);//计算 c_B^trans * B_inv，结果存储在 temp 中
        matmat(temp, A_N, reducedCost, 0, 0); // 计算 c_B^trans * B_inv * A_N，结果存储在 reducedCost 中
        for (int i = 0; i < reducedCost.size(); ++i) {
            double tmp_redu = c_N[i] - reducedCost[i];
            reducedCost[i] = tmp_redu;
        }
        max_sigma = *max_element(reducedCost.begin(), reducedCost.end());
        iterNum++;


        printMatrix(A);


    }

    // 确定解的状态
    for (int i = 0; i < reducedCost.size(); ++i) {
        if (reducedCost[i] == 0) {
            solutionStatus = "Alternative optimal solution";
            break;
        }
        else {
            solutionStatus = "Optimal";
        }
    }

    // 计算最终解
    vector<double> x_basic(row_num);
    matmat(B_inv, b, x_basic,0,1);
    z_opt = 0;
    for (int i = 0; i < Basic.size(); ++i) {
        x_opt[Basic[i]] = x_basic[i];
        z_opt += c_B[i] * x_basic[i];
    }

    cout << "Simplex iteration: " << iterNum << endl;
    cout << "objective: " << z_opt << endl;
    cout << "optimal solution: ";
    for (const auto& xi : x_opt) {
        cout << xi << " ";
    }
    cout << endl;

    return 0;
}

