#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "slap.h"
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct RunResult
{
    int n;
    std::string solver;
    double time_ms;
    double residual;
    double max_err;
    double mean_err;
};

double elapsed_ms(Clock::time_point t0)
{
    return Ms(Clock::now() - t0).count();
}

Matrix<double> build_matrix(int n)
{
    Matrix<double> A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A(i, i) = 4.0;
        if (i > 0)
        {
            A(i, i - 1) = -1.0;
            A(i - 1, i) = -1.0;
        }
    }
    return A;
}

vec<double> build_exact_solution(int n)
{
    vec<double> x_exact(n);
    const double pi = std::acos(-1.0);
    for (int i = 0; i < n; ++i)
    {
        x_exact(i) = std::sin((i + 1) * pi / (n + 1));
    }
    return x_exact;
}

double residual_norm(Matrix<double> &A, vec<double> &x, vec<double> &b)
{
    vec<double> r = b - A * x;
    return r.norm();
}

double max_abs_error(vec<double> &x, vec<double> &x_exact)
{
    double e = 0.0;
    for (int i = 0; i < x.getSize(); ++i)
    {
        const double d = std::fabs(x(i) - x_exact(i));
        if (d > e)
        {
            e = d;
        }
    }
    return e;
}

double mean_abs_error(vec<double> &x, vec<double> &x_exact)
{
    double s = 0.0;
    for (int i = 0; i < x.getSize(); ++i)
    {
        s += std::fabs(x(i) - x_exact(i));
    }
    return s / x.getSize();
}

RunResult run_cg(int n, Matrix<double> &A, vec<double> &b, vec<double> &x_exact)
{
    vec<double> x0(n);
    auto t0 = Clock::now();
    vec<double> x = cg(A, x0, b, 50000, 1e-12);
    const double time_ms = elapsed_ms(t0);

    RunResult r;
    r.n = n;
    r.solver = "CG";
    r.time_ms = time_ms;
    r.residual = residual_norm(A, x, b);
    r.max_err = max_abs_error(x, x_exact);
    r.mean_err = mean_abs_error(x, x_exact);
    return r;
}

RunResult run_pcg(int n, Matrix<double> &A, vec<double> &b, vec<double> &x_exact)
{
    vec<double> x0(n);
    auto t0 = Clock::now();
    vec<double> x = pcg(A, x0, b, 50000, 1e-12);
    const double time_ms = elapsed_ms(t0);

    RunResult r;
    r.n = n;
    r.solver = "PCG";
    r.time_ms = time_ms;
    r.residual = residual_norm(A, x, b);
    r.max_err = max_abs_error(x, x_exact);
    r.mean_err = mean_abs_error(x, x_exact);
    return r;
}

void print_result(const RunResult &r)
{
    std::cout << std::left << std::setw(6) << r.solver
              << std::right << std::fixed << std::setprecision(3) << std::setw(12) << r.time_ms
              << std::scientific << std::setprecision(4) << std::setw(16) << r.residual
              << std::setw(16) << r.max_err
              << std::setw(16) << r.mean_err << "\n";
}

int main()
{
    omp_set_num_threads(1);
    std::cout << "CG vs PCG (known solution comparison)\n";
    std::cout << "A(i,i)=4, A(i,i-1)=A(i-1,i)=-1, x_exact(i)=sin((i+1)*pi/(n+1))\n";
    std::cout << "tol=1e-12, max_iter=50000\n\n";

    const std::vector<int> sizes = {10000};

    std::cout << std::left << std::setw(8) << "n"
              << std::setw(6) << "solver"
              << std::right << std::setw(12) << "time(ms)"
              << std::setw(16) << "residual"
              << std::setw(16) << "max_err"
              << std::setw(16) << "mean_err" << "\n";
    std::cout << std::string(74, '-') << "\n";

    for (int n : sizes)
    {
        Matrix<double> A = build_matrix(n);
        vec<double> x_exact = build_exact_solution(n);
        vec<double> b = A * x_exact;

        RunResult cg_r = run_cg(n, A, b, x_exact);
        RunResult pcg_r = run_pcg(n, A, b, x_exact);

        std::cout << std::left << std::setw(8) << n;
        print_result(cg_r);
        std::cout << std::left << std::setw(8) << "";
        print_result(pcg_r);

        const double speedup = cg_r.time_ms / pcg_r.time_ms;
        std::cout << std::left << std::setw(8) << "" << "speedup(CG/PCG): "
                  << std::fixed << std::setprecision(2) << speedup << "x\n";
        std::cout << std::string(74, '-') << "\n";
    }

    return 0;
}
