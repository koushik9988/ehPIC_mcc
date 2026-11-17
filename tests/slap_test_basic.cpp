#include <iostream>
#include <cmath>
#include "slap.h"

int passed_tests = 0;
int failed_tests = 0;

void print_test(const std::string& name)
{
    std::cout << "\nTEST: " << name << std::endl;
}

void print_result(const std::string& desc, bool ok)
{
    if(ok)
    {
        std::cout << "PASS: " << desc << std::endl;
        passed_tests++;
    }
    else
    {
        std::cout << "FAIL: " << desc << std::endl;
        failed_tests++;
    }
}

bool approx(double a, double b, double tol=1e-9)
{
    return std::abs(a-b) < tol;
}

// ======================================
// MATRIX MULTIPLICATION
// ======================================

void test_matrix_multiply()
{
    print_test("Matrix multiplication");

    Matrix<double> A(2,2);
    Matrix<double> B(2,2);

    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;

    Matrix<double> C = A * B;

    print_result("C(0,0)=19", approx(C(0,0),19));
    print_result("C(0,1)=22", approx(C(0,1),22));
    print_result("C(1,0)=43", approx(C(1,0),43));
    print_result("C(1,1)=50", approx(C(1,1),50));
}

// ======================================
// MATRIX VECTOR MULTIPLICATION
// ======================================

void test_matrix_vector()
{
    print_test("Matrix vector multiplication");

    Matrix<double> A(2,2);
    vec<double> v(2);

    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    v(0)=5;
    v(1)=6;

    vec<double> r = A * v;

    print_result("r(0)=17", approx(r(0),17));
    print_result("r(1)=39", approx(r(1),39));
}

// ======================================
// VECTOR ADDITION
// ======================================

void test_vector_add()
{
    print_test("Vector addition");

    vec<double> a(3), b(3);

    a(0)=1; a(1)=2; a(2)=3;
    b(0)=4; b(1)=5; b(2)=6;

    vec<double> c = a + b;

    print_result("c(0)=5", approx(c(0),5));
    print_result("c(1)=7", approx(c(1),7));
    print_result("c(2)=9", approx(c(2),9));
}

// ======================================
// DOT PRODUCT
// ======================================

void test_dot()
{
    print_test("Dot product");

    vec<double> a(4), b(4);

    a(0)=1; a(1)=2; a(2)=3; a(3)=4;
    b(0)=2; b(1)=3; b(2)=4; b(3)=5;

    double dot = a * b;

    print_result("dot=40", approx(dot,40));
}

// ======================================
// VECTOR NORM
// ======================================

void test_norm()
{
    print_test("Vector norm");

    vec<double> v(3);

    v(0)=3;
    v(1)=4;
    v(2)=0;

    double n = v.norm();

    print_result("norm=5", approx(n,5));
}

// ======================================
// MATRIX ADDITION
// ======================================

void test_matrix_add()
{
    print_test("Matrix addition");

    Matrix<double> A(2,2), B(2,2);

    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;

    Matrix<double> C = A + B;

    print_result("C(0,0)=6", approx(C(0,0),6));
    print_result("C(0,1)=8", approx(C(0,1),8));
    print_result("C(1,0)=10", approx(C(1,0),10));
    print_result("C(1,1)=12", approx(C(1,1),12));
}

// ======================================
// MAIN
// ======================================

int main()
{
    std::cout << "LINEAR ALGEBRA TESTS\n";

    test_matrix_multiply();
    test_matrix_vector();
    test_vector_add();
    test_dot();
    test_norm();
    test_matrix_add();

    std::cout << "\nSUMMARY\n";
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;
    std::cout << "Total : " << passed_tests + failed_tests << std::endl;

    return failed_tests;
}