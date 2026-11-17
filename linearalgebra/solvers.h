#ifndef _SOLVERS_
#define _SOLVERS_

#include <vector>
#include <stdexcept>
#include "linalg.h"
#include "function.h"
//#include "sparse.h"

template<typename T>
class Matrix;

template<typename T>
class vec;

template<typename T>
struct SparseMatrix;


// Function declarations
template<typename T>
vec<T> cg(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance);

template<typename T>
vec<T> cg_sparse(SparseMatrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance);

template<typename T>
vec<T> pcg(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance);

template<typename T>
vec<T> direct(vec<T> rho, int n);

template<typename T>
vec<T> direct(vec<T> a, vec<T> b, vec<T> c, vec<T> d);

template<typename T>
vec<T> direct(vec<T> rhs, int n, bool neumann_end);

template<typename T>
vec<T> gausselimination(Matrix<T> &A, vec<T> &b);

template<typename T>
vec<T> UpperTriangularSolve(Matrix<T> &A, vec<T> &b, int size);

template<typename T>
vec<T> gmres(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration = 10000, double tolerance = 1e-6, int restart = 30);

template<typename T>
vec<T> gmres_sparse(SparseMatrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration = 1000, double tolerance = 1e-6, int restart = 30);

//==========

// Conjugate Gradient method
template<typename T>
vec<T> cg(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance)
{
    if(symmetrycheck(A))
    {
        vec<T> r0 = b - A * x;
        vec<T> p0 = r0;

        for (int i = 0; i < std::min(b.getSize(), max_iteration); ++i)
        {
            if (r0.norm() < tolerance)
            {
                break;
            }

            vec<T> Ap0 = A * p0;
            double r02 = r0 * r0;
            double pAp = p0 * Ap0;
            double alpha = r02 / pAp;
            vec<T> x1 = x + p0 * alpha;
            vec<T> r1 = r0 - Ap0 * alpha;
            double r12 = r1 * r1;
            double beta = r12 / r02;
            vec<T> p1 = r1 + p0 * beta;

            x = x1;
            r0 = r1;
            p0 = p1;
        }
    }
    else
    {
        throw std::invalid_argument("fatal error! the Matrix is not symmetric,pcg/cg solver requre a symmetric matrix ");
    }

    return x;
}

///sparse matrix cg
template<typename T>
vec<T> cg_sparse(SparseMatrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance)
{
    vec<T> r0 = b - spMVmul(A, x);
    vec<T> p0 = r0;

    for (int i = 0; i < std::min(b.getSize(), max_iteration); ++i)
    {
        if (r0.norm() < tolerance)
        {
            break;
        }

        vec<T> Ap0 = spMVmul(A, p0);
        double r02 = r0 * r0;
        double pAp = p0 * Ap0;
        double alpha = r02 / pAp;
        vec<T> x1 = x + p0 * alpha;
        vec<T> r1 = r0 - Ap0 * alpha;
        double r12 = r1 * r1;
        double beta = r12 / r02;
        vec<T> p1 = r1 + p0 * beta;

        x = x1;
        r0 = r1;
        p0 = p1;
    }

    return x;
}


////
// Preconditioned Conjugate Gradient method
template<typename T>
vec<T> pcg(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance)
{
    vec<T> r = b - A * x;
    Matrix<T> M_inv = diagonalPreconditioner(A);
    vec<T> w = M_inv * r;
    vec<T> v = w;
    double alpha = w * r;

    for (int iter = 0; iter < std::min(A.row * A.col, max_iteration); ++iter)
    {
        if (r.norm() < tolerance)
        {
            break;
        }

        vec<T> u = A * v;
        double t = alpha / (v * u);
        x = x + v * t;
        r = r - u * t;
        w = M_inv * r;
        double beta = w * r;
        double s = beta / alpha;
        v = w + v * s;
        alpha = beta;
    }

    return x;
}

// Direct solution using tridiagonal matrix solver
template<typename T>
vec<T> direct(vec<T> rho, int n)
{
    vec<T> a(n), b(n), c(n), x(n);

    for (int i = 1; i < n - 1; ++i)
    {
        a(i) = 1;
        b(i) = -2;
        c(i) = 1;
    }

    a(0) = 0;
    b(0) = 1;
    c(0) = 0;
    a(n - 1) = 0;
    b(n - 1) = 1;
    c(n - 1) = 0;

    x(0) = rho(0);
    x(n - 1) = rho(n - 1);

    for (int i = 1; i < n - 1; ++i)
    {
        x(i) = rho(i);
    }

    c(0) /= b(0);
    x(0) /= b(0);

    for (int i = 1; i < n; ++i)
    {
        double id = (b(i) - c(i - 1) * a(i));
        c(i) /= id;
        x(i) = (x(i) - x(i - 1) * a(i)) / id;
    }

    for (int i = n - 2; i >= 0; --i)
    {
        x(i) = x(i) - c(i) * x(i + 1);
    }

    return x;
}

template<typename T>
vec<T> gausselimination(Matrix<T> &A,vec<T> &b)
{
    int row = A.row;
    int col = A.col;
    double c;

    // Forward Elimination
    for(int i = 0; i < row-1; i++)
    {
        for(int k = i+1; k < row; k++)
        {
            c = A(k,i) / A(i,i);
            for(int j = 0; j < col; j++)
            {
                A(k,j) -= c * A(i,j);
            }
            b(k) -= c * b(i);
        }
    }
    vec<T> x(b.getSize());
    // Back Substitution
    for(int i = row-1; i >= 0; i--)
    {
        x(i) = b(i);
        for(int j = i+1; j < row; j++)
        {
            x(i) -= A(i,j) * x(j);
        }
        //x(i) /= A(i,i);
    }

    return x;
}




// Overloaded direct solver for a GENERAL tridiagonal system Ax = d
//
//   a[i] = subdiagonal   (a[0] is unused, first used entry is a[1])
//   b[i] = main diagonal
//   c[i] = superdiagonal (c[n-1] is unused)
//   d[i] = right-hand side
//
// Algorithm: Thomas algorithm (TDMA) — O(n)
//
//   Forward sweep eliminates 'a' by row-reducing downward:
//     factor  = a[i] / b[i-1]
//     b[i]   -= factor * c[i-1]        <- update diagonal
//     d[i]   -= factor * d[i-1]        <- update RHS
//
//   Back substitution solves from the bottom up:
//     x[n-1] = d[n-1] / b[n-1]
//     x[i]   = (d[i] - c[i]*x[i+1]) / b[i]
template<typename T>
vec<T> direct(vec<T> a, vec<T> b, vec<T> c, vec<T> d)
{
    int n = d.getSize();

    if (a.getSize() != n || b.getSize() != n || c.getSize() != n)
    {
        throw std::invalid_argument("direct (tridiagonal): all vectors must have the same length n");
    }

    // Forward sweep
    // Eliminate subdiagonal 'a' row by row, modifying b and d in-place.
    // We copy b and d so the caller's vectors are unchanged.
    vec<T> b1 = b;
    vec<T> d1 = d;

    for (int i = 1; i < n; ++i)
    {
        T factor = a(i) / b1(i - 1);
        b1(i)   -= factor * c(i - 1);
        d1(i)   -= factor * d1(i - 1);
    }

    // Back substitution 
    vec<T> x(n);
    x(n - 1) = d1(n - 1) / b1(n - 1);

    for (int i = n - 2; i >= 0; --i)
    {
        x(i) = (d1(i) - c(i) * x(i + 1)) / b1(i);
    }

    return x;
}

// Overloaded direct solver where one end is neumann boundary dphi/dx = g
// first row of Ax = b system is phi[0] = v_left
// Consider a ghost cell beyond domain [n-1] at [n] such that
// Ceentral differecne phi[n]-phi[n-1] = g*2*dx
// phi[n]= phi[n-1] + g*2*dx
// Last row : i[n-2]-2*phi[n-1] + phi[n] = -R*rho(n-1)*dx*dx (R is a consnat coming out of plasma scale choosing)
// Last row become  2*phi[n-2] - 2*phi[n-1] = -R*rho(n-1)*dx*dx -2*dx*g
// rhs[n-1] = -R*rho[n-1]*dx^2 - 2*dx*dphi/dx|_L   (R = L^2/LDe^2)
// a(n-1)= 2, b(n-1) =2 , c(n-1) = 0 where a,b,c are matrix element of Ax=b system
template<typename T>
vec<T> direct(vec<T> rhs, int n, bool neumann_end)
{
    vec<T> a(n), b(n), c(n);
    for (int i = 1; i < n - 1; ++i)
    {
         a(i) = 1; 
         b(i) = -2; 
         c(i) = 1;
    }
    a(0) = 0;  b(0) = 1;  c(0) = 0;          // Dirichlet bc
    a(n-1) = 2; b(n-1) = -2; c(n-1) = 0;     // Neumann bc at end
    return direct(a, b, c, rhs);
}


// Upper traingular matrix solver Ax = b
// We have to start at the bottom and then climb back to top
// general formula x_i = b_i - (\Sum_j A(i,j)*x(j))
// i index run form (b.size-1) to 0 
// j index runs from i+1 to (b.size-1)
template<typename T>
vec<T> UpperTriangularSolve(Matrix<T> &A, vec<T> &b, int size)
{
    vec<T> x(size);

    for (int i = size - 1; i >= 0; --i)
    {
        T sum = 0;
        for (int j = i + 1; j < size; j++)
        {
            sum += A(i, j) * x(j);
        }
        x(i) = (b(i) - sum) / A(i, i);
    }

    return x;
}

// GMRES (Generalized Minimal Residual) - Dense Matrix version
// Uses Arnoldi iteration with Givens rotations to solve Ax = b
// for non-symmetric systems. Supports restarting every `restart`
// iterations to control memory usage.

template<typename T>
vec<T> gmres(Matrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance, int restart)
{
    int n = b.getSize();
    //number of krylov space basis 
    int m = restart;

    // MATRIX in where each clumn stote one vector v0,v1,v3...
    // There vector form krylov spaces K_m = spane{r0,Ar0,A2r0,......}
    // Matrix size row = n as vector size is n and column index is m+1
    // column 0      →  v0   (filled before the loop, it is r/beta)
    // column 1      →  v1   (filled at end of j=0)
    // column 2      →  v2   (filled at end of j=1)
    //  ...
    // column m →  v(m)  (filled at end of j=m-1)
    // that is why need column index of  V matrix m+1

    //allocate heap outside loops if possible
    Matrix<T> V(n,m+1);
    Matrix<T> H(m+1,m);

    // Right-hand side of the residue minimiing problem: |e1 * beta - Vy|
    vec<T> g(m + 1);

    //givens rotation
    vec<T> cs(m);
    vec<T> sn(m);

    for(int iter = 0; iter < max_iteration; ++iter)
    {
        vec<T> r = b - A * x;
        double beta = r.norm();
        double residual = beta;
        
        if(beta <tolerance)
        {
            return x;
        }

        g(0) = beta;
        // Store first basis vector v0 = r / beta
        for (int i = 0; i < n; ++i)
        {
            V(i, 0) = r(i) / beta;
        }
        
        // Generate upto m order krylov subspace elements
        int j = 0;
        for(j = 0 ; j < m; ++j)
        {
            //arnoldi steps
            vec<T> vj(n);

            for (int i = 0; i < n; ++i)
            {
                vj(i) = V(i, j);
            }
                
            //this is the step bacause of which krylov space expnad after each loop counter from j to m-1
            // j = 0 generates r1 = A*r0 
            // j = 1 generates r2 = A*r1 = A^2 * r0
            // so on each iteration expands kyylop subspaces and here we capped at m
            vec<T> w = A*vj;
            
            // Modified Gram-Schmidt orthogonalisation
            for (int i = 0; i <= j; ++i)
            {
                //we have to normalize each new w such thta it orthonormal to rest of vj={v0,v1....}
                // whcih are craeted before this w in previous iterations.
                // loop runs from 0 to j
                // As j = 0 means one 1 basis exist till now
                // j = 1 means two basis are created/exist
                // j = 2 mean three basis are already created or exist
                vec<T> vi(n);
                //extract vectors of V matrix 
                for (int k = 0; k < n; ++k)
                {   
                    vi(k) = V(k, i);
                }

                H(i,j) = w*vi;
                //calculate amound of vk in w
                vec<T> scaled  = vi*H(i,j);
                //remove that amound from w to make it indepedned from others
                w = w - scaled; 
                //w = w - vi*H(i,j);   
            }

            double w_norm = w.norm();
            H(j+1,j) = w_norm;

            // Store newly constructed basis vector in V matrix if bnorm is non zero or donot check
            //if (w_norm > 1e-14)
            //{
            for (int i = 0; i < n; ++i)
            {
                V(i, j + 1) = w(i) / w_norm;
            }
                    
            //}

            //givens rotaion to make H matrix complely upper traingular
            //Apply previous Givens rotations to new column of H
            // at j = 0 one column of H created so we thsi previous rotation 
            // loop skips and we then apply tptaion to make off diignola belwo element zerp
            // but as j = 1 one new column if added to H so to make things consinst we apply previous 
            // rotation to this
            for (int i = 0; i < j; ++i)
            {
                double h_ij   = H(i, j);
                //element below diagonal which we have to make zero
                double h_i1j  = H(i + 1, j);
                H(i, j) =  cs(i) * h_ij + sn(i) * h_i1j;
                H(i + 1, j) = -sn(i) * h_ij + cs(i) * h_i1j;
            }

            // Compute new Givens rotation to zero out H(j+1, j)
            //temp varibale
            double hjj   = H(j, j);
            double hj1j  = H(j + 1, j);
            double denom = std::sqrt(hjj * hjj + hj1j * hj1j);

            
            cs(j) = hjj  / denom;
            sn(j) = hj1j / denom;
            

            // Apply rotation to H and g
            H(j, j) = cs(j) * hjj + sn(j) * hj1j;
            H(j + 1,j) = 0.0;

            double gj   = g(j);
            g(j)     =  cs(j) * gj;
            g(j + 1) = -sn(j) * gj;

            residual = std::abs(g(j + 1));

            if (residual < tolerance)
            {
                //H.display();
                ++j;
                break;
            }
        }

            //now solve Hy = g to find step size y_step thta minimizes te residue
            vec<T> y_step = UpperTriangularSolve(H,g,j);

            for (int i = 0; i < j; ++i)
            {
                for (int k = 0; k < n; ++k)
                {
                    // this is the solution here y_step denotes how much we move in direction of
                    // kryolov space to find actual solution
                    x(k) = x(k) + V(k, i) * y_step(i);
                }
                    
            }

            if (residual < tolerance)
                return x;
        }

    return x;
    
}


template<typename T>
vec<T> gmres_sparse(SparseMatrix<T> &A, vec<T> &x, vec<T> &b, int max_iteration, double tolerance, int restart)
{
    int n = b.getSize();
    int m = restart;

    Matrix<T> V(n,m+1);
    Matrix<T> H(m+1,m);

    vec<T> g(m + 1);
    vec<T> cs(m);
    vec<T> sn(m);

    for(int iter = 0; iter < max_iteration; ++iter)
    {
        vec<T> r = b - spMVmul(A, x);
        double beta = r.norm();
        double residual = beta;
        
        if(beta < tolerance)
        {
            return x;
        }

        g(0) = beta;

        for (int i = 0; i < n; ++i)
        {
            V(i, 0) = r(i) / beta;
        }
        
        int j = 0;
        for(j = 0 ; j < m; ++j)
        {
            vec<T> vj(n);

            for (int i = 0; i < n; ++i)
            {
                vj(i) = V(i, j);
            }
                
            vec<T> w = spMVmul(A, vj);
            
            for (int i = 0; i <= j; ++i)
            {
                vec<T> vi(n);

                for (int k = 0; k < n; ++k)
                {   
                    vi(k) = V(k, i);
                }

                H(i,j) = w*vi;
                vec<T> scaled  = vi*H(i,j);
                w = w - scaled; 
            }

            double w_norm = w.norm();
            H(j+1,j) = w_norm;

            for (int i = 0; i < n; ++i)
            {
                V(i, j + 1) = w(i) / w_norm;
            }

            for (int i = 0; i < j; ++i)
            {
                double h_ij   = H(i, j);
                double h_i1j  = H(i + 1, j);

                H(i, j)     =  cs(i) * h_ij + sn(i) * h_i1j;
                H(i + 1, j) = -sn(i) * h_ij + cs(i) * h_i1j;
            }

            double hjj   = H(j, j);
            double hj1j  = H(j + 1, j);
            double denom = std::sqrt(hjj * hjj + hj1j * hj1j);

            cs(j) = hjj  / denom;
            sn(j) = hj1j / denom;

            H(j, j) = cs(j) * hjj + sn(j) * hj1j;
            H(j + 1,j) = 0.0;

            double gj   = g(j);
            g(j)     =  cs(j) * gj;
            g(j + 1) = -sn(j) * gj;

            residual = std::abs(g(j + 1));

            if (residual < tolerance)
            {
                ++j;
                break;
            }
        }

        vec<T> y_step = UpperTriangularSolve(H,g,j);

        for (int i = 0; i < j; ++i)
        {
            for (int k = 0; k < n; ++k)
            {
                x(k) = x(k) + V(k, i) * y_step(i);
            }
        }

        if (residual < tolerance)
            return x;
    }

    return x;
}




#endif // _SOLVERS_
