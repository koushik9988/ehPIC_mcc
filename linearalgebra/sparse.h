/*
Simple CSR storage for Sparse matrix routine with Sparse matrix and vector multiplication
*/

#ifndef _SPARSE_MATRIX_SYSTEM_
#define _SPARSE_MATRIX_SYSTEM_

#include "linalg.h"

/*
Let us consider an sparse system exmaple 
    0 1 0 3
M = 0 0 8 0 
    1 0 0 7
    0 3 0 0
    *Now nz_value = {1,3,8,1,7,3} index ={0,1,2,3,4,5}
    *column_idx = {1,3,2,0,3,1} index ={0,1,2,3,4,5}
    * both column and nz_value vector share same indexes
    * 
    *row_ptr consint of index of nz_value 
    *1.let say hete 0th row start with 1 and index of in nz_value vector is 0
    *2.1st row start with 8 and index of 8 is 2 becaus it is 3rd element
    *row_ptr = {0,2,3,5,4} last element is just total number of row 
    *which is required such that calculation doenot go out of bounds

Row pointer :
    *now row_ptr(0) = 0 and row_ptr(0+1) = 2 which means that
    *0th row contain value of index in nz_vector from 0 to (2-1 = 1)
    *which are just nz_value(0) = 1 and nz_value(1) = 3
    *Similarly 1st row row_ptr(1) = 1 and row_ptr(1+1) = 2 so
    *it conatin value of index 1 to 2-1 = 1 so just one value which is 8
    *2nd row row_ptr(2) = 3 and row_ptr(2+1) = 5 so conatin vale form nz_value vector
    * with index between 3 and 5-1 = 4 so it conatin two values which are 1 and 7
    * and so on.     
*/

//Sparse matrix CSR (compressed sparse row) storage
template <typename T>
struct SparseMatrix
{
    //vector to store non zero values  
    vec<T> nz_value;
    //column index for nin-zero values;
    vec<int> column_idx;
    //row pointer which store index of the value of nz_value vector
    //of each row and also store total number of nnz in its last index
    vec<int> row_pointer;
    int nnz;
    int col;
    int row;
};


template <typename T>
struct Triplet
{
    int row_idx;
    int col_idx;
    T value;

    //Default constructor
    Triplet() : row_idx(0), col_idx(0), value(0) {}

    Triplet(int row_idx, int col_idx, T value)
    {
        this->row_idx = row_idx;
        this->col_idx = col_idx;
        this->value = value;
    }
};

template <typename T>
SparseMatrix<T> CooToSparse(vec<Triplet<T>>& triplet_list,int nrow, int ncol)
{  
    SparseMatrix<T> SM;

    int count = 0;
    for(auto &t : triplet_list)
    {
        count++;
    }

    //std::cout<<count;

    SM.row = nrow;
    SM.col = ncol;
    SM.nz_value = vec<T>(count);
    SM.column_idx =vec<int>(count);
    SM.row_pointer = vec<int>(nrow+1);

    //int index = 0;
    for(auto &t : triplet_list)
    {
        //SM.nz_value(index) = t.value;
        //SM.column_idx(index) = t.col_idx;
        SM.row_pointer(t.row_idx+1)++;
        //index++;
        //std::cout<<index;
    }
    //cumulative
    for(int i = 0; i < nrow; i++)
    {
        SM.row_pointer(i+1) += SM.row_pointer(i);
    }


    //copy not pass 
    vec<int> counter = SM.row_pointer;
    for(auto &t : triplet_list)
    {
        int j = counter(t.row_idx);
        SM.column_idx(j) = t.col_idx;
        SM.nz_value(j) = t.value;

        //increment for same row if row has multple value
        counter(t.row_idx)++;

    }

    return SM;
}


template <typename T>
SparseMatrix<T> DenseToSparse(Matrix<T> &M)
{
    SparseMatrix<T> SM;
    
    SM.row = M.row;
    SM.col = M.col;
    SM.nnz = 0;

    int count = 0;

    for(int i = 0; i < SM.row; i++)
    {
        for(int j = 0; j < SM.col; j++)
        {
            if(M(i,j) != 0)
                count++;
        }
    }

    SM.nnz = count;

    SM.nz_value   = vec<T>(SM.nnz);
    SM.column_idx = vec<int>(SM.nnz);
    SM.row_pointer= vec<int>(SM.row + 1);

    int nz_id = 0;
    for(int i = 0;i <SM.row; ++i)
    {
        SM.row_pointer(i) = nz_id;
        for(int j = 0; j <SM.col; ++j)
        {
            if(M(i,j) != 0)
            {
                SM.column_idx(nz_id) = j;
                SM.nz_value(nz_id) = M(i,j);
                nz_id++;
            }
        }
    }
    //one extra needed
    SM.row_pointer(SM.row) = nz_id;
    
    return SM;
}


template <typename T>
Matrix<T> SparseToDense(SparseMatrix<T> &SM)
{
    Matrix<T> M(SM.row, SM.col);

    for(int i = 0; i < SM.row; i++)
    {
        // i th row conatin value having index between 
        // start and end (excluding end) in nz_value vector
        int start = SM.row_pointer(i);
        int end   = SM.row_pointer(i+1);

        for(int k = start; k < end; ++k)
        {
            // extract column index for same non zero value index 
            int j = SM.column_idx(k);
            M(i,j) = SM.nz_value(k);
        }
    }

    return M;
}


template <typename T>
vec<T> spMVmul(SparseMatrix<T> &SM, vec<T> &x)
{
    vec<T> result(SM.row);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < SM.row; i++)
    {
        // i th row conatin value having index between 
        // start and end (excluding end) in nz_value vector
        T sum = 0;
        int start = SM.row_pointer(i);
        int end   = SM.row_pointer(i+1);

        for(int k = start; k < end; ++k)
        {
            // extract column index for same non zero value index 
            int j = SM.column_idx(k);
            sum += SM.nz_value(k)*x(j);
        }

        result(i) = sum;
    }

    return result;

}


//Type aliases
using Tripletd = Triplet<double>;
using Tripleti = Triplet<int>;
using Tripletf = Triplet<float>;

using SparseMatrixd = SparseMatrix<double>;
using SparseMatrixf = SparseMatrix<float>;
using SparseMatrixi = SparseMatrix<int>;

#endif

