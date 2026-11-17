#ifndef _MATH_
#define _MATH_

#include "slap.h"
#include <functional>

//trapezoidal rule
inline double trape(vec<double> &field, double h, int n)
{
    double sum = 0.5 * (field(0) + field(n-1)); 

 
    for (int i = 1; i < n-1; ++i)
    {
        sum += field(i);
    }

    return h * sum;
}

//Simpson's 3/8 rule
inline double simpson_38(vec<double> &field, double h, int n)
{
    double sum = field(0) + field(n-1);

    for (int i = 1; i < n-1; ++i)
    {
        if (i % 3 == 0)
        {  
            sum += 2 * field(i);
        }
        else
        {
            sum += 3 * field(i);
        }
    }

    return (3 * h / 8) * sum;
}

//Simpson's 1/3 rule
inline double simpson_13(vec<double> &field, double h, int n)
{
    double sum = field(0) + field(n-1);

    for (int i = 1; i < n-1; ++i)
    {
        if (i % 2 == 0)
        { 
            sum += 2 * field(i);
        }
        else
        {
            sum += 4 * field(i);
        }
    }

    return (h / 3) * sum;
}


inline double bisection(double lb, double ub, std::function<double(double)> expre)
{
    double a = lb;
    double b = ub;
    double c = a;

    for (int iter = 0; iter < 100; ++iter)
    {
        c = (a + b) * 0.5;

        // Check if the middle point is the root or if the interval is small enough
        if (std::abs(expre(c)) < 1e-5) break; 

        if (expre(a) * expre(c) > 0)
        {
            a = c; // Root is in [c, b]
        } else
        {
            b = c; // Root is in [a, c]
        }
    }
    return (a + b) * 0.5;
}

#endif
