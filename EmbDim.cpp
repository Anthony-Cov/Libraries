#include "EmbDim.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>
#include <fstream>
using namespace std;

float Norm(double* x1, double* x2, int l)
{
    int i;
    float s = 0.;
    for (i = 0; i < l; i++)
        s += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    return sqrt(s);
}
int Heaviside(double x1)
{
    if (x1 > 0)
        return 1;
    else
        return 0;
}
int Scale(double* ser, int n)
{
    int i;
    float mi, ma;
    mi = 0.;
    ma = 0.;
    for (i = 0; i < n; i++)
    {
        if (ser[i] < mi)
            mi = ser[i];
        if (ser[i] > ma)
            ma = ser[i];
    }
    for (i = 0; i < n; i++)
    {
        ser[i] -= mi;
        ser[i] /= ma - mi;
    }
    return 0;
}

float MiMa(double** mat, int l, char q) //q = 'i' -> min; q = 'a' ->max
{
    int i,j;
    double extr;
    extr = mat[0][1];
    for (i = 0; i < l; i++)
        for (j = i + 1; j < l; j++)
            if (q == 'i')
            {
                if (mat[i][j] < extr) extr = mat[i][j];
            }
            else
            {
                if (mat[i][j] > extr) extr = mat[i][j];
            }
    return extr;
}

int EmbDim(double* ser, int n)
{
    int i, j, k, l;
    double ls, step, mi, ma, c, dc, d0=0.;
    double x1[nw], x2[nw], cl[21], ll[21];
    double** w;
    double** ro;
    w = new double*[nw];
    ro = new double*[nw];
    for (i=0; i<nw; i++)
    {
       w[i] = new double[nw];
       ro[i] = new double[nw];
    }    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            w[i][j] = 0.;
            ro[i][j] = 0.;
        }
    for (k = 2; k < n/5; k++)
    {
        for (i = 0; i < n - k; i++)         //delay matrix n /2
            for (j = i; j < i + k; j++) 
                w[j - i][i] = ser[j];
        for (i = 0; i < n - k; i++)          //distances for w
            for (j = 0; j < n - k; j++)
            {
                for (l = 0; l < k; l++)
                { 
                    x1[l] = w[l][i];
                    x2[l] = w[l][j];
                }
                ro[i][j] = Norm(x1,x2,k);
            }
        mi = MiMa(ro, n - k, 'i');
        ma = MiMa(ro, n - k, 'a');
        step = (ma - mi) / 20.;
        l = 0;
        for (ls = mi+.01; ls <= ma; ls += step)
        {
            c = 0;
            for (i = 0; i < n - k; i++)
                for (j = i + 1; j < n - k; j++)
                    c += Heaviside(ls - ro[i][j]);
            if (c > 0)
                cl[l] = log(c / (n - k) / (n - k));
            else
                cl[l] = 0;
            ll[l] = log(ls);
            l++;
        }
        dc = (cl[1] - cl[0]) / (ll[1] - ll[0]);
        if (abs(dc - d0) > (ma - mi) / 50.)
            d0 = dc;
        else
        {
            k--;
            dc = d0;
            break;
        } 
//    for (i = 0; i < nw; i++)
//    {
//        delete (w[i]);
//        delete (ro[i]);
//    }
//    delete (w);
//    delete (ro);  
    }
    return k;
}
