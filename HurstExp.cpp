#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>

#include "HurstExp.h"

using namespace std;

double Sum(double* ser, int num)
{
    int i;
    double s = 0.;
    for (i = 0; i < num; i++)
        s += ser[i];
    return s;
}

double Mean(double* ser, int num)
{
    int i;
    double s = 0.;
    for (i = 0; i < num; i++)
        s += ser[i];
    if (num > 0)
    {
        s = s / num;
        return s;
    }
    else
        return 0.;
}

double Std(double* ser, int num)
{
    int i;
    double s = 0, m;
    m = Mean(ser, num);
    for (i = 0; i < num; i++)
        s += (ser[i]-m) * (ser[i] - m);
    s = sqrt(s/num);
    return s;
}

int Scale(double* ser, int n)
{
    int i;
    double mi, ma;
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

void MLS(double* x, double* y, double* a, double* b, int n)
{
    double sumx = 0, sumy = 0, sumx2 = 0, sumxy = 0;
    for (int i = 0; i < n; i++) {
        sumx += x[i];
        sumy += y[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
    }
    *a = (n * sumxy - (sumx * sumy)) / (n * sumx2 - sumx * sumx);
    *b = (sumy - *a * sumx) / n;
    return;
}

double HurstExp(double* ser, int num)
{
    int t, i, j;
    double* lgs = new double[num];
    double* h = new double[num - 3];
    double* y;
    double* tau;

    double hexp, m, r, x, s, maxx=0., minx=0.;
    Scale(ser, num);
    lgs[0] = 0;
    for (i = 1; i < num; i++)
        if (ser[i] * ser[i - 1] != 0.)
            lgs[i] = log(ser[i] / ser[i - 1]);
        else
            lgs[i] = 0.;
    for (t = 3; t < num; t++)
    {
        maxx = 0.;
        minx = 0;
        m = Mean(lgs, t);
        s = Std(lgs, t);
        for (i = 1; i < t; i++)
        {
            x = 0.;
            for (j = 0; j < i; j++)
                x += lgs[j] - m;
            if (maxx < x)
                maxx = x;
            if (minx > x)
                minx = x;
        }
        r = maxx - minx;
        if (r*s > 0.)
            h[t - 3] = log(r / s);
        else
            h[t - 3] = 0.;   
    }
    tau = new double[num - 3];
    tau[0] = 0.;
    for (i = 1; i < num - 3; i++)
        tau[i] = log(1.0 * i / 2.0);
    MLS(tau, h, &hexp, &x, num - 3);
    delete[] tau;
    delete[] h;
    delete[] lgs;
    return hexp;
}
