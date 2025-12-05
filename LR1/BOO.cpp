#include <iostream>
#include <cmath>
using namespace std;

double func(double x) {
    double f = 1.4 * x + exp(abs(x - 2));
    return f;
}

double DOP(double e) {
    double a, b, x0, x1, x2, x3, x4, y1, y2, y3, yl, min;
    a = 0;
    b = 2;
    double N = 1;

    x2 = (a + b) / 2;
    y2 = func(x2);

    while ((b - a) > (2 * e)) {
        x1 = (a + x2) / 2;
        y1 = func(x1);
        x3 = (x2 + b) / 2;
        y3 = func(x3);
        N += 2;

        x0 = a;
        x4 = b;

        int min_index = 1;
        min = y1;

        if (y2 < min) {
            min = y2;
            min_index = 2;
        }
        if (y3 < min) {
            min = y3;
            min_index = 3;
        }

        if (min_index == 1) {
            a = x0;
            b = x2;
            x2 = x1;
            y2 = y1;
        }
        else if (min_index == 2) {
            a = x1;
            b = x3;
        }
        else {
            a = x2;
            b = x4;
            x2 = x3;
            y2 = y3;
        }
    }


    cout << "x min = " << x2 << " y min = " << y2 << " Кол-во экспериментов = " << N << " Точность = " << e << endl;
    return 0;
}

double ZS(double e) {
    double a, b, lambda, N, x1, y1, x2, y2;

    lambda = (1 + sqrt(5)) / 2;
    a = 0;
    b = 2;
    N = 2;

    x1 = b - (b - a) / lambda;
    x2 = a + (b - a) / lambda;
    y1 = func(x1);
    y2 = func(x2);

    while ((b - a) > e) {
        if (y1 <= y2) {
            b = x2;
            x2 = x1;
            y2 = y1;
            x1 = a + b - x2;
            y1 = func(x1);
        } else {
            a = x1;
            x1 = x2;
            y1 = y2;
            x2 = a + b - x1;
            y2 = func(x2);
        }
        N += 1;
    }

    cout << "x min = " << a << " y min = " << y1 << " Кол-во экспериментов = " << N << " Точность = " << e << endl;
    return 0;
}

int main() {
    setlocale(LC_ALL, "Russian");
    double E1 = 0.5;
    double E2 = 0.5;
    cout << "Метод деления отрезка пополам\n";
    for (int i = 0; i < 3; i++) {
        DOP(E1);
        E1 /= 10;
    }
    cout << endl;
    cout << "Метод золотого сечения\n";
    for (int i = 0; i < 3; i++) {
        ZS(E2);
        E2 /= 10;
    }
    return 0;
}
