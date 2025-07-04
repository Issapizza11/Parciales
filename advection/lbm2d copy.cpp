#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <cstring> // Para strcmp

const int Nx = 100;
const int Ny = 100;
const int steps = 10000;
const double Lx = 1.0, Ly = 1.0;
const double dx = Lx / Nx;
const double dy = Ly / Ny;
const double dt = 0.0001;

const double alpha_g = 0.01, alpha_s = 0.005;
const double u_phys_x = 0.1;
const double u_phys_y = 0.05;
const double hv_phys = 5.0;
const double T_in = 0.0;
const double S_value = 315.0;

const double cs2 = 1.0 / 3.0;

const double u_lb_x = u_phys_x * dt / dx;
const double u_lb_y = u_phys_y * dt / dy;
const double hv_lb = hv_phys * dt;
const double tau_g = 0.5 + alpha_g * dt / (cs2 * dx * dx);
const double tau_s = 0.5 + alpha_s * dt / (cs2 * dx * dx);

std::array<int, 9> ex = {0, 1, 0, -1, 0, 1, -1, -1, 1};
std::array<int, 9> ey = {0, 0, 1, 0, -1, 1, 1, -1, -1};
std::array<double, 9> w = {4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36};

double feq(double T, int k, double ux, double uy)
{
    double eu = ex[k] * ux + ey[k] * uy;
    double uu = ux * ux + uy * uy;
    return w[k] * T * (1 + eu / cs2 + 0.5 * (eu * eu) / (cs2 * cs2) - 0.5 * uu / cs2);
}

double source_term(int i, int j)
{
    double x = i * dx;
    double y = j * dy;

    double S = ((x >= 0.45 && x <= 0.55) && (y >= 0.45 && y <= 0.55)) ? S_value * dt : 0.0;

    return S;
}

int main(int argc, char *argv[])
{
    bool report_mode = false;
    if (argc > 1 && (strcmp(argv[1], "-r") == 0 || strcmp(argv[1], "--report") == 0))
    {
        report_mode = true;
    }

    std::vector<std::vector<std::array<double, 9>>> f(Nx, std::vector<std::array<double, 9>>(Ny));
    std::vector<std::vector<std::array<double, 9>>> g = f;
    std::vector<std::vector<double>> Tg(Nx, std::vector<double>(Ny));
    std::vector<std::vector<double>> Ts = Tg;

    // Inicialización
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
        {
            double T0 = T_in * (1.0 - (i * dx) / Lx);
            Tg[i][j] = Ts[i][j] = T0;
            for (int k = 0; k < 9; ++k)
            {
                f[i][j][k] = feq(T0, k, u_lb_x, u_lb_y);
                g[i][j][k] = feq(T0, k, 0.0, 0.0);
            }
        }

    double start_time = omp_get_wtime();

    for (int t = 0; t < steps; ++t)
    {
        if (t % (steps / 100) == 0)
        {
            double progress = 100.0 * t / steps;
            std::cout << "\rProgreso: " << std::fixed << std::setprecision(1)
                      << progress << "%" << std::flush;
        }

        // Colisión
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
            {
                double Tgas = 0, Tsol = 0;
                for (int k = 0; k < 9; ++k)
                {
                    Tgas += f[i][j][k];
                    Tsol += g[i][j][k];
                }
                Tg[i][j] = Tgas;
                Ts[i][j] = Tsol;

                double coupling = hv_lb * (Tgas - Tsol);
                double source = source_term(i, j);
                double Sg = -coupling + source;
                double Ss = coupling;

                for (int k = 0; k < 9; ++k)
                {
                    double feqg = feq(Tgas, k, u_lb_x, u_lb_y);
                    double feqs = feq(Tsol, k, 0.0, 0.0);
                    double Fkg = w[k] * Sg * (1 + (ex[k] * u_lb_x + ey[k] * u_lb_y) / cs2);
                    double Fks = w[k] * Ss;

                    f[i][j][k] = f[i][j][k] - (f[i][j][k] - feqg) / tau_g + Fkg;
                    g[i][j][k] = g[i][j][k] - (g[i][j][k] - feqs) / tau_s + Fks;
                }
            }

        // Streaming
        auto f_temp = f, g_temp = g;
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < 9; ++k)
                {
                    int ni = i - ex[k];
                    int nj = j - ey[k];
                    if (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny)
                    {
                        f[i][j][k] = f_temp[ni][nj][k];
                        g[i][j][k] = g_temp[ni][nj][k];
                    }
                }

        // Condición de entrada (i=0): temperatura fija
        for (int j = 0; j < Ny; ++j)
        {
            for (int k = 0; k < 9; ++k)
            {
                if (ex[k] > 0) // dirección hacia adentro
                    f[0][j][k] = feq(T_in, k, u_lb_x, u_lb_y);
                g[0][j][k] = feq(T_in, k, 0.0, 0.0);
            }
        }
    }

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    std::cout << "\nTiempo de ejecución: " << total_time << " segundos\n";

    const double throughput = (static_cast<double>(Nx) * Ny * steps) / total_time / 1e6;
    std::cout << "Throughput: " << throughput << " MLUPs (Million Lattice Updates Per Second)\n";

    if (!report_mode)
    {
        std::ofstream file("lbm2d_output.csv");
        file << "x,y,Tg,Ts\n";
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                file << i * dx << "," << j * dy << "," << Tg[i][j] << "," << Ts[i][j] << "\n";

        std::cout << "Simulación 2D LBM completada. Resultados en 'lbm2d_output.csv'.\n";
    }

    return 0;
}