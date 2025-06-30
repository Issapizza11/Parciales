#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>

// Constantes físicas y parámetros de simulación
const int Nx = 500;
const int Ny = 500;
const int steps = 10000;
const double Lx = 1.0;
const double Ly = 1.0;
const double dx = Lx / Nx;
const double dy = Ly / Ny;
const double dt = 0.0001;

const double alpha_g = 0.01;
const double alpha_s = 0.005;
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

// Vectores de velocidad y pesos
const std::array<int, 9> ex = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const std::array<int, 9> ey = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const std::array<double, 9> w = {4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36};

// Función de equilibrio
double feq(double T, int k, double ux, double uy)
{
    double eu = ex[k] * ux + ey[k] * uy;
    double uu = ux * ux + uy * uy;
    return w[k] * T * (1 + eu / cs2 + 0.5 * (eu * eu) / (cs2 * cs2) - 0.5 * uu / cs2);
}

// Término fuente
double source_term(int i, int j)
{
    double x = i * dx;
    double y = j * dy;

    double S1 = ((x >= 0.2 && x <= 0.3) && (y >= 0.4 && y <= 0.5)) ? S_value : 0.0;
    double S2 = ((x >= 0.7 && x <= 0.8) && (y >= 0.6 && y <= 0.7)) ? S_value : 0.0;

    return (S1 + S2) * dt;
}

// Estructura de datos optimizada
struct LatticeData
{
    std::vector<std::array<double, 9>> f;
    std::vector<std::array<double, 9>> g;
    std::vector<double> Tg;
    std::vector<double> Ts;

    LatticeData() : f(Nx * Ny), g(Nx * Ny), Tg(Nx * Ny), Ts(Nx * Ny) {}
};

int main()
{
    LatticeData data;
    LatticeData temp_data;

// Inicialización
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            int idx = i * Ny + j;
            double T0 = T_in * (1.0 - (i * dx) / Lx);
            data.Tg[idx] = data.Ts[idx] = T0;

            for (int k = 0; k < 9; ++k)
            {
                data.f[idx][k] = feq(T0, k, u_lb_x, u_lb_y);
                data.g[idx][k] = feq(T0, k, 0.0, 0.0);
            }
        }
    }

    // Bucle temporal principal
    for (int t = 0; t < steps; ++t)
    {
// Fase de Colisión
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < Nx * Ny; ++idx)
        {
            double Tgas = 0.0, Tsol = 0.0;

            // Sumamos todas las direcciones
            for (int k = 0; k < 9; ++k)
            {
                Tgas += data.f[idx][k];
                Tsol += data.g[idx][k];
            }

            data.Tg[idx] = Tgas;
            data.Ts[idx] = Tsol;

            int i = idx / Ny, j = idx % Ny;
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

                data.f[idx][k] = data.f[idx][k] - (data.f[idx][k] - feqg) / tau_g + Fkg;
                data.g[idx][k] = data.g[idx][k] - (data.g[idx][k] - feqs) / tau_s + Fks;
            }
        }

// Fase de Streaming
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < Nx * Ny; ++idx)
        {
            int i = idx / Ny, j = idx % Ny;
            for (int k = 0; k < 9; ++k)
            {
                int ni = i - ex[k];
                int nj = j - ey[k];
                if (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny)
                {
                    int src_idx = ni * Ny + nj;
                    temp_data.f[idx][k] = data.f[src_idx][k];
                    temp_data.g[idx][k] = data.g[src_idx][k];
                }
                else
                {
                    // Condición de rebote (simplificada)
                    temp_data.f[idx][k] = data.f[idx][k];
                    temp_data.g[idx][k] = data.g[idx][k];
                }
            }
        }
        std::swap(data.f, temp_data.f);
        std::swap(data.g, temp_data.g);

// Condiciones de frontera
#pragma omp parallel for schedule(static)
        for (int j = 0; j < Ny; ++j)
        {
            int idx = j; // i=0
            for (int k = 0; k < 9; ++k)
            {
                if (ex[k] > 0)
                {
                    data.f[idx][k] = feq(T_in, k, u_lb_x, u_lb_y);
                }
                data.g[idx][k] = feq(T_in, k, 0.0, 0.0);
            }
        }
    }

    // Escritura de resultados
    std::ofstream file("lbm2d_output.csv");
    file << "x,y,Tg,Ts\n";
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            int idx = i * Ny + j;
            file << i * dx << "," << j * dy << "," << data.Tg[idx] << "," << data.Ts[idx] << "\n";
        }
    }

    std::cout << "Simulación 2D LBM completada. Resultados en 'lbm2d_output.csv'.\n";
    return 0;
}