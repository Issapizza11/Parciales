#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <cstring>

// Parámetros de simulación
const int Nx = 200;
const int Ny = 200;
const int steps = 10000;

const double Lx = 1.0, Ly = 1.0;
const double dx = Lx / Nx;
const double dy = Ly / Ny;
const double dt = 0.0001;

const double alpha_g = 0.01, alpha_s = 0.005;
const double u_phys_x = 0.1, u_phys_y = 0.05;
const double hv_phys = 5.0;
const double T_in = 0.0;
const double S_value = 315.0;

const double cs2 = 1.0 / 3.0;
const double inv_cs2 = 1.0 / cs2;
const double inv_cs2_sq = 1.0 / (cs2 * cs2);

const double u_lb_x = u_phys_x * dt / dx;
const double u_lb_y = u_phys_y * dt / dy;
const double hv_lb = hv_phys * dt;

const double tau_g = 0.5 + alpha_g * dt / (cs2 * dx * dx);
const double tau_s = 0.5 + alpha_s * dt / (cs2 * dx * dx);

const double inv_tau_g = 1.0 / tau_g;
const double inv_tau_s = 1.0 / tau_s;

const std::array<int, 9> ex = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const std::array<int, 9> ey = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const std::array<double, 9> w = {4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9,
                                 1. / 36, 1. / 36, 1. / 36, 1. / 36};

// Función de equilibrio
inline double feq(double T, int k, double ux, double uy)
{
    double eu = ex[k] * ux + ey[k] * uy;
    double uu = ux * ux + uy * uy;
    return w[k] * T * (1.0 + eu * inv_cs2 + 0.5 * eu * eu * inv_cs2_sq - 0.5 * uu * inv_cs2);
}

// Mapa de fuente precalculado
struct SourceMap
{
    std::vector<double> sources;

    SourceMap() : sources(Nx * Ny)
    {
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                double x = i * dx;
                double y = j * dy;
                double S = ((x >= 0.45 && x <= 0.55) && (y >= 0.45 && y <= 0.55)) ? S_value * dt : 0.0;
                sources[i * Ny + j] = S;
            }
        }
    }

    inline double get(int i, int j) const
    {
        return sources[i * Ny + j];
    }
};

// Datos del nodo de la red
struct LatticeData
{
    struct Node
    {
        double f[9];
        double g[9];
        double Tg, Ts;
    };

    std::vector<Node> current;
    std::vector<Node> next;

    LatticeData() : current(Nx * Ny), next(Nx * Ny) {}

    inline int index(int i, int j) const { return i * Ny + j; }

    void swap_buffers()
    {
        current.swap(next);
    }
};

int main(int argc, char *argv[])
{
    bool report_mode = false;
    if (argc > 1 && (strcmp(argv[1], "-r") == 0 || strcmp(argv[1], "--report") == 0))
    {
        report_mode = true;
    }

    LatticeData data;
    SourceMap source_map;

    // Inicialización
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            int idx = data.index(i, j);
            double T0 = T_in * (1.0 - (i * dx) / Lx);
            data.current[idx].Tg = T0;
            data.current[idx].Ts = T0;
            for (int k = 0; k < 9; ++k)
            {
                data.current[idx].f[k] = feq(T0, k, u_lb_x, u_lb_y);
                data.current[idx].g[k] = feq(T0, k, 0.0, 0.0);
            }
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

        // Colisión y propagación
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                int idx = data.index(i, j);
                double Tgas = 0.0, Tsol = 0.0;
                for (int k = 0; k < 9; ++k)
                {
                    Tgas += data.current[idx].f[k];
                    Tsol += data.current[idx].g[k];
                }

                data.current[idx].Tg = Tgas;
                data.current[idx].Ts = Tsol;

                double coupling = hv_lb * (Tgas - Tsol);
                double source = source_map.get(i, j);
                double Sg = -coupling + source;
                double Ss = coupling;

                for (int k = 0; k < 9; ++k)
                {
                    double feqg = feq(Tgas, k, u_lb_x, u_lb_y);
                    double feqs = feq(Tsol, k, 0.0, 0.0);

                    double u_dot_e = ex[k] * u_lb_x + ey[k] * u_lb_y;
                    double Fkg = w[k] * Sg * (1.0 + u_dot_e * inv_cs2);
                    double Fks = w[k] * Ss;

                    double f_new = data.current[idx].f[k] - (data.current[idx].f[k] - feqg) * inv_tau_g + Fkg;
                    double g_new = data.current[idx].g[k] - (data.current[idx].g[k] - feqs) * inv_tau_s + Fks;

                    int ni = i + ex[k];
                    int nj = j + ey[k];

                    if (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny)
                    {
                        int dst_idx = data.index(ni, nj);
                        data.next[dst_idx].f[k] = f_new;
                        data.next[dst_idx].g[k] = g_new;
                    }
                    else
                    {
                        data.next[idx].f[k] = f_new;
                        data.next[idx].g[k] = g_new;
                    }
                }
            }
        }

        data.swap_buffers();

        // Condición de entrada (i=0)
        for (int j = 0; j < Ny; ++j)
        {
            int idx = data.index(0, j);
            for (int k = 0; k < 9; ++k)
            {
                if (ex[k] > 0)
                    data.current[idx].f[k] = feq(T_in, k, u_lb_x, u_lb_y);
                data.current[idx].g[k] = feq(T_in, k, 0.0, 0.0);
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
        file.precision(10);

        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                int idx = data.index(i, j);
                file << i * dx << "," << j * dy << ","
                     << data.current[idx].Tg << "," << data.current[idx].Ts << "\n";
            }
        }

        std::cout << "Simulación 2D LBM completada. Resultados en 'lbm2d_output.csv'.\n";
    }

    return 0;
}