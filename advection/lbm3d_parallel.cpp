#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <iomanip>

const int Nx = 50, Ny = 50, Nz = 50;
const int steps = 5000;
const int BLOCK = 16;

const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
const double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;
const double dt = 0.0001;

const double alpha_g = 0.01, alpha_s = 0.005;
const double u_phys_x = 0.1, u_phys_y = 0.05, u_phys_z = 0.0;
const double hv_phys = 5.0;
const double T_in = 0.0;
const double S_value = 315.0;

const double cs2 = 1.0 / 3.0;
const double inv_cs2 = 1.0 / cs2;
const double inv_cs2_sq = 1.0 / (cs2 * cs2);

const double u_lb_x = u_phys_x * dt / dx;
const double u_lb_y = u_phys_y * dt / dy;
const double u_lb_z = u_phys_z * dt / dz;
const double hv_lb = hv_phys * dt;

const double tau_g = 0.5 + alpha_g * dt / (cs2 * dx * dx);
const double tau_s = 0.5 + alpha_s * dt / (cs2 * dx * dx);

const double inv_tau_g = 1.0 / tau_g;
const double inv_tau_s = 1.0 / tau_s;

const std::array<int, 19> ex = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1};
const std::array<int, 19> ey = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
const std::array<int, 19> ez = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1};
const std::array<double, 19> w = {
    1. / 3,
    1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18,
    1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36,
    1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36};

inline int idx(int i, int j, int k) { return i * Ny * Nz + j * Nz + k; }

inline double feq(double T, int d, double ux, double uy, double uz)
{
    double eu = ex[d] * ux + ey[d] * uy + ez[d] * uz;
    double uu = ux * ux + uy * uy + uz * uz;
    return w[d] * T * (1. + eu * inv_cs2 + 0.5 * eu * eu * inv_cs2_sq - 0.5 * uu * inv_cs2);
}

int main(int argc, char *argv[])
{
    bool report_mode = argc > 1 && (strcmp(argv[1], "-r") == 0 || strcmp(argv[1], "--report") == 0);
    const int size = Nx * Ny * Nz;

    std::array<std::vector<double>, 19> f_current, f_next;
    std::array<std::vector<double>, 19> g_current, g_next;
    for (int d = 0; d < 19; ++d)
    {
        f_current[d].resize(size);
        f_next[d].resize(size);
        g_current[d].resize(size);
        g_next[d].resize(size);
    }

    std::vector<double> Tg(size), Ts(size), source(size);

    // Inicialización
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
            {
                int id = idx(i, j, k);
                double x = i * dx, y = j * dy, z = k * dz;
                double T0 = T_in * (1.0 - (i * dx) / Lx);
                Tg[id] = T0;
                Ts[id] = T0;
                bool in_cube = (x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 && z >= 0.4 && z <= 0.6);
                source[id] = in_cube ? S_value * dt : 0.0;
                for (int d = 0; d < 19; ++d)
                {
                    f_current[d][id] = feq(T0, d, u_lb_x, u_lb_y, u_lb_z);
                    g_current[d][id] = feq(T0, d, 0.0, 0.0, 0.0);
                }
            }

    double t0 = omp_get_wtime();

    for (int t = 0; t < steps; ++t)
    {
        if (t % (steps / 100) == 0)
            std::cout << "\rProgreso: " << std::fixed << std::setprecision(1) << 100.0 * t / steps << "%" << std::flush;

// Blocking + pull + colisión
#pragma omp parallel for collapse(3) schedule(static)
        for (int ii = 0; ii < Nx; ii += BLOCK)
            for (int jj = 0; jj < Ny; jj += BLOCK)
                for (int kk = 0; kk < Nz; kk += BLOCK)
                    for (int i = ii; i < std::min(ii + BLOCK, Nx); ++i)
                        for (int j = jj; j < std::min(jj + BLOCK, Ny); ++j)
                            for (int k = kk; k < std::min(kk + BLOCK, Nz); ++k)
                            {
                                int id = idx(i, j, k);
                                double f_tmp[19], g_tmp[19], Tg_local = 0.0, Ts_local = 0.0;

                                for (int d = 0; d < 19; ++d)
                                {
                                    int ni = i - ex[d], nj = j - ey[d], nk = k - ez[d];
                                    int src = (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny && nk >= 0 && nk < Nz)
                                                  ? idx(ni, nj, nk)
                                                  : id;
                                    f_tmp[d] = f_current[d][src];
                                    g_tmp[d] = g_current[d][src];
                                    Tg_local += f_tmp[d];
                                    Ts_local += g_tmp[d];
                                }

                                Tg[id] = Tg_local;
                                Ts[id] = Ts_local;

                                double coupling = hv_lb * (Tg_local - Ts_local);
                                double Sg = -coupling + source[id];
                                double Ss = coupling;

                                for (int d = 0; d < 19; ++d)
                                {
                                    double feqg = feq(Tg_local, d, u_lb_x, u_lb_y, u_lb_z);
                                    double feqs = feq(Ts_local, d, 0, 0, 0);
                                    double u_dot_e = ex[d] * u_lb_x + ey[d] * u_lb_y + ez[d] * u_lb_z;
                                    double Fkg = w[d] * Sg * (1.0 + u_dot_e * inv_cs2);
                                    double Fks = w[d] * Ss;
                                    f_next[d][id] = f_tmp[d] - (f_tmp[d] - feqg) * inv_tau_g + Fkg;
                                    g_next[d][id] = g_tmp[d] - (g_tmp[d] - feqs) * inv_tau_s + Fks;
                                }
                            }

        // Swap buffers
        for (int d = 0; d < 19; ++d)
        {
            std::swap(f_current[d], f_next[d]);
            std::swap(g_current[d], g_next[d]);
        }

// Entrada
#pragma omp parallel for collapse(2)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
            {
                int id = idx(0, j, k);
                for (int d = 0; d < 19; ++d)
                {
                    if (ex[d] > 0)
                        f_current[d][id] = feq(T_in, d, u_lb_x, u_lb_y, u_lb_z);
                    g_current[d][id] = feq(T_in, d, 0.0, 0.0, 0.0);
                }
            }
    }

    double t1 = omp_get_wtime();
    std::cout << "\nTiempo de ejecución: " << (t1 - t0) << " segundos\n";
    uint64_t total_updates = (uint64_t)Nx * Ny * Nz * steps;
    double throughput = total_updates / (t1 - t0) / 1e6;
    std::cout << "Throughput: " << throughput << " MLUPs (Million Lattice Updates Per Second)\n";

    if (!report_mode)
    {
        std::ofstream file("lbm3d_output.csv");
        file << "x,y,z,Tg,Ts\n";
        file.precision(10);
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k)
                {
                    int id = idx(i, j, k);
                    file << i * dx << "," << j * dy << "," << k * dz << ","
                         << Tg[id] << "," << Ts[id] << "\n";
                }
        std::cout << "Simulación 3D LBM completada. Resultados en 'lbm3d_output.csv'.\n";
    }

    return 0;
}
