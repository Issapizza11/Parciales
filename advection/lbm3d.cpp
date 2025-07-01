// Igual definición de parámetros y estructuras que la versión paralela, pero sin #pragma omp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
#include <iomanip>
#include <cstring>

const int Nx = 50, Ny = 50, Nz = 50;
const int steps = 5000;

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

inline double feq(double T, int k, double ux, double uy, double uz)
{
    double eu = ex[k] * ux + ey[k] * uy + ez[k] * uz;
    double uu = ux * ux + uy * uy + uz * uz;
    return w[k] * T * (1.0 + eu * inv_cs2 + 0.5 * eu * eu * inv_cs2_sq - 0.5 * uu * inv_cs2);
}

struct SourceMap
{
    std::vector<double> sources;
    SourceMap() : sources(Nx * Ny * Nz)
    {
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k)
                {
                    double x = i * dx, y = j * dy, z = k * dz;
                    bool in_cube = (x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 && z >= 0.4 && z <= 0.6);
                    sources[i * Ny * Nz + j * Nz + k] = in_cube ? S_value * dt : 0.0;
                }
    }
    inline double get(int i, int j, int k) const
    {
        return sources[i * Ny * Nz + j * Nz + k];
    }
};

struct LatticeData
{
    struct Node
    {
        double f[19], g[19];
        double Tg, Ts;
    };
    std::vector<Node> current, next;
    LatticeData() : current(Nx * Ny * Nz), next(Nx * Ny * Nz) {}
    inline int index(int i, int j, int k) const { return i * Ny * Nz + j * Nz + k; }
    void swap_buffers() { current.swap(next); }
};

int main(int argc, char *argv[])
{
    bool report_mode = argc > 1 && (strcmp(argv[1], "-r") == 0 || strcmp(argv[1], "--report") == 0);

    LatticeData data;
    SourceMap source_map;

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
            {
                int idx = data.index(i, j, k);
                double T0 = T_in * (1.0 - (i * dx) / Lx);
                data.current[idx].Tg = T0;
                data.current[idx].Ts = T0;
                for (int d = 0; d < 19; ++d)
                {
                    data.current[idx].f[d] = feq(T0, d, u_lb_x, u_lb_y, u_lb_z);
                    data.current[idx].g[d] = feq(T0, d, 0.0, 0.0, 0.0);
                }
            }

    double t0 = clock() / (double)CLOCKS_PER_SEC;

    for (int t = 0; t < steps; ++t)
    {
        if (t % (steps / 100) == 0)
        {
            double pct = 100.0 * t / steps;
            std::cout << "\rProgreso: " << std::fixed << std::setprecision(1) << pct << "%" << std::flush;
        }

        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k)
                {
                    int idx = data.index(i, j, k);
                    double f_tmp[19], g_tmp[19];

                    for (int d = 0; d < 19; ++d)
                    {
                        int ni = i - ex[d], nj = j - ey[d], nk = k - ez[d];
                        if (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny && nk >= 0 && nk < Nz)
                        {
                            int src = data.index(ni, nj, nk);
                            f_tmp[d] = data.current[src].f[d];
                            g_tmp[d] = data.current[src].g[d];
                        }
                        else
                        {
                            f_tmp[d] = data.current[idx].f[d];
                            g_tmp[d] = data.current[idx].g[d];
                        }
                    }

                    double Tg = 0.0, Ts = 0.0;
                    for (int d = 0; d < 19; ++d)
                    {
                        Tg += f_tmp[d];
                        Ts += g_tmp[d];
                    }

                    data.next[idx].Tg = Tg;
                    data.next[idx].Ts = Ts;

                    double coupling = hv_lb * (Tg - Ts);
                    double source = source_map.get(i, j, k);
                    double Sg = -coupling + source;
                    double Ss = coupling;

                    for (int d = 0; d < 19; ++d)
                    {
                        double feqg = feq(Tg, d, u_lb_x, u_lb_y, u_lb_z);
                        double feqs = feq(Ts, d, 0.0, 0.0, 0.0);
                        double u_dot_e = ex[d] * u_lb_x + ey[d] * u_lb_y + ez[d] * u_lb_z;
                        double Fkg = w[d] * Sg * (1.0 + u_dot_e * inv_cs2);
                        double Fks = w[d] * Ss;
                        data.next[idx].f[d] = f_tmp[d] - (f_tmp[d] - feqg) * inv_tau_g + Fkg;
                        data.next[idx].g[d] = g_tmp[d] - (g_tmp[d] - feqs) * inv_tau_s + Fks;
                    }
                }

        data.swap_buffers();

        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
            {
                int idx = data.index(0, j, k);
                for (int d = 0; d < 19; ++d)
                {
                    if (ex[d] > 0)
                        data.current[idx].f[d] = feq(T_in, d, u_lb_x, u_lb_y, u_lb_z);
                    data.current[idx].g[d] = feq(T_in, d, 0.0, 0.0, 0.0);
                }
            }
    }

    double t1 = clock() / (double)CLOCKS_PER_SEC;
    std::cout << "\nTiempo de ejecución: " << (t1 - t0) << " segundos\n";

    if (!report_mode)
    {
        std::ofstream file("lbm3d_output.csv");
        file << "x,y,z,Tg,Ts\n";
        file.precision(10);
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k)
                {
                    int idx = data.index(i, j, k);
                    file << i * dx << "," << j * dy << "," << k * dz << ","
                         << data.current[idx].Tg << "," << data.current[idx].Ts << "\n";
                }
        std::cout << "Simulación 3D LBM completada. Resultados en 'lbm3d_output.csv'.\n";
    }

    return 0;
}
