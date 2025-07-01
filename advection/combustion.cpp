#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <array>
#include <omp.h>

const int Nx = 300, Ny = 300, Q = 9, Ns = 2;
const int steps = 5000;
const double dx = 1.0 / Nx, dy = 1.0 / Ny, dt = 1e-4;
const double cs2 = 1.0 / 3.0;

const std::array<int, Q> ex = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const std::array<int, Q> ey = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const std::array<double, Q> w = {4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9,
                                 1. / 36, 1. / 36, 1. / 36, 1. / 36};

// Parámetros físicos
const double cp = 1.0;
const double lambda = 0.01; // conductividad térmica
const double Q_reac = 3.0;  // calor liberado
const double A = 5000.0;    // prefactor cinético
const double Ea = 1.5;      // energía de activación reducida

struct Cell
{
    double rho = 1.0, T = 1.0;
    std::array<double, Q> f{}, feq{};
    std::array<std::array<double, Q>, Ns> g{}, geq{};
    std::array<double, Ns> Y{0.0, 0.0}; // combustible, producto
    std::array<double, Ns> omega{0.0, 0.0};
};

std::vector<Cell> grid(Nx *Ny), grid_new(Nx *Ny);

inline int idx(int i, int j)
{
    return i * Ny + j;
}

inline double reaction_rate(double T, double Yf)
{
    return A * Yf * std::exp(-Ea / T);
}

void compute_equilibria(Cell &cell, double ux, double uy)
{
    double u2 = ux * ux + uy * uy;
    for (int k = 0; k < Q; ++k)
    {
        double eu = ex[k] * ux + ey[k] * uy;
        double feq_val = w[k] * cell.rho * (1.0 + eu / cs2 + 0.5 * (eu * eu) / (cs2 * cs2) - 0.5 * u2 / cs2);
        cell.feq[k] = feq_val;
        for (int s = 0; s < Ns; ++s)
            cell.geq[s][k] = feq_val * cell.Y[s];
    }
}

void initialize()
{
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
        {
            int id = idx(i, j);
            Cell &c = grid[id];
            c.rho = 1.0;
            c.T = 0.5;
            if (i < Nx / 3)
            {
                c.T = 1.0;
                c.Y[0] = 1.0; // combustible
                c.Y[1] = 0.0; // producto
            }
            else
            {
                c.Y[0] = 0.0;
                c.Y[1] = 1.0;
            }
            compute_equilibria(c, 0.0, 0.0);
            for (int k = 0; k < Q; ++k)
            {
                c.f[k] = c.feq[k];
                for (int s = 0; s < Ns; ++s)
                    c.g[s][k] = c.geq[s][k];
            }
        }
}

void collide_and_stream()
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            int id = idx(i, j);
            Cell &c = grid[id];

            double ux = 0.0, uy = 0.0, rho = 0.0;
            for (int k = 0; k < Q; ++k)
            {
                rho += c.f[k];
                ux += c.f[k] * ex[k];
                uy += c.f[k] * ey[k];
            }

            c.rho = rho;
            ux /= rho;
            uy /= rho;

            for (int s = 0; s < Ns; ++s)
            {
                c.Y[s] = 0.0;
                for (int k = 0; k < Q; ++k)
                    c.Y[s] += c.g[s][k];
                c.Y[s] /= rho;
            }

            // Reacción química
            double R = reaction_rate(c.T, c.Y[0]);
            c.omega[0] = -R;
            c.omega[1] = +R;

            // Avance temporal: especies y energía
            for (int s = 0; s < Ns; ++s)
                c.Y[s] += dt * c.omega[s];

            double Qdot = Q_reac * c.omega[1];
            c.T += dt * (Qdot / cp); // sin difusión térmica explícita aquí

            compute_equilibria(c, ux, uy);

            // Colisión y streaming
            for (int k = 0; k < Q; ++k)
            {
                int ni = i + ex[k], nj = j + ey[k];
                int nid = (ni >= 0 && ni < Nx && nj >= 0 && nj < Ny) ? idx(ni, nj) : id;

                grid_new[nid].f[k] = c.f[k] - (c.f[k] - c.feq[k]);

                for (int s = 0; s < Ns; ++s)
                    grid_new[nid].g[s][k] = c.g[s][k] - (c.g[s][k] - c.geq[s][k]);
            }
        }
    }
    std::swap(grid, grid_new);
}

void output_csv(const std::string &filename)
{
    std::ofstream f(filename);
    f << "x,y,T,Y_fuel,Y_prod\n";
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
        {
            int id = idx(i, j);
            f << i * dx << "," << j * dy << ","
              << grid[id].T << "," << grid[id].Y[0] << "," << grid[id].Y[1] << "\n";
        }
}

int main()
{
    initialize();
    double t0 = omp_get_wtime();

    for (int t = 0; t < steps; ++t)
    {
        if (t % (steps / 10) == 0)
            std::cout << "Paso " << t << " / " << steps << "\n";
        collide_and_stream();
    }

    double t1 = omp_get_wtime();
    std::cout << "Simulación completada en " << (t1 - t0) << " segundos.\n";
    output_csv("combustion_output.csv");
    return 0;
}
