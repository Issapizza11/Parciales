#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <mpi.h>

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

// Equilibrio con velocidades en x e y
double feq(double T, int k, double ux, double uy)
{
    double eu = ex[k] * ux + ey[k] * uy;
    double uu = ux * ux + uy * uy;
    return w[k] * T * (1 + eu / cs2 + 0.5 * (eu * eu) / (cs2 * cs2) - 0.5 * uu / cs2);
}

// Fuente localizada
double source_term(int i_global, int j)
{
    double x = i_global * dx;
    double y = j * dy;

    double S1 = ((x >= 0.2 && x <= 0.3) && (y >= 0.4 && y <= 0.5)) ? S_value : 0.0;
    double S2 = ((x >= 0.7 && x <= 0.8) && (y >= 0.6 && y <= 0.7)) ? S_value : 0.0;

    return (S1 + S2) * dt;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Descomposición de dominio en dirección X
    int local_Nx = Nx / size;
    int remainder = Nx % size;
    int start_i = rank * local_Nx + std::min(rank, remainder);
    if (rank < remainder)
        local_Nx++;

    // Añadir capas fantasma para comunicación
    int ghost_layers = 1;
    int total_local_Nx = local_Nx + 2 * ghost_layers;

    // Arrays locales con capas fantasma
    std::vector<std::vector<std::array<double, 9>>> f(total_local_Nx, std::vector<std::array<double, 9>>(Ny));
    std::vector<std::vector<std::array<double, 9>>> g = f;
    std::vector<std::vector<double>> Tg(total_local_Nx, std::vector<double>(Ny));
    std::vector<std::vector<double>> Ts = Tg;

    // Identificar vecinos
    int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Inicialización
    for (int i = ghost_layers; i < local_Nx + ghost_layers; ++i)
        for (int j = 0; j < Ny; ++j)
        {
            int global_i = start_i + i - ghost_layers;
            double T0 = T_in * (1.0 - (global_i * dx) / Lx);
            Tg[i][j] = Ts[i][j] = T0;
            for (int k = 0; k < 9; ++k)
            {
                f[i][j][k] = feq(T0, k, u_lb_x, u_lb_y);
                g[i][j][k] = feq(T0, k, 0.0, 0.0);
            }
        }

    // Buffers para comunicación
    std::vector<std::array<double, 9>> send_left(Ny), send_right(Ny);
    std::vector<std::array<double, 9>> recv_left(Ny), recv_right(Ny);
    std::vector<std::array<double, 9>> send_left_g(Ny), send_right_g(Ny);
    std::vector<std::array<double, 9>> recv_left_g(Ny), recv_right_g(Ny);

    for (int t = 0; t < steps; ++t)
    {
        // Intercambio de capas fantasma
        MPI_Request requests[8];
        int req_count = 0;

        // Preparar datos para envío
        for (int j = 0; j < Ny; ++j)
        {
            if (left_neighbor != MPI_PROC_NULL)
            {
                send_left[j] = f[ghost_layers][j];
                send_left_g[j] = g[ghost_layers][j];
            }
            if (right_neighbor != MPI_PROC_NULL)
            {
                send_right[j] = f[local_Nx + ghost_layers - 1][j];
                send_right_g[j] = g[local_Nx + ghost_layers - 1][j];
            }
        }

        // Comunicación no bloqueante
        if (left_neighbor != MPI_PROC_NULL)
        {
            MPI_Isend(send_left.data(), Ny * 9, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(recv_left.data(), Ny * 9, MPI_DOUBLE, left_neighbor, 1, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Isend(send_left_g.data(), Ny * 9, MPI_DOUBLE, left_neighbor, 2, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(recv_left_g.data(), Ny * 9, MPI_DOUBLE, left_neighbor, 3, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (right_neighbor != MPI_PROC_NULL)
        {
            MPI_Isend(send_right.data(), Ny * 9, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(recv_right.data(), Ny * 9, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Isend(send_right_g.data(), Ny * 9, MPI_DOUBLE, right_neighbor, 3, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Irecv(recv_right_g.data(), Ny * 9, MPI_DOUBLE, right_neighbor, 2, MPI_COMM_WORLD, &requests[req_count++]);
        }

        // Esperar a que termine la comunicación
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Actualizar capas fantasma
        if (left_neighbor != MPI_PROC_NULL)
        {
            for (int j = 0; j < Ny; ++j)
            {
                f[0][j] = recv_left[j];
                g[0][j] = recv_left_g[j];
            }
        }
        if (right_neighbor != MPI_PROC_NULL)
        {
            for (int j = 0; j < Ny; ++j)
            {
                f[local_Nx + ghost_layers][j] = recv_right[j];
                g[local_Nx + ghost_layers][j] = recv_right_g[j];
            }
        }

        // Colisión (solo en el dominio local, no en capas fantasma)
        for (int i = ghost_layers; i < local_Nx + ghost_layers; ++i)
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

                int global_i = start_i + i - ghost_layers;
                double coupling = hv_lb * (Tgas - Tsol);
                double source = source_term(global_i, j);
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
        for (int i = ghost_layers; i < local_Nx + ghost_layers; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < 9; ++k)
                {
                    int ni = i - ex[k];
                    int nj = j - ey[k];
                    if (ni >= 0 && ni < total_local_Nx && nj >= 0 && nj < Ny)
                    {
                        f[i][j][k] = f_temp[ni][nj][k];
                        g[i][j][k] = g_temp[ni][nj][k];
                    }
                }

        // Condición de entrada (solo proceso 0)
        if (rank == 0)
        {
            for (int j = 0; j < Ny; ++j)
            {
                for (int k = 0; k < 9; ++k)
                {
                    if (ex[k] > 0)
                        f[ghost_layers][j][k] = feq(T_in, k, u_lb_x, u_lb_y);
                    g[ghost_layers][j][k] = feq(T_in, k, 0.0, 0.0);
                }
            }
        }
    }

    // Recolección de resultados en proceso 0
    if (rank == 0)
    {
        // Arrays globales para recolectar resultados
        std::vector<std::vector<double>> global_Tg(Nx, std::vector<double>(Ny));
        std::vector<std::vector<double>> global_Ts(Nx, std::vector<double>(Ny));

        // Copiar datos locales del proceso 0
        for (int i = 0; i < local_Nx; ++i)
            for (int j = 0; j < Ny; ++j)
            {
                global_Tg[i][j] = Tg[i + ghost_layers][j];
                global_Ts[i][j] = Ts[i + ghost_layers][j];
            }

        // Recibir datos de otros procesos
        for (int proc = 1; proc < size; ++proc)
        {
            int proc_local_Nx = Nx / size;
            if (proc < remainder)
                proc_local_Nx++;
            int proc_start_i = proc * (Nx / size) + std::min(proc, remainder);

            std::vector<double> recv_Tg(proc_local_Nx * Ny);
            std::vector<double> recv_Ts(proc_local_Nx * Ny);

            MPI_Recv(recv_Tg.data(), proc_local_Nx * Ny, MPI_DOUBLE, proc, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_Ts.data(), proc_local_Nx * Ny, MPI_DOUBLE, proc, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < proc_local_Nx; ++i)
                for (int j = 0; j < Ny; ++j)
                {
                    global_Tg[proc_start_i + i][j] = recv_Tg[i * Ny + j];
                    global_Ts[proc_start_i + i][j] = recv_Ts[i * Ny + j];
                }
        }

        // Escribir archivo
        std::ofstream file("lbm2d_output.csv");
        file << "x,y,Tg,Ts\n";
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                file << i * dx << "," << j * dy << "," << global_Tg[i][j] << "," << global_Ts[i][j] << "\n";

        std::cout << "Simulación 2D LBM completada. Resultados en 'lbm2d_output.csv'.\n";
    }
    else
    {
        // Enviar datos locales al proceso 0
        std::vector<double> send_Tg(local_Nx * Ny);
        std::vector<double> send_Ts(local_Nx * Ny);

        for (int i = 0; i < local_Nx; ++i)
            for (int j = 0; j < Ny; ++j)
            {
                send_Tg[i * Ny + j] = Tg[i + ghost_layers][j];
                send_Ts[i * Ny + j] = Ts[i + ghost_layers][j];
            }

        MPI_Send(send_Tg.data(), local_Nx * Ny, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        MPI_Send(send_Ts.data(), local_Nx * Ny, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}