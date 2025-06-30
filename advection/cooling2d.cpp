#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

const int Nx = 100;
const int Ny = 100;
const int steps = 2000;
const double dx = 0.01;
const double dy = 0.01;
const double dt = 0.0005;
const double alpha = 0.01;
const double u = 0.1;
const double T_in = 20.0;

int main()
{
    std::vector<std::vector<double>> T(Nx, std::vector<double>(Ny, T_in));
    std::vector<std::vector<double>> T_new = T;

    for (int t = 0; t < steps; ++t)
    {
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                double convection = -u * (T[i][j] - T[i - 1][j]) / dx;
                double diffusion = alpha * ((T[i + 1][j] - 2 * T[i][j] + T[i - 1][j]) / (dx * dx) +
                                            (T[i][j + 1] - 2 * T[i][j] + T[i][j - 1]) / (dy * dy));
                T_new[i][j] = T[i][j] + dt * (convection + diffusion);
            }
        }

        // Condiciones de frontera
        for (int j = 0; j < Ny; ++j)
        {
            T_new[0][j] = 20.0;                  // Entrada
            T_new[Nx - 1][j] = T_new[Nx - 2][j]; // Salida
        }
        for (int i = 0; i < Nx; ++i)
        {
            T_new[i][0] = 80.0;                  // Pared inferior caliente
            T_new[i][Ny - 1] = T_new[i][Ny - 2]; // Pared superior aislada
        }

        T = T_new;
    }

    std::ofstream file("cooling2d_output.csv");
    file << "x,y,T\n";
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            file << i * dx << "," << j * dy << "," << T[i][j] << "\n";

    std::cout << "Simulación 2D completada. Datos en 'cooling2d_output.csv'.\n";
    return 0;
}
