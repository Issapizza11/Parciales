#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

const int Nx = 500;
const int Ny = 500;
const int steps = 10000;
const double Lx = 1.0, Ly = 1.0;
const double dx = Lx / (Nx - 1);
const double dy = Ly / (Ny - 1);
const double dt = 0.00001;
const double alpha_g = 0.01;
const double alpha_s = 0.005;
const double u_x = 0.1;
const double u_y = 0.05;
const double hv = 5.0;
const double T_in = 0.0;
const double S_value = 315.0;

// double S(int i, int j)
// {
//     double x = i * dx;
//     double y = j * dy;
//     return (x >= 0.1 && x <= 0.3 && y >= 0.4 && y <= 0.6) ? S_value : 0.0;
// }

double S(int i, int j)
{
    // Escalamiento exacto en base al dominio físico
    double x = static_cast<double>(i) / (Nx - 1) * Lx;
    double y = static_cast<double>(j) / (Ny - 1) * Ly;

    // Evita errores de redondeo usando márgenes suaves
    double eps = 1e-10;

    double S1 = ((x >= 0.2 - eps && x <= 0.3 + eps) &&
                 (y >= 0.4 - eps && y <= 0.5 + eps))
                    ? S_value
                    : 0.0;

    double S2 = ((x >= 0.7 - eps && x <= 0.8 + eps) &&
                 (y >= 0.6 - eps && y <= 0.7 + eps))
                    ? S_value
                    : 0.0;

    return S1 + S2;
}

int main()
{
    int fuente_activa = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            if (S(i, j) > 0.0)
            {
                fuente_activa++;
            }
        }
    }
    std::cout << "Número de celdas con fuente activa: " << fuente_activa << std::endl;

    std::vector<std::vector<double>> Tg(Nx, std::vector<double>(Ny, 0.0));
    std::vector<std::vector<double>> Ts(Nx, std::vector<double>(Ny, 0.0));
    std::vector<std::vector<double>> Tg_new = Tg;
    std::vector<std::vector<double>> Ts_new = Ts;

    for (int t = 0; t < steps; ++t)
    {
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                double convection_x = -u_x * (Tg[i][j] - Tg[i - 1][j]) / dx;
                double convection_y = -u_y * (Tg[i][j] - Tg[i][j - 1]) / dy;
                double diffusion_g = alpha_g * ((Tg[i + 1][j] - 2 * Tg[i][j] + Tg[i - 1][j]) / (dx * dx) +
                                                (Tg[i][j + 1] - 2 * Tg[i][j] + Tg[i][j - 1]) / (dy * dy));
                double diffusion_s = alpha_s * ((Ts[i + 1][j] - 2 * Ts[i][j] + Ts[i - 1][j]) / (dx * dx) +
                                                (Ts[i][j + 1] - 2 * Ts[i][j] + Ts[i][j - 1]) / (dy * dy));
                Tg_new[i][j] = Tg[i][j] + dt * (convection_x + convection_y + diffusion_g - hv * (Tg[i][j] - Ts[i][j]) + S(i, j));
                Ts_new[i][j] = Ts[i][j] + dt * (diffusion_s + hv * (Tg[i][j] - Ts[i][j]));

                if (t % 500 == 0 && i == Nx / 2 && j == Ny / 2)
                {
                    std::cout << "Paso " << t << " - Tg centro: " << Tg_new[i][j]
                              << ", Ts centro: " << Ts_new[i][j] << std::endl;
                }
            }
        }

        // Bordes: Dirichlet en entrada, Neumann en salida
        for (int j = 0; j < Ny; ++j)
        {
            Tg_new[0][j] = Ts_new[0][j] = T_in;
            Tg_new[Nx - 1][j] = Tg_new[Nx - 2][j];
            Ts_new[Nx - 1][j] = Ts_new[Nx - 2][j];
        }
        for (int i = 0; i < Nx; ++i)
        {
            Tg_new[i][0] = Tg_new[i][1];
            Ts_new[i][0] = Ts_new[i][1];
            Tg_new[i][Ny - 1] = Tg_new[i][Ny - 2];
            Ts_new[i][Ny - 1] = Ts_new[i][Ny - 2];
        }

        Tg = Tg_new;
        Ts = Ts_new;
    }

    std::ofstream file("fdm2d_output.csv");
    file << "x,y,Tg,Ts\n";
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            file << i * dx << "," << j * dy << "," << Tg[i][j] << "," << Ts[i][j] << "\n";
        }
    }

    std::cout << "Simulación 2D FDM completada. Datos en 'fdm2d_output.csv'.\n";
    return 0;
}
