#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

const int N = 100;            // Número de nodos espaciales
const int steps = 2000;       // Pasos de tiempo
const double dx = 0.01;       // Tamaño de celda espacial
const double dt = 0.001;      // Paso de tiempo
const double alpha_g = 0.01;  // Difusividad térmica fase gas
const double alpha_s = 0.005; // Difusividad térmica fase sólida
const double u = 0.1;         // Velocidad convectiva
const double hv = 5.0;        // Coeficiente de transferencia de calor entre fases
const double T_in = 15.0;     // Temperatura de entrada
const double S_value = 315.0; // Valor del término fuente (localizado)

// Fuente puntual entre nodos 40 y 60
double S(int i)
{
    return (i >= 40 && i <= 60) ? S_value : 0.0;
}

int main()
{
    std::vector<double> Tg(N, 0.0), Ts(N, 0.0);
    std::vector<double> Tg_new(N, 0.0), Ts_new(N, 0.0);

    // Condición inicial: temperatura uniforme 0
    for (int i = 0; i < N; ++i)
    {
        Tg[i] = Ts[i] = 0.0;
    }

    for (int t = 0; t < steps; ++t)
    {
        for (int i = 1; i < N - 1; ++i)
        {
            // GASEOSA: convección + difusión + transferencia + fuente
            double convection = -u * (Tg[i] - Tg[i - 1]) / dx;
            double diffusion = alpha_g * (Tg[i + 1] - 2 * Tg[i] + Tg[i - 1]) / (dx * dx);
            Tg_new[i] = Tg[i] + dt * (convection + diffusion - hv * (Tg[i] - Ts[i]) + S(i));

            // SÓLIDA: sólo difusión + transferencia
            double diffusion_s = alpha_s * (Ts[i + 1] - 2 * Ts[i] + Ts[i - 1]) / (dx * dx);
            Ts_new[i] = Ts[i] + dt * (diffusion_s + hv * (Tg[i] - Ts[i]));
        }

        // Condiciones de frontera
        Tg_new[0] = Ts_new[0] = T_in;
        Tg_new[N - 1] = Tg_new[N - 2]; // ∂T/∂x ≈ 0
        Ts_new[N - 1] = Ts_new[N - 2];

        std::copy(Tg_new.begin(), Tg_new.end(), Tg.begin());
        std::copy(Ts_new.begin(), Ts_new.end(), Ts.begin());
    }

    // Exportar resultados
    std::ofstream file("fdm_output.csv");
    file << "x,Tg,Ts\n";
    for (int i = 0; i < N; ++i)
        file << i * dx << "," << Tg[i] << "," << Ts[i] << "\n";

    std::cout << "Simulación FDM finalizada. Resultados guardados en 'fdm_output.csv'.\n";
    return 0;
}
