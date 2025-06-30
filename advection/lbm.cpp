#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// Parámetros físicos
const double L = 1.0;         // Longitud del dominio
const double alpha_g = 0.01;  // Difusividad térmica gas
const double alpha_s = 0.005; // Difusividad térmica sólido
const double u_phys = 0.1;    // Velocidad física del gas
const double hv_phys = 5.0;   // Coeficiente transferencia calor
const double T_in = 15.0;     // Temperatura entrada
const double S_value = 315.0; // Fuente de calor

// Parámetros de discretización
const int N = 100;             // Número de nodos
const double dx = L / (N - 1); // Tamaño de celda
const int steps = 40000;       // Pasos de tiempo

// Parámetros LBM
const double cs2 = 1.0 / 3.0; // Velocidad del sonido al cuadrado
const double dt = 0.00005;    // Paso temporal pequeño para estabilidad

// Conversión a unidades LBM
const double u_lb = u_phys * dt / dx; // Velocidad en unidades LBM
const double hv_lb = hv_phys * dt;    // Coeficiente acoplamiento en unidades LBM

// Parámetros de relajación
const double tau_g = 0.5 + alpha_g * dt / (cs2 * dx * dx);
const double tau_s = 0.5 + alpha_s * dt / (cs2 * dx * dx);

// Función de equilibrio térmica D1Q3
double feq(double T, int k, double u_local)
{
    double w[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
    double e[3] = {1.0, 0.0, -1.0};

    double eu = e[k] * u_local;
    return w[k] * T * (1.0 + eu / cs2 + 0.5 * eu * eu / (cs2 * cs2) - 0.5 * u_local * u_local / cs2);
}

// Término fuente
double source_term(int i)
{
    double x = i * dx;
    return (x >= 0.4 * L && x <= 0.6 * L) ? S_value * dt : 0.0;
}

int main()
{
    // Verificar estabilidad CFL
    double CFL = u_lb * dt / dx;
    double diffusion_CFL = max(alpha_g, alpha_s) * dt / (dx * dx);

    cout << "=== Parámetros de simulación ===" << endl;
    cout << "N = " << N << ", dx = " << dx << ", dt = " << dt << endl;
    cout << "u_lb = " << u_lb << ", CFL = " << CFL << endl;
    cout << "tau_g = " << tau_g << ", tau_s = " << tau_s << endl;
    cout << "Difusión CFL = " << diffusion_CFL << endl;

    if (CFL > 1.0 || diffusion_CFL > 0.5 || tau_g < 0.5 || tau_s < 0.5)
    {
        cout << "ADVERTENCIA: Parámetros potencialmente inestables!" << endl;
    }
    cout << "===============================" << endl;

    // Distribuciones de probabilidad
    vector<vector<double>> f(3, vector<double>(N, 0.0)); // Gas
    vector<vector<double>> g(3, vector<double>(N, 0.0)); // Sólido

    // Temperaturas
    vector<double> Tg(N, 0.0);
    vector<double> Ts(N, 0.0);

    // Inicialización
    for (int i = 0; i < N; i++)
    {
        // Condición inicial: perfil lineal
        double x = i * dx;
        Tg[i] = T_in * (1.0 - x / L);
        Ts[i] = T_in * (1.0 - x / L);

        for (int k = 0; k < 3; k++)
        {
            f[k][i] = feq(Tg[i], k, u_lb);
            g[k][i] = feq(Ts[i], k, 0.0);
        }
    }

    // Archivos de salida
    ofstream progress_file("lbm_progress.dat");
    progress_file << "# time max_Tg max_Ts avg_Tg avg_Ts\n";

    // Bucle temporal principal
    for (int t = 0; t < steps; t++)
    {
        // 1. Calcular temperaturas macroscópicas
        for (int i = 0; i < N; i++)
        {
            Tg[i] = f[0][i] + f[1][i] + f[2][i];
            Ts[i] = g[0][i] + g[1][i] + g[2][i];
        }

        // 2. Paso de colisión con términos fuente
        for (int i = 0; i < N; i++)
        {
            // Términos de acoplamiento
            double coupling = hv_lb * (Tg[i] - Ts[i]);
            double source = source_term(i);

            // Términos fuente para cada fase
            double Sg = -coupling + source; // Gas pierde calor al sólido + fuente
            double Ss = coupling;           // Sólido gana calor del gas

            for (int k = 0; k < 3; k++)
            {
                // Distribuciones de equilibrio
                double feq_g = feq(Tg[i], k, u_lb);
                double feq_s = feq(Ts[i], k, 0.0);

                // Colisión BGK con términos fuente
                // Método de He & Luo para términos fuente en LBM térmico
                double w[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
                double e[3] = {1.0, 0.0, -1.0};

                // Términos fuente en las distribuciones
                double Fk_g = w[k] * Sg * (1.0 + e[k] * u_lb / cs2);
                double Fk_s = w[k] * Ss;

                // Colisión con términos fuente
                f[k][i] = f[k][i] - (f[k][i] - feq_g) / tau_g + Fk_g;
                g[k][i] = g[k][i] - (g[k][i] - feq_s) / tau_s + Fk_s;
            }
        }

        // 3. Paso de streaming
        vector<vector<double>> f_temp = f;
        vector<vector<double>> g_temp = g;

        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < 3; k++)
            {
                int e[3] = {1, 0, -1};
                int i_prev = i - e[k];

                if (i_prev >= 0 && i_prev < N)
                {
                    f[k][i] = f_temp[k][i_prev];
                    g[k][i] = g_temp[k][i_prev];
                }
            }
        }

        // 4. Condiciones de contorno
        // Entrada (i=0): Temperatura fija
        for (int k = 0; k < 3; k++)
        {
            int e[3] = {1, 0, -1};
            if (e[k] > 0) // Direcciones entrantes
            {
                f[k][0] = feq(T_in, k, u_lb);
                g[k][0] = feq(T_in, k, 0.0);
            }
        }

        // Salida (i=N-1): Condición de salida libre
        for (int k = 0; k < 3; k++)
        {
            int e[3] = {1, 0, -1};
            if (e[k] < 0) // Direcciones salientes
            {
                f[k][N - 1] = f[k][N - 2];
                g[k][N - 1] = g[k][N - 2];
            }
        }

        // 5. Monitoreo de progreso
        if (t % 1000 == 0)
        {
            // Recalcular temperaturas para monitoreo
            for (int i = 0; i < N; i++)
            {
                Tg[i] = f[0][i] + f[1][i] + f[2][i];
                Ts[i] = g[0][i] + g[1][i] + g[2][i];
            }

            double max_Tg = *max_element(Tg.begin(), Tg.end());
            double max_Ts = *max_element(Ts.begin(), Ts.end());
            double avg_Tg = 0.0, avg_Ts = 0.0;

            for (int i = 0; i < N; i++)
            {
                avg_Tg += Tg[i];
                avg_Ts += Ts[i];
            }
            avg_Tg /= N;
            avg_Ts /= N;

            cout << "t=" << t << " | Max: Tg=" << fixed << setprecision(4) << max_Tg
                 << " Ts=" << max_Ts << " | Avg: Tg=" << avg_Tg << " Ts=" << avg_Ts << endl;

            progress_file << t * dt << " " << max_Tg << " " << max_Ts << " "
                          << avg_Tg << " " << avg_Ts << endl;
        }
    }

    // Calcular temperaturas finales
    for (int i = 0; i < N; i++)
    {
        Tg[i] = f[0][i] + f[1][i] + f[2][i];
        Ts[i] = g[0][i] + g[1][i] + g[2][i];
    }

    // Exportar resultados finales
    ofstream results_file("lbm_output.csv");
    results_file << "x,Tg,Ts,difference,source\n";
    for (int i = 0; i < N; i++)
    {
        double x = i * dx;
        double src = (x >= 0.4 * L && x <= 0.6 * L) ? S_value : 0.0;
        results_file << fixed << setprecision(6) << x << ","
                     << Tg[i] << "," << Ts[i] << ","
                     << (Tg[i] - Ts[i]) << "," << src << endl;
    }

    progress_file.close();
    results_file.close();

    // Estadísticas finales
    double max_Tg = *max_element(Tg.begin(), Tg.end());
    double max_Ts = *max_element(Ts.begin(), Ts.end());
    double avg_diff = 0.0;

    for (int i = 0; i < N; i++)
    {
        avg_diff += abs(Tg[i] - Ts[i]);
    }
    avg_diff /= N;

    cout << "\n=== RESULTADOS FINALES ===" << endl;
    cout << "Temperatura máxima gas: " << max_Tg << endl;
    cout << "Temperatura máxima sólido: " << max_Ts << endl;
    cout << "Diferencia promedio |Tg-Ts|: " << avg_diff << endl;
    cout << "Archivos generados:" << endl;
    cout << "  - lbm_output.csv (perfil final)" << endl;
    cout << "  - lbm_progress.dat (evolución temporal)" << endl;
    cout << "=========================" << endl;

    return 0;
}