#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <omp.h>

// Constantes físicas
const double epsilon0 = 8.854187817e-12;  // Permitividad del vacío (F/m)
const double mu0 = 4 * M_PI * 1e-7;      // Permeabilidad del vacío (H/m)
const double c = 1.0 / sqrt(epsilon0 * mu0); // Velocidad de la luz (m/s)

using Campo = std::vector<double>;

void guardarCampos(const Campo& Ex, const Campo& Hy, int paso, const std::string& subcarpeta) {
    // Carpeta principal modificada a "resultados_parallel"
    const std::string carpeta_principal = "resultados_parallel";
    std::filesystem::create_directories(carpeta_principal + "/" + subcarpeta);

    std::ostringstream nombre;
    nombre << carpeta_principal << "/" << subcarpeta << "/campos_t"
           << std::setw(4) << std::setfill('0') << paso << ".dat";
    std::ofstream archivo(nombre.str());

    for (size_t k = 0; k < Ex.size(); ++k) {
        archivo << k << " " << Ex[k] << " " << Hy[k] << "\n";
    }

    archivo.close();
}

class FDTD1D {
private:
    int Nz;                  // Número de puntos en la grilla espacial
    double dz;               // Paso espacial (m)
    double dt;               // Paso temporal (s)
    double beta;             // Parámetro de estabilidad (número de Courant)

    Campo Ex;                // Campo eléctrico (normalizado)
    Campo Hy;                // Campo magnético

    // Índices para condiciones periódicas
    int prev(int i) const { return (i == 0) ? Nz-1 : i-1; }
    int next(int i) const { return (i == Nz-1) ? 0 : i+1; }

public:
    FDTD1D(int numCells, double cellSize, double courantFactor)
        : Nz(numCells), dz(cellSize) {

        // Calcular paso temporal según condición de Courant
        dt = courantFactor * dz / c;
        beta = c * dt / dz;

        // Inicializar campos
        Ex.resize(Nz, 0.0);
        Hy.resize(Nz, 0.0);

        std::cout << "FDTD 1D inicializado con:\n";
        std::cout << "  Nz = " << Nz << ", dz = " << dz << " m\n";
        std::cout << "  dt = " << dt << " s, beta (Factor de Courant) = " << beta << "\n";
    }

    // Actualizar campo eléctrico (paso medio temporal)
    void updateE() {
        double start_time = omp_get_wtime();
        int max_threads = omp_get_max_threads();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < Nz; k++) {
                Ex[k] += beta * (Hy[prev(k)] - Hy[k]);
            }
        }
    }

    // Actualizar campo magnético (paso completo temporal)
    void updateH() {
        double start_time = omp_get_wtime();
        int max_threads = omp_get_max_threads();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < Nz; k++) {
                Hy[k] += beta * (Ex[k] - Ex[next(k)]);
            }
        }
    }

    // Ejecutar un paso completo de simulación
    void step() {
        updateE();
        updateH();
    }

    // Establecer condición inicial de pulso gaussiano
    void setGaussianPulse(double center, double width) {
        double start_time = omp_get_wtime();
        int max_threads = omp_get_max_threads();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < Nz; k++) {
                double z = k * dz;
                Ex[k] = exp(-0.5 * pow((z - center)/width, 2));
                Hy[k] = 0.0;
            }
        }
        
        double end_time = omp_get_wtime();
        std::cout << "  setGaussianPulse: " << (end_time - start_time) * 1000 << " ms, ";
        std::cout << "Threads usados: " << omp_get_num_threads() << "/" << max_threads << "\n";
    }

    // Establecer condición inicial de pulso senosoidal
    void setSinusoidalPulse() {
        double start_time = omp_get_wtime();
        int max_threads = omp_get_max_threads();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < Nz; k++) {
                Ex[k] = 0.1 * sin(2 * M_PI * k / (Nz / 10.0));
                Hy[k] = 0.1 * sin(2 * M_PI * k / (Nz / 10.0));
            }
        }
        
        double end_time = omp_get_wtime();
        std::cout << "  setSinusoidalPulse: " << (end_time - start_time) * 1000 << " ms, ";
        std::cout << "Threads usados: " << omp_get_num_threads() << "/" << max_threads << "\n";
    }

    // Acceso a los campos
    const Campo& getEx() const { return Ex; }
    const Campo& getHy() const { return Hy; }
};

void ejecutarSimulacion(FDTD1D& fdtd, const std::string& subcarpeta, const std::string& tipoPulso, int numSteps) {
    auto startTime = std::chrono::high_resolution_clock::now();
    double total_omp_time = 0.0;
    int max_threads_used = 0;

    std::cout << "\nIniciando simulación con pulso " << tipoPulso << " (" << numSteps << " pasos)\n";
    
    for (int n = 0; n < numSteps; n++) {
        double step_start = omp_get_wtime();
        
        fdtd.step();
        
        double step_end = omp_get_wtime();
        total_omp_time += (step_end - step_start);
        
        int current_threads = omp_get_max_threads();
        if (current_threads > max_threads_used) {
            max_threads_used = current_threads;
        }

        if (n % 10 == 0) { // Guardar cada 10 pasos
            guardarCampos(fdtd.getEx(), fdtd.getHy(), n, subcarpeta);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nResumen de la simulación:\n";
    std::cout << "  Tipo de pulso: " << tipoPulso << "\n";
    std::cout << "  Pasos totales: " << numSteps << "\n";
    std::cout << "  Tiempo total de ejecución: " << duration.count() << " ms\n";
    std::cout << "  Tiempo total en regiones paralelas: " << total_omp_time * 1000 << " ms\n";
    std::cout << "  Máximo número de hilos utilizados: " << max_threads_used << "\n";
    std::cout << "  Tiempo promedio por paso: " << duration.count() / numSteps << " ms/paso\n";
}

int main() {
    // Configuración inicial de OpenMP
    omp_set_dynamic(0);     // Desactivar ajuste dinámico de threads
    omp_set_num_threads(omp_get_max_threads()); // Usar todos los threads disponibles
    
    std::cout << "Configuración de paralelización:\n";
    std::cout << "  Número máximo de hilos disponibles: " << omp_get_max_threads() << "\n";
    
    // Crear carpeta principal de resultados (ahora "resultados_parallel")
    std::filesystem::create_directory("resultados_parallel");

    // Parámetros de la simulación
    double lambda_deseada = 1.0e-6;
    int num_celdas_grilla = 1000;
    double dz_calculado = lambda_deseada / 20.0;
    double factor_courant_estabilidad = 0.5;
    int pasos_temporales_simulacion = 2000;

    std::cout << "\n--- Configuración de la simulación ---\n";
    std::cout << "Ingrese la longitud de onda deseada (en metros, ej. 1e-6): ";
    std::cin >> lambda_deseada;
    std::cout << "Ingrese el número de puntos en la grilla espacial (Nz, ej. 1000): ";
    std::cin >> num_celdas_grilla;
    
    dz_calculado = lambda_deseada / 20.0;
    std::cout << "Se calculará el paso espacial (dz) como lambda / 20 = " << dz_calculado << " m\n";
    
    std::cout << "Ingrese el factor de Courant (dt * c / dz). Para estabilidad, debe ser <= 1.0. Recomendado 0.5: ";
    std::cin >> factor_courant_estabilidad;
    
    if (factor_courant_estabilidad > 1.0) {
        std::cerr << "Advertencia: El factor de Courant ingresado (" << factor_courant_estabilidad
                  << ") es mayor que 1.0. Esto causará inestabilidad numérica. Se ajustará a 0.99 para intentar la estabilidad.\n";
        factor_courant_estabilidad = 0.99;
    } else if (factor_courant_estabilidad <= 0.0) {
        std::cerr << "Advertencia: El factor de Courant no puede ser cero o negativo. Se ajustará a 0.5.\n";
        factor_courant_estabilidad = 0.5;
    }

    std::cout << "Ingrese el número total de pasos temporales para la simulación (ej. 2000): ";
    std::cin >> pasos_temporales_simulacion;

    std::cout << "\nParámetros finales de la simulación:\n";
    std::cout << "  Longitud de onda base (lambda): " << lambda_deseada << " m\n";
    std::cout << "  Número de celdas (Nz): " << num_celdas_grilla << "\n";
    std::cout << "  Paso espacial (dz): " << dz_calculado << " m\n";
    std::cout << "  Factor de Courant (beta): " << factor_courant_estabilidad << "\n";
    std::cout << "  Número de pasos temporales: " << pasos_temporales_simulacion << "\n\n";

    // Simulación con pulso gaussiano
    {
        std::cout << "\n=== Iniciando simulación con pulso gaussiano ===\n";
        FDTD1D fdtd_gauss(num_celdas_grilla, dz_calculado, factor_courant_estabilidad);
        fdtd_gauss.setGaussianPulse(num_celdas_grilla / 4.0 * dz_calculado, lambda_deseada / 4.0);
        ejecutarSimulacion(fdtd_gauss, "gaussiano", "gaussiano", pasos_temporales_simulacion);
    }

    // Simulación con pulso senosoidal
    {
        std::cout << "\n=== Iniciando simulación con pulso senosoidal ===\n";
        FDTD1D fdtd_sin(num_celdas_grilla, dz_calculado, factor_courant_estabilidad);
        fdtd_sin.setSinusoidalPulse();
        ejecutarSimulacion(fdtd_sin, "senosoidal", "senosoidal", pasos_temporales_simulacion);
    }

    return 0;
}