#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>

// Constantes físicas
const double epsilon0 = 8.854187817e-12;     // Permitividad del vacío (F/m)
const double mu0 = 4 * M_PI * 1e-7;          // Permeabilidad del vacío (H/m)
const double c = 1.0 / sqrt(epsilon0 * mu0); // Velocidad de la luz (m/s)

using Campo = std::vector<double>;

void guardarCampos(const Campo &Ex, const Campo &Hy, int paso, const std::string &subcarpeta)
{
    // Crear carpeta principal y subcarpeta
    const std::string carpeta_principal = "resultados";
    std::filesystem::create_directories(carpeta_principal + "/" + subcarpeta);

    std::ostringstream nombre;
    nombre << carpeta_principal << "/" << subcarpeta << "/campos_t"
           << std::setw(4) << std::setfill('0') << paso << ".dat";
    std::ofstream archivo(nombre.str());

    for (size_t k = 0; k < Ex.size(); ++k)
    {
        archivo << k << " " << Ex[k] << " " << Hy[k] << "\n";
    }

    archivo.close();
}

class FDTD1D
{
private:
    int Nz;      // Número de puntos en la grilla espacial
    double dz;   // Paso espacial (m)
    double dt;   // Paso temporal (s)
    double beta; // Parámetro de estabilidad

    Campo Ex; // Campo eléctrico (normalizado)
    Campo Hy; // Campo magnético

    // Índices para condiciones periódicas
    int prev(int i) const { return (i == 0) ? Nz - 1 : i - 1; }
    int next(int i) const { return (i == Nz - 1) ? 0 : i + 1; }

public:
    FDTD1D(int numCells, double cellSize, double courantFactor = 0.5)
        : Nz(numCells), dz(cellSize)
    {

        // Calcular paso temporal según condición de Courant
        dt = courantFactor * dz / c;
        beta = c * dt / dz;

        // Inicializar campos
        Ex.resize(Nz, 0.0);
        Hy.resize(Nz, 0.0);

        std::cout << "FDTD 1D inicializado con:\n";
        std::cout << "  Nz = " << Nz << ", dz = " << dz << " m\n";
        std::cout << "  dt = " << dt << " s, beta = " << beta << "\n";
    }

    // Actualizar campo eléctrico (paso medio temporal)
    void updateE()
    {
        for (int k = 0; k < Nz; k++)
        {
            Ex[k] = Ex[k] + beta * (Hy[prev(k)] - Hy[k]);
        }
    }

    // Actualizar campo magnético (paso completo temporal)
    void updateH()
    {
        for (int k = 0; k < Nz; k++)
        {
            Hy[k] = Hy[k] + beta * (Ex[k] - Ex[next(k)]);
        }
    }

    // Ejecutar un paso completo de simulación
    void step()
    {
        updateE();
        updateH();
    }

    // Establecer condición inicial de pulso gaussiano
    void setGaussianPulse(double center, double width)
    {
        for (int k = 0; k < Nz; k++)
        {
            double z = k * dz;
            Ex[k] = exp(-0.5 * pow((z - center) / width, 2));
            Hy[k] = 0.0; // Inicializar Hy a cero
        }
    }

    // Establecer condición inicial de pulso senosoidal
    void setSinusoidalPulse()
    {
        for (int k = 0; k < Nz; k++)
        {
            Ex[k] = 0.1 * sin(2 * M_PI * k / 100.0);
            Hy[k] = 0.1 * sin(2 * M_PI * k / 100.0);
        }
    }

    // Acceso a los campos
    const Campo &getEx() const { return Ex; }
    const Campo &getHy() const { return Hy; }
};

void ejecutarSimulacion(FDTD1D &fdtd, const std::string &subcarpeta, const std::string &tipoPulso)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    const int numSteps = 50000;
    for (int n = 0; n < numSteps; n++)
    {
        fdtd.step();

        if (n % 10 == 0)
        {
            guardarCampos(fdtd.getEx(), fdtd.getHy(), n, subcarpeta);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Simulación con pulso " << tipoPulso << " completada en: " << duration.count() << "ms\n";
}

int main()
{
    // Crear carpeta principal de resultados
    std::filesystem::create_directory("resultados");

    // Parámetros de la simulación
    const double lambda = 1.0e-6;     // Longitud de onda (m)
    const double dz = lambda / 20.0;  // Paso espacial (< lambda/10)
    const int Nz = 1000;              // Número de celdas
    const double courantFactor = 0.5; // Factor de Courant

    // Simulación con pulso gaussiano
    {
        FDTD1D fdtd_gauss(Nz, dz, courantFactor);
        fdtd_gauss.setGaussianPulse(Nz / 4 * dz, lambda / 4);
        ejecutarSimulacion(fdtd_gauss, "gaussiano", "gaussiano");
    }

    // Simulación con pulso senosoidal
    {
        FDTD1D fdtd_sin(Nz, dz, courantFactor);
        fdtd_sin.setSinusoidalPulse();
        ejecutarSimulacion(fdtd_sin, "senosoidal", "senosoidal");
    }

    return 0;
}