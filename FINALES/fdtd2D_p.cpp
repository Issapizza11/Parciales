#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <cstring>
#include <sys/stat.h> // Para mkdir

// Constantes físicas
const double epsilon0 = 8.854187817e-12;
const double mu0 = 4 * M_PI * 1e-7;
const double c = 1.0 / std::sqrt(epsilon0 * mu0);

using CampoPlano = std::vector<double>;

// Crear directorio si no existe (estilo POSIX)
void crearDirectorioSiNoExiste(const std::string &nombre)
{
    struct stat st;
    if (stat(nombre.c_str(), &st) != 0)
    {
        mkdir(nombre.c_str(), 0755);
    }
}

void guardarCampos2D(const CampoPlano &Ex, const CampoPlano &Ey, const CampoPlano &Hz,
                     int Nx, int Ny, int paso, const std::string &subcarpeta)
{
    const std::string carpeta_principal = "data2D";
    crearDirectorioSiNoExiste(carpeta_principal);
    crearDirectorioSiNoExiste(carpeta_principal + "/" + subcarpeta);

    std::ostringstream nombre;
    nombre << carpeta_principal << "/" << subcarpeta << "/campos_t"
           << std::setw(4) << std::setfill('0') << paso << ".dat";
    std::ofstream archivo(nombre.str());

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            int id = i * Ny + j;
            archivo << i << " " << j << " "
                    << Ex[id] << " "
                    << Ey[id] << " "
                    << Hz[id] << "\n";
        }
        archivo << "\n";
    }
    archivo.close();
}

class FDTD2D
{
private:
    int Nx, Ny;
    double dx, dy;
    double dt;
    double beta_x, beta_y;

    CampoPlano Ex, Ey, Hz;

    inline int idx(int i, int j) const { return i * Ny + j; }

public:
    FDTD2D(int numCellsX, int numCellsY, double cellSizeX, double cellSizeY,
           double courantFactor = 0.5)
        : Nx(numCellsX), Ny(numCellsY), dx(cellSizeX), dy(cellSizeY)
    {
        double courantMax = 1.0 / (c * std::sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)));
        dt = courantFactor * courantMax;

        if (courantFactor >= 1.0)
        {
            std::cerr << "¡Advertencia! Courant (" << courantFactor
                      << ") puede ser inestable. Recomendado < 0.7\n";
        }

        beta_x = c * dt / dx;
        beta_y = c * dt / dy;

        Ex.resize(Nx * Ny, 0.0);
        Ey.resize(Nx * Ny, 0.0);
        Hz.resize(Nx * Ny, 0.0);

        std::cout << "FDTD 2D inicializado con " << omp_get_max_threads() << " hilos\n";
    }

    void updateE()
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                int id = idx(i, j);
                Ex[id] += beta_y * (Hz[id] - Hz[idx(i, j - 1)]);
                Ey[id] -= beta_x * (Hz[id] - Hz[idx(i - 1, j)]);
            }
        }
    }

    void updateH()
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < Nx - 1; ++i)
        {
            for (int j = 0; j < Ny - 1; ++j)
            {
                int id = idx(i, j);
                Hz[id] += beta_x * (Ey[idx(i + 1, j)] - Ey[id]) -
                          beta_y * (Ex[idx(i, j + 1)] - Ex[id]);
            }
        }
    }

    void step()
    {
        updateE();
        updateH();
    }

    void setGaussianPulse(double centerX, double centerY, double width)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                double x = i * dx;
                double y = j * dy;
                double r2 = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY);
                double val = std::exp(-0.5 * r2 / (width * width));
                int id = idx(i, j);
                Ex[id] = val;
                Ey[id] = val;
                Hz[id] = val;
            }
        }
    }

    void setSinusoidalPulse()
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                double val = 0.1 * std::sin(2 * M_PI * i / 50.0) * std::sin(2 * M_PI * j / 50.0);
                int id = idx(i, j);
                Ex[id] = val;
                Ey[id] = val;
                Hz[id] = val;
            }
        }
    }

    const CampoPlano &getEx() const { return Ex; }
    const CampoPlano &getEy() const { return Ey; }
    const CampoPlano &getHz() const { return Hz; }
    int getNx() const { return Nx; }
    int getNy() const { return Ny; }
};

void ejecutarSimulacion2D(FDTD2D &fdtd, const std::string &subcarpeta, const std::string &tipoPulso,
                          bool report_mode)
{
    const int steps = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < steps; ++n)
    {
        fdtd.step();

        if (!report_mode && (n % 10 == 0))
        {
            guardarCampos2D(fdtd.getEx(), fdtd.getEy(), fdtd.getHz(),
                            fdtd.getNx(), fdtd.getNy(), n, subcarpeta);
        }

        if (n % (steps / 100) == 0)
        {
            std::cout << "\rProgreso: " << std::fixed << std::setprecision(1)
                      << 100.0 * n / steps << "% " << std::flush;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "\rProgreso: 100.0%\n";
    std::cout << "Tiempo de ejecución: " << seconds << " s\n";

    const double updates = static_cast<double>(fdtd.getNx()) * fdtd.getNy() * steps;
    const double mlups = updates / seconds / 1e6;
    std::cout << "Throughput: " << mlups << " MLUP/s\n";
}

int main(int argc, char *argv[])
{
    int Nx = 250, Ny = 250;
    int num_threads = -1;
    bool report_mode = false;
    double lambda = 1.0e-6;
    double dx = lambda / 20.0, dy = lambda / 20.0;
    double courantFactor = 0.3;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--report") == 0)
            report_mode = true;
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--malla") == 0) && i + 1 < argc)
        {
            int val = std::stoi(argv[++i]);
            Nx = Ny = val;
        }
        else if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--hilos") == 0) && i + 1 < argc)
        {
            num_threads = std::stoi(argv[++i]);
        }
    }

    if (num_threads > 0)
        omp_set_num_threads(num_threads);

#pragma omp parallel
    {
#pragma omp single
        std::cout << "Se están usando " << omp_get_num_threads() << " hilos.\n";
    }

    std::cout << "Resolución de malla: " << Nx << " x " << Ny << "\n";

    dx = lambda / 20.0;
    dy = lambda / 20.0;

    if (!report_mode)
        crearDirectorioSiNoExiste("data2D");

#pragma omp parallel sections
    {
#pragma omp section
        {
            FDTD2D fdtd_gauss(Nx, Ny, dx, dy, courantFactor);
            fdtd_gauss.setGaussianPulse(Nx / 4 * dx, Ny / 4 * dy, lambda / 4);
            ejecutarSimulacion2D(fdtd_gauss, "gaussiano", "gaussiano", report_mode);
        }
#pragma omp section
        {
            FDTD2D fdtd_sin(Nx, Ny, dx, dy, courantFactor);
            fdtd_sin.setSinusoidalPulse();
            ejecutarSimulacion2D(fdtd_sin, "senosoidal", "senosoidal", report_mode);
        }
    }

    return 0;
}