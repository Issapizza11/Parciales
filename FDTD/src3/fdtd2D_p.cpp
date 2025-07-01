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
const double epsilon0 = 8.854187817e-12;
const double mu0 = 4 * M_PI * 1e-7;
const double c = 1.0 / sqrt(epsilon0 * mu0);

using Campo2D = std::vector<std::vector<double>>;

void guardarCampos2D(const Campo2D& Ex, const Campo2D& Ey, const Campo2D& Hz, 
                    int paso, const std::string& subcarpeta) {
    const std::string carpeta_principal = "data2D";
    std::filesystem::create_directories(carpeta_principal + "/" + subcarpeta);

    std::ostringstream nombre;
    nombre << carpeta_principal << "/" << subcarpeta << "/campos_t" 
           << std::setw(4) << std::setfill('0') << paso << ".dat";
    std::ofstream archivo(nombre.str());

    int Nx = Ex.size();
    int Ny = Ex[0].size();

    // Versión serial para guardado de datos (más seguro para I/O)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            archivo << i << " " << j << " " << Ex[i][j] << " " << Ey[i][j] << " " << Hz[i][j] << "\n";
        }
        archivo << "\n";
    }

    archivo.close();
}

class FDTD2D {
private:
    int Nx, Ny;
    double dx, dy;
    double dt;
    double beta_x, beta_y;
    
    Campo2D Ex;
    Campo2D Ey;
    Campo2D Hz;
    
public:
    FDTD2D(int numCellsX, int numCellsY, double cellSizeX, double cellSizeY, 
            double courantFactor = 0.5)  // Reducir a 0.3 si persisten problemas
        : Nx(numCellsX), Ny(numCellsY), dx(cellSizeX), dy(cellSizeY) {
    
        double courantMax = 1.0 / (c * sqrt(1.0/(dx*dx) + 1.0/(dy*dy)));
        dt = courantFactor * courantMax;
    
        if(courantFactor >= 1.0) {
            std::cerr << "¡Advertencia! Factor de Courant (" << courantFactor 
                      << ") puede ser inestable. Recomendado < 0.7" << std::endl;
        }
    
        beta_x = c * dt / dx;
        beta_y = c * dt / dy;

        
        Ex.resize(Nx, std::vector<double>(Ny, 0.0));
        Ey.resize(Nx, std::vector<double>(Ny, 0.0));
        Hz.resize(Nx, std::vector<double>(Ny, 0.0));
        
        std::cout << "FDTD 2D inicializado con " << omp_get_max_threads() << " hilos\n";
    }
    
    void updateE() {
        // Paralelización sin collapse para evitar problemas de dependencias
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < Nx-1; i++) {
            for (int j = 1; j < Ny-1; j++) {
                Ex[i][j] += beta_y * (Hz[i][j] - Hz[i][j-1]);
                Ey[i][j] -= beta_x * (Hz[i][j] - Hz[i-1][j]);
            }
        }
    }
    
    void updateH() {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx-1; i++) {
            for (int j = 0; j < Ny-1; j++) {
                Hz[i][j] += beta_x * (Ey[i+1][j] - Ey[i][j]) - beta_y * (Ex[i][j+1] - Ex[i][j]);
            }
        }
    }
    
void step() {
    updateE();
    updateH();
    
    // Verificar valores finitos
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if(!std::isfinite(Ex[i][j]) || !std::isfinite(Ey[i][j]) || !std::isfinite(Hz[i][j])) {
                std::cerr << "¡Error! Valores no finitos detectados en (" 
                          << i << "," << j << ")" << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

    
    void setGaussianPulse(double centerX, double centerY, double width) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double x = i * dx;
                double y = j * dy;
                double r2 = pow(x-centerX,2) + pow(y-centerY,2);
                double val = exp(-0.5*r2/(width*width));
                Ex[i][j] = val;
                Ey[i][j] = val;
                Hz[i][j] = val;
            }
        }
    }
    
    void setSinusoidalPulse() {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double val = 0.1 * sin(2*M_PI*i/50.0) * sin(2*M_PI*j/50.0);
                Ex[i][j] = val;
                Ey[i][j] = val;
                Hz[i][j] = val;
            }
        }
    }
    
    const Campo2D& getEx() const { return Ex; }
    const Campo2D& getEy() const { return Ey; }
    const Campo2D& getHz() const { return Hz; }
};

void ejecutarSimulacion2D(FDTD2D& fdtd, const std::string& subcarpeta, const std::string& tipoPulso) {
    auto start = std::chrono::high_resolution_clock::now();
    
    const int numSteps = 1000;
    for (int n = 0; n < numSteps; n++) {
        fdtd.step();
        
        if (n % 10 == 0) {
            guardarCampos2D(fdtd.getEx(), fdtd.getEy(), fdtd.getHz(), n, subcarpeta);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulación completada en " << duration.count() << " ms\n";
}

int main() {
    omp_set_num_threads(omp_get_max_threads());  //omp_get_max_threads()
    std::filesystem::create_directory("data2D");

    const double lambda = 1.0e-6;
    const double dx = lambda/20.0, dy = lambda/20.0;
    const int Nx = 25, Ny = 25;
    const double courantFactor = 0.3;

    // Simulación gaussiana
    FDTD2D fdtd_gauss(Nx, Ny, dx, dy, courantFactor);
    fdtd_gauss.setGaussianPulse(Nx/4*dx, Ny/4*dy, lambda/4);
    ejecutarSimulacion2D(fdtd_gauss, "gaussiano", "gaussiano");

    // Simulación sinusoidal
    FDTD2D fdtd_sin(Nx, Ny, dx, dy, courantFactor);
    fdtd_sin.setSinusoidalPulse();
    ejecutarSimulacion2D(fdtd_sin, "senosoidal", "senosoidal");

    return 0;
}