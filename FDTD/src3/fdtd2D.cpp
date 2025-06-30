#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>

// Constantes físicas
const double epsilon0 = 8.854187817e-12;  // Permitividad del vacío (F/m)
const double mu0 = 4 * M_PI * 1e-7;      // Permeabilidad del vacío (H/m)
const double c = 1.0 / sqrt(epsilon0 * mu0); // Velocidad de la luz (m/s)

using Campo2D = std::vector<std::vector<double>>;

void guardarCampos2D(const Campo2D& Ex, const Campo2D& Ey, const Campo2D& Hz, 
                    int paso, const std::string& subcarpeta) {
    // Crear carpeta principal y subcarpeta
    const std::string carpeta_principal = "data2D";
    std::filesystem::create_directories(carpeta_principal + "/" + subcarpeta);

    std::ostringstream nombre;
    nombre << carpeta_principal << "/" << subcarpeta << "/campos_t" 
           << std::setw(4) << std::setfill('0') << paso << ".dat";
    std::ofstream archivo(nombre.str());

    int Nx = Ex.size();
    int Ny = Ex[0].size();

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            archivo << i << " " << j << " " << Ex[i][j] << " " << Ey[i][j] << " " << Hz[i][j] << "\n";
        }
        archivo << "\n"; // Separador para gnuplot
    }

    archivo.close();
}

class FDTD2D {
private:
    int Nx, Ny;             // Número de puntos en la grilla espacial
    double dx, dy;          // Pasos espaciales (m)
    double dt;              // Paso temporal (s)
    double beta_x, beta_y;  // Parámetros de estabilidad
    
    Campo2D Ex;             // Componente x del campo eléctrico
    Campo2D Ey;             // Componente y del campo eléctrico
    Campo2D Hz;             // Componente z del campo magnético
    
public:
    FDTD2D(int numCellsX, int numCellsY, double cellSizeX, double cellSizeY, 
          double courantFactor = 0.5) 
        : Nx(numCellsX), Ny(numCellsY), dx(cellSizeX), dy(cellSizeY) {
        
        // Calcular paso temporal según condición de Courant 2D
        dt = courantFactor / (c * sqrt(1.0/(dx*dx) + 1.0/(dy*dy)));
        beta_x = c * dt / dx;
        beta_y = c * dt / dy;
        
        // Inicializar campos
        Ex.resize(Nx, std::vector<double>(Ny, 0.0));
        Ey.resize(Nx, std::vector<double>(Ny, 0.0));
        Hz.resize(Nx, std::vector<double>(Ny, 0.0));
        
        std::cout << "FDTD 2D inicializado con:\n";
        std::cout << "  Nx = " << Nx << ", Ny = " << Ny << "\n";
        std::cout << "  dx = " << dx << " m, dy = " << dy << " m\n";
        std::cout << "  dt = " << dt << " s, beta_x = " << beta_x << ", beta_y = " << beta_y << "\n";
    }
    
    // Actualizar campo eléctrico E (paso medio temporal)
    void updateE() {
        // Actualizar Ex (depende de Hz en la dirección y)
        for (int i = 1; i < Nx-1; i++) {
            for (int j = 1; j < Ny-1; j++) {
                Ex[i][j] = Ex[i][j] + beta_y * (Hz[i][j] - Hz[i][j-1]);
            }
        }
        
        // Actualizar Ey (depende de Hz en la dirección x)
        for (int i = 1; i < Nx-1; i++) {
            for (int j = 1; j < Ny-1; j++) {
                Ey[i][j] = Ey[i][j] - beta_x * (Hz[i][j] - Hz[i-1][j]);
            }
        }
    }
    
    // Actualizar campo magnético H (paso completo temporal)
    void updateH() {
        for (int i = 0; i < Nx-1; i++) {
            for (int j = 0; j < Ny-1; j++) {
                Hz[i][j] = Hz[i][j] + beta_x * (Ey[i+1][j] - Ey[i][j]) 
                                      - beta_y * (Ex[i][j+1] - Ex[i][j]);
            }
        }
    }
    
    // Ejecutar un paso completo de simulación
    void step() {
        updateE();
        updateH();
    }
    
    // Establecer condición inicial de pulso gaussiano
    void setGaussianPulse(double centerX, double centerY, double width) {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                double x = i * dx;
                double y = j * dy;
                double r2 = pow((x - centerX), 2) + pow((y - centerY), 2);
                Ex[i][j] = exp(-0.5 * r2 / (width*width));
                Ey[i][j] = exp(-0.5 * r2 / (width*width));
                Hz[i][j] = exp(-0.5 * r2 / (width*width));
            }
        }
    }
    
    // Establecer condición inicial de pulso senosoidal
    void setSinusoidalPulse() {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                Ex[i][j] = 0.1 * sin(2 * M_PI * i / 50.0) * sin(2 * M_PI * j / 50.0);
                Ey[i][j] = 0.1 * sin(2 * M_PI * i / 50.0) * sin(2 * M_PI * j / 50.0);
                Hz[i][j] = 0.1 * sin(2 * M_PI * i / 50.0) * sin(2 * M_PI * j / 50.0);
            }
        }
    }
    
    // Acceso a los campos
    const Campo2D& getEx() const { return Ex; }
    const Campo2D& getEy() const { return Ey; }
    const Campo2D& getHz() const { return Hz; }
};

void ejecutarSimulacion2D(FDTD2D& fdtd, const std::string& subcarpeta, const std::string& tipoPulso) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    const int numSteps = 1000;  // Menos pasos porque 2D es más intensivo computacionalmente
    for (int n = 0; n < numSteps; n++) {
        fdtd.step();
        
        if (n % 10 == 0) {
            guardarCampos2D(fdtd.getEx(), fdtd.getEy(), fdtd.getHz(), n, subcarpeta);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Simulación 2D con pulso " << tipoPulso << " completada en: " << duration.count() << "ms\n";
}

int main() {
    // Crear carpeta principal de resultados
    std::filesystem::create_directory("data2D");

    // Parámetros de la simulación
    const double lambda = 1.0e-7;    // Longitud de onda (m)
    const double dx = lambda / 20.0; // Paso espacial (< lambda/10)
    const double dy = lambda / 20.0;
    const int Nx = 200;              // Número de celdas en x
    const int Ny = 200;              // Número de celdas en y
    const double courantFactor = 0.5; // Factor de Courant
    
    // Simulación con pulso gaussiano
    {
        FDTD2D fdtd_gauss(Nx, Ny, dx, dy, courantFactor);
        fdtd_gauss.setGaussianPulse(Nx/4 * dx, Ny/4 * dy, lambda/4);
        ejecutarSimulacion2D(fdtd_gauss, "gaussiano", "gaussiano");
    }
    
    // Simulación con pulso senosoidal
    {
        FDTD2D fdtd_sin(Nx, Ny, dx, dy, courantFactor);
        fdtd_sin.setSinusoidalPulse();
        ejecutarSimulacion2D(fdtd_sin, "senosoidal", "senosoidal");
    }
    
    return 0;
}