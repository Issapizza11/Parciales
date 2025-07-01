import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from matplotlib import cm
import argparse

def load_data_3d(file_pattern, max_frames=None):
    """Carga los datos y los prepara para visualización 3D"""
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"No se encontraron archivos que coincidan con {file_pattern}")
    
    # Limitar número de frames si se especifica
    if max_frames is not None and max_frames < len(files):
        files = files[:max_frames]
    
    # Cargar el primer archivo para determinar dimensiones
    test_data = np.loadtxt(files[0])
    x_coords = np.unique(test_data[:, 0])
    y_coords = np.unique(test_data[:, 1])
    nx, ny = len(x_coords), len(y_coords)
    
    # Crear malla 3D
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Inicializar arrays para almacenar todos los datos
    all_frames = []
    
    for file in files:
        data = np.loadtxt(file)
        # Extraer componentes del campo
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        Hz = np.zeros((nx, ny))
        
        for row in data:
            if len(row) > 0:  # Ignorar líneas vacías
                ix = int(row[0])
                iy = int(row[1])
                Ex[ix, iy] = row[2]
                Ey[ix, iy] = row[3]
                Hz[ix, iy] = row[4]
        
        all_frames.append((Ex, Ey, Hz))
    
    return X, Y, all_frames, files

def interactive_3d_animation(X, Y, frames, field='Hz', elev=30, azim=45, interval=100):
    """Muestra una animación 3D interactiva"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Configurar límites iniciales
    Ex0, Ey0, Hz0 = frames[0]
    if field == 'Ex':
        Z = Ex0
    elif field == 'Ey':
        Z = Ey0
    else:  # 'Hz'
        Z = Hz0
    
    # Normalizar para mejor visualización
    max_val = np.max(np.abs(Z))
    if max_val > 0:
        Z = Z / max_val
    
    # Crear superficie inicial
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True,
                         rstride=1, cstride=1, alpha=0.8)
    
    # Configuración de la vista
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Y')
    ax.set_zlabel(f'Intensidad de {field}')
    ax.set_title(f'Propagación 3D del campo {field} - Paso 0/{len(frames)}')
    ax.set_zlim(-1, 1)
    
    # Barra de color
    mappable = cm.ScalarMappable(cmap=cm.coolwarm)
    mappable.set_array(Z)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Intensidad normalizada')
    
    # Función de actualización
    def update(frame_num):
        ax.clear()
        Ex, Ey, Hz = frames[frame_num]
        
        if field == 'Ex':
            Z = Ex
        elif field == 'Ey':
            Z = Ey
        else:  # 'Hz'
            Z = Hz
        
        # Normalizar
        max_val = np.max(np.abs(Z))
        if max_val > 0:
            Z = Z / max_val
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                             linewidth=0, antialiased=True,
                             rstride=1, cstride=1, alpha=0.8)
        
        ax.set_zlim(-1, 1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')
        ax.set_zlabel(f'Intensidad de {field}')
        ax.set_title(f'Propagación 3D del campo {field} - Paso {frame_num+1}/{len(frames)}')
        
        return surf,
    
    # Crear animación interactiva
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def main():
    parser = argparse.ArgumentParser(description='Visualización 3D interactiva de simulaciones FDTD 2D')
    parser.add_argument('--tipo', type=str, required=True,
                      choices=['gaussiano', 'senosoidal'],
                      help='Tipo de pulso a visualizar')
    parser.add_argument('--campo', type=str, default='Hz',
                      choices=['Ex', 'Ey', 'Hz'],
                      help='Campo a visualizar en 3D')
    parser.add_argument('--elevacion', type=float, default=30,
                      help='Ángulo de elevación para la vista 3D')
    parser.add_argument('--azimut', type=float, default=45,
                      help='Ángulo de azimut para la vista 3D')
    parser.add_argument('--intervalo', type=int, default=100,
                      help='Intervalo entre frames en milisegundos')
    parser.add_argument('--max_frames', type=int, default=None,
                      help='Número máximo de frames a cargar')
    args = parser.parse_args()

    file_pattern = f"data2D/{args.tipo}/campos_t*.dat"
    
    try:
        X, Y, frames, files = load_data_3d(file_pattern, args.max_frames)
        print(f"Cargados {len(frames)} frames para visualización 3D interactiva")
        print("Mostrando animación... Puedes interactuar con la vista 3D durante la reproducción")
        
        # Mostrar animación interactiva
        anim = interactive_3d_animation(X, Y, frames, 
                                      field=args.campo,
                                      elev=args.elevacion,
                                      azim=args.azimut,
                                      interval=args.intervalo)
    except Exception as e:
        print(f"Error: {e}")
        print("Verifica que:")
        print(f"1. Los archivos existen en {file_pattern}")
        print("2. La simulación FDTD se ejecutó correctamente")
        print("3. Tienes instaladas todas las dependencias (matplotlib, numpy)")

if __name__ == "__main__":
    main()