import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

def load_data(file_pattern):
    """Carga los datos de los archivos .dat que coinciden con el patrón"""
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"No se encontraron archivos que coincidan con {file_pattern}")
    
    # Cargar el primer archivo para determinar las dimensiones
    test_data = np.loadtxt(files[0])
    nx = len(np.unique(test_data[:, 0]))
    ny = len(np.unique(test_data[:, 1]))
    n_fields = 3  # Ex, Ey, Hz
    
    # Inicializar arrays para almacenar todos los datos
    all_data = np.zeros((len(files), nx, ny, n_fields))
    
    for i, file in enumerate(files):
        data = np.loadtxt(file)
        # Reorganizar los datos en una cuadrícula 2D
        for idx, row in enumerate(data):
            if len(row) > 0:  # Ignorar líneas vacías (separadores)
                ix = int(row[0])
                iy = int(row[1])
                all_data[i, ix, iy, :] = row[2:5]
    
    return all_data, files

def create_animation(data, output_file, title="", field_names=("Ex", "Ey", "Hz"), cmap='RdBu'):
    """Crea una animación a partir de los datos cargados"""
    n_frames, nx, ny, n_fields = data.shape
    
    fig, axes = plt.subplots(1, n_fields, figsize=(6*n_fields, 5))
    if n_fields == 1:
        axes = [axes]
    
    images = []
    for i in range(n_fields):
        vmax = np.max(np.abs(data[:, :, :, i]))
        vmin = -vmax if field_names[i] in ['Ex', 'Ey', 'Hz'] else 0
        
        im = axes[i].imshow(data[0, :, :, i], cmap=cmap, vmin=vmin, vmax=vmax, 
                          origin='lower', extent=[0, nx, 0, ny])
        axes[i].set_title(field_names[i])
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
        images.append(im)
    
    fig.suptitle(title)
    plt.tight_layout()

    def update(frame):
        for i in range(n_fields):
            images[i].set_array(data[frame, :, :, i])
        return images
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    anim.save(output_file, writer='ffmpeg', fps=10, dpi=100)
    plt.close()
    print(f"Animación guardada como {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualización de simulaciones FDTD 2D')
    parser.add_argument('--tipo', type=str, required=True, 
                       choices=['gaussiano', 'senosoidal'],
                       help='Tipo de pulso a visualizar (gaussiano o senosoidal)')
    parser.add_argument('--campo', type=str, default='todos',
                       choices=['Ex', 'Ey', 'Hz', 'todos'],
                       help='Campo específico a visualizar o todos')
    parser.add_argument('--output', type=str, default='animacion.mp4',
                       help='Nombre del archivo de salida para la animación')
    args = parser.parse_args()

    # Patrón de búsqueda de archivos
    file_pattern = f"data2D/{args.tipo}/campos_t*.dat"
    
    try:
        # Cargar todos los datos
        all_data, files = load_data(file_pattern)
        print(f"Se cargaron {len(files)} archivos de datos")
        
        # Seleccionar campos a visualizar
        if args.campo == 'Ex':
            field_idx = [0]
            field_names = ['Ex']
        elif args.campo == 'Ey':
            field_idx = [1]
            field_names = ['Ey']
        elif args.campo == 'Hz':
            field_idx = [2]
            field_names = ['Hz']
        else:  # 'todos'
            field_idx = [0, 1, 2]
            field_names = ['Ex', 'Ey', 'Hz']
        
        # Extraer solo los campos seleccionados
        plot_data = all_data[:, :, :, field_idx]
        
        # Crear animación
        output_name = f"{args.tipo}_{args.campo}_{args.output}"
        create_animation(plot_data, output_name, 
                        title=f"Evolución FDTD 2D - Pulso {args.tipo}", 
                        field_names=field_names)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de que:")
        print(f"1. Los archivos de datos existen en {file_pattern}")
        print("2. Has ejecutado primero la simulación en C++")
        print("3. El directorio data2D está en el mismo lugar que este script")

if __name__ == "__main__":
    main()