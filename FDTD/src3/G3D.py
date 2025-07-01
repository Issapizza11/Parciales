import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from matplotlib import cm
from multiprocessing import Process

def load_data_3d(file_pattern):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"No se encontraron archivos que coincidan con {file_pattern}")
    
    test_data = np.loadtxt(files[0])
    x_coords = np.unique(test_data[:, 0])
    y_coords = np.unique(test_data[:, 1])
    nx, ny = len(x_coords), len(y_coords)
    
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    all_frames = []

    for file in files:
        data = np.loadtxt(file)
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        Hz = np.zeros((nx, ny))
        
        for row in data:
            if len(row) > 0:
                ix = int(row[0])
                iy = int(row[1])
                Ex[ix, iy] = row[2]
                Ey[ix, iy] = row[3]
                Hz[ix, iy] = row[4]
        
        all_frames.append((Ex, Ey, Hz))
    
    return X, Y, all_frames

def create_3d_animation(X, Y, frames, field, elev, azim, interval, output_file, tag):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    total_frames = len(frames)
    Z0 = {'Ex': frames[0][0], 'Ey': frames[0][1], 'Hz': frames[0][2]}[field]
    Z0 = Z0 / np.max(np.abs(Z0)) if np.max(np.abs(Z0)) > 0 else Z0

    surf = [ax.plot_surface(X, Y, Z0, cmap=cm.coolwarm, linewidth=0,
                            antialiased=True, rstride=1, cstride=1, alpha=0.8)]
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(-1, 1)

    def update(frame_num):
        if frame_num % max(1, total_frames // 20) == 0:
            print(f"[{tag}] Paso {frame_num + 1}/{total_frames}")
        ax.clear()
        Z = {'Ex': frames[frame_num][0], 'Ey': frames[frame_num][1], 'Hz': frames[frame_num][2]}[field]
        Z = Z / np.max(np.abs(Z)) if np.max(np.abs(Z)) > 0 else Z

        surf[0] = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0,
                                  antialiased=True, rstride=1, cstride=1, alpha=0.8)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')
        ax.set_zlabel(f'Intensidad de {field}')
        ax.set_title(f'{field} - Paso {frame_num + 1}/{total_frames}')
        return surf[0],

    print(f"[{tag}] Exportando {output_file} ...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=False)
    anim.save(output_file, writer='pillow', fps=1000 // interval)
    plt.close(fig)
    print(f"[{tag}] Finalizado: {output_file}")

def ejecutar_gif(subdir, campo, elev, azim, interval):
    base_dir = "data2D"
    ruta_archivos = os.path.join(base_dir, subdir, "campos_t*.dat")
    tag = f"{subdir}-{campo}"
    try:
        X, Y, frames = load_data_3d(ruta_archivos)
        nombre_salida = f"{subdir}_{campo}.gif"
        create_3d_animation(X, Y, frames,
                            field=campo,
                            elev=elev,
                            azim=azim,
                            interval=interval,
                            output_file=nombre_salida,
                            tag=tag)
    except Exception as e:
        print(f"❌ [{tag}] Error: {e}")

def main():
    base_dir = "data2D"
    elev = 30
    azim = 45
    interval = 100

    if not os.path.isdir(base_dir):
        print("La carpeta data2D no existe.")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        print("No se encontraron subcarpetas en data2D/")
        return

    campos = ['Ex', 'Ey', 'Hz']
    procesos = []

    for subdir in subdirs:
        for campo in campos:
            p = Process(target=ejecutar_gif, args=(subdir, campo, elev, azim, interval))
            p.start()
            procesos.append(p)

    for p in procesos:
        p.join()

    print("✅ Todos los GIFs han sido generados.")

if __name__ == "__main__":
    main()
