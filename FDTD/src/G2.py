import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def cargar_datos(archivo):
    data = np.loadtxt(archivo)
    z = data[:, 0]
    Ex = data[:, 1]
    Hy = data[:, 2]
    return z, Ex, Hy

# Preguntar al usuario
print("Seleccione el conjunto de datos a graficar:")
print("1 - Gaussiano")
print("2 - Senosoidal")
print("3 - Gaussiano Paralelizado")
print("4 - Senosoidal Paralelizado")
opcion = input("Opción: ").strip()

if opcion == "1":
    carpeta = "resultados/gaussiano"
    color_ex = 'green'
    color_hy = 'purple'
    etiqueta = "gaussiano"
elif opcion == "2":
    carpeta = "resultados/senosoidal"
    color_ex = 'orange'
    color_hy = 'blue'
    etiqueta = "senosoidal"
elif opcion == "3":
    carpeta = "resultados_parallel/gaussiano"
    color_ex = 'orange'
    color_hy = 'blue'
    etiqueta = "gaussiano_parallel"
elif opcion == "4":
    carpeta = "resultados_parallel/senosoidal"
    color_ex = 'orange'
    color_hy = 'blue'
    etiqueta = "senosoidal_parallel"

else:
    print("Opción inválida. Finalizando programa.")
    exit(1)

archivos = sorted(glob.glob(f"{carpeta}/campos_t*.dat"))
if not archivos:
    print(f"No se encontraron archivos en {carpeta}")
    exit(1)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 200)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.2)
ax.set_xlabel('z')
ax.set_ylabel('E_x')
ax.set_zlabel('H_y')
ax.set_title(f'Campo EM 3D - {etiqueta}')

line_ex, = ax.plot([], [], [], color=color_ex, label='E_x')
line_hy, = ax.plot([], [], [], color=color_hy, label='H_y')
ax.legend()

def init():
    line_ex.set_data([], [])
    line_ex.set_3d_properties([])
    line_hy.set_data([], [])
    line_hy.set_3d_properties([])
    return line_ex, line_hy

def update(frame):
    archivo = archivos[frame]
    z, Ex, Hy = cargar_datos(archivo)
    line_ex.set_data(z, Ex)
    line_ex.set_3d_properties(np.zeros_like(z))
    line_hy.set_data(z, np.zeros_like(z))
    line_hy.set_3d_properties(Hy)
    ax.set_title(f"{etiqueta} - {os.path.basename(archivo)}")
    return line_ex, line_hy

ani = animation.FuncAnimation(fig, update, frames=len(archivos), init_func=init, blit=False, interval=200)

plt.tight_layout()
plt.show()
