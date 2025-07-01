# Makefile

CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp -ffast-math
SOURCES = fdtd2D_p.cpp lbm_parallel.cpp
BIN_FDTD = fdtd2D_p
BIN_LBM = lbm_parallel

# Compila si no existen binarios
compile:
	@for src in $(SOURCES); do \
		out=$${src%.cpp}; \
		if [ ! -f $$out ]; then \
			echo "Compilando $$src -> $$out"; \
			$(CXX) $(CXXFLAGS) $$src -o $$out; \
		fi; \
	done

# Ejecuta con banderas: MODE=fdtd/lbm, M, H, R=1
run: compile
	@if [ "$(MODE)" = "fdtd" ]; then \
		bin="./$(BIN_FDTD)"; \
	elif [ "$(MODE)" = "lbm" ]; then \
		bin="./$(BIN_LBM)"; \
	else \
		echo "Error: debes usar MODE=fdtd o MODE=lbm"; exit 1; \
	fi; \
	args=""; \
	if [ "$(R)" = "1" ]; then args="$$args -r"; fi; \
	if [ -n "$(M)" ]; then args="$$args -m $(M)"; fi; \
	if [ -n "$(H)" ]; then args="$$args -h $(H)"; fi; \
	echo "Ejecutando: $$bin $$args"; \
	$$bin $$args

# Benchmark con progreso en la misma línea
benchmark:
	@command -v perf > /dev/null || { echo "Se requiere 'perf'. Instálalo."; exit 1; }
	@if [ ! -f ./$(BIN_FDTD) ]; then echo "ERROR: falta '$(BIN_FDTD)'. Usa 'make compile'."; exit 1; fi
	@if [ ! -f ./$(BIN_LBM) ]; then echo "ERROR: falta '$(BIN_LBM)'. Usa 'make compile'."; exit 1; fi

	@rm -f benchmark_*.csv

	@cores=$$(nproc); \
	total=$(if $(TOTAL),$(TOTAL),10); \
	for bin in $(BIN_LBM) $(BIN_FDTD); do \
		if [ "$$bin" = "$(BIN_LBM)" ]; then \
			mallas="100 500 1000 1500"; \
		else \
			mallas="500 2500 5000 7500"; \
		fi; \
		for m in $$mallas; do \
			out="benchmark_$$bin-m$$m.csv"; \
			echo "hilos,ipc,instructions,cycles,cache_miss,branch_miss,tiempo,mlups" > $$out; \
			for h in $$(seq 1 $$cores); do \
				ipc_sum=0; ins_sum=0; cyc_sum=0; cmiss_sum=0; bmiss_sum=0; time_sum=0; mlup_sum=0; \
				printf "\r[$$bin | malla=$$m | hilos=$$h]   0%%"; \
				i=1; \
				while [ $$i -le $$total ]; do \
					sleep 2; \
					res=$$(perf stat -e cycles,instructions,cache-misses,branch-misses \
					 ./$$bin -r -m $$m -h $$h 2>&1); \
					ipc=$$(echo "$$res" | grep -m1 'insn per cycle' | awk '{print $$4}'); \
					ins=$$(echo "$$res" | grep instructions | awk '{print $$1}' | tr -d ,); \
					cyc=$$(echo "$$res" | grep cycles | awk '{print $$1}' | tr -d ,); \
					cmiss=$$(echo "$$res" | grep cache-misses | awk '{print $$1}' | tr -d ,); \
					bmiss=$$(echo "$$res" | grep branch-misses | awk '{print $$1}' | tr -d ,); \
					etime=$$(echo "$$res" | grep 'seconds time elapsed' | awk '{print $$1}'); \
					mlups=$$(echo "$$res" | grep -i 'Throughput:' | tail -1 | awk '{print $$2}'); \
					ipc_sum=$$(echo "$$ipc_sum + $$ipc" | bc); \
					ins_sum=$$(echo "$$ins_sum + $$ins" | bc); \
					cyc_sum=$$(echo "$$cyc_sum + $$cyc" | bc); \
					cmiss_sum=$$(echo "$$cmiss_sum + $$cmiss" | bc); \
					bmiss_sum=$$(echo "$$bmiss_sum + $$bmiss" | bc); \
					time_sum=$$(echo "$$time_sum + $$etime" | bc); \
					mlup_sum=$$(echo "$$mlup_sum + $$mlups" | bc); \
					percent=$$(echo "scale=0; 100 * $$i / $$total" | bc); \
					printf "\r[$$bin | malla=$$m | hilos=$$h] %3d%%" $$percent; \
					i=$$(expr $$i + 1); \
				done; \
				avg_ipc=$$(echo "scale=3; $$ipc_sum / $$total" | bc); \
				avg_ins=$$(echo "scale=0; $$ins_sum / $$total" | bc); \
				avg_cyc=$$(echo "scale=0; $$cyc_sum / $$total" | bc); \
				avg_cmiss=$$(echo "scale=0; $$cmiss_sum / $$total" | bc); \
				avg_bmiss=$$(echo "scale=0; $$bmiss_sum / $$total" | bc); \
				avg_time=$$(echo "scale=3; $$time_sum / $$total" | bc); \
				avg_mlup=$$(echo "scale=2; $$mlup_sum / $$total" | bc); \
				printf "\r[$$bin | malla=$$m | hilos=$$h] 100%%"; \
				echo "$$h,$$avg_ipc,$$avg_ins,$$avg_cyc,$$avg_cmiss,$$avg_bmiss,$$avg_time,$$avg_mlup" >> $$out; \
			done; \
			echo ""; \
			echo "✅ Benchmark completado para malla de $$m x $$m. Resultados en: $$out"; \
		done; \
	done

# Limpieza
clean:
	@echo "Limpiando archivos..."
	@find . -maxdepth 1 -type f ! -name '*.cpp' ! -name '*.py' ! -name '*.ipynb' ! -name 'Makefile' -exec rm -f {} +