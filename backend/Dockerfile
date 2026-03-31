FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && rm -rf /var/lib/apt/lists/*

# Clone all 4 simulator repos from the FluidSimulator GitHub org
RUN git clone --depth=1 https://github.com/FluidSimulator/Wind_Tunnel.git        /simulators/Wind_Tunnel && \
    git clone --depth=1 https://github.com/FluidSimulator/Fire_sim.git            /simulators/Fire_sim && \
    git clone --depth=1 https://github.com/FluidSimulator/water_sim.git           /simulators/water_sim && \
    git clone --depth=1 https://github.com/FluidSimulator/Eulerian_paint_sim.git  /simulators/Eulerian_paint_sim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py sim_worker.py ./

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]