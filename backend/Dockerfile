FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    libx11-6 libxi6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV TI_ENABLE_GUI=0
ENV TAICHI_SKIP_VERSION_CHECK=ON
ENV TI_LOG_LEVEL=error

RUN git clone --depth=1 https://github.com/FluidSimulator/Wind_Tunnel.git        /simulators/Wind_Tunnel && \
    git clone --depth=1 https://github.com/FluidSimulator/Fire_sim.git            /simulators/Fire_sim && \
    git clone --depth=1 https://github.com/FluidSimulator/water_sim.git           /simulators/water_sim && \
    git clone --depth=1 https://github.com/FluidSimulator/Eulerian_paint_sim.git  /simulators/Eulerian_paint_sim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py sim_worker.py ./

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]