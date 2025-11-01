# 🚦 Traffic Light Timing Optimization Using Genetic Algorithms in SUMO

### 🧠 Computational Intelligence Project — Habib University  
**Authors:**  
- Ahsan Siddiqui — [as08155@st.habib.edu.pk](mailto:as08155@st.habib.edu.pk)
- Zohaib Aslam — [za08134@st.habib.edu.pk](mailto:za08134@st.habib.edu.pk)  

---

## 📘 Overview

Efficient traffic signal control is crucial for reducing congestion and improving vehicle flow in urban areas.  
This project implements a **Genetic Algorithm (GA)** to optimize traffic light timings using the **Simulation of Urban MObility (SUMO)** platform.  

The objective is to **minimize the average vehicle waiting time** at a four-way intersection under both **light** and **heavy** traffic scenarios.  

Our GA-based approach evolves the signal phase durations by running SUMO simulations iteratively, evaluating the effectiveness of each configuration, and refining them to improve performance.

---

## 🎯 Objectives

- Implement a GA-based optimization method for traffic light timing.  
- Integrate SUMO simulation with Python via **TraCI API**.  
- Evaluate performance under **light** and **heavy** traffic scenarios.  
- Compare results with a **baseline static configuration**.  
- Visualize improvements in **average vehicle waiting time**.

---

## ⚙️ Technical Stack

| Component | Description |
|------------|-------------|
| **SUMO (Simulation of Urban Mobility)** | Open-source microscopic traffic simulator for modeling traffic networks. |
| **Python** | For GA implementation and SUMO integration. |
| **TraCI API** | Provides real-time control and data exchange between Python and SUMO. |
| **Matplotlib / Pandas** | Used for performance visualization and analysis. |

---

## 🧬 Genetic Algorithm Design

Each GA individual encodes a traffic light configuration:

| Parameter | Description | Range |
|------------|-------------|--------|
| `g1` | Green duration (North-South) | 10–60 sec |
| `y1` | Yellow duration (North-South) | 3–5 sec |
| `g2` | Green duration (East-West) | 10–60 sec |
| `y2` | Yellow duration (East-West) | 3–5 sec |

### GA Parameters
- **Population Size:** 10  
- **Generations:** 20  
- **Crossover:** Single-point  
- **Mutation Rate:** 0.2  
- **Selection Method:** Tournament (size = 2)  
- **Fitness Function:** Average vehicle waiting time (lower is better)

---

## 🚗 SUMO Simulation Setup

The traffic simulation consists of a **four-way intersection** with configurable inflow from two directions.

### Simulation Files
- `net.xml` — Defines the road network (lanes, junctions, signals)  
- `route.xml` — Defines vehicle routes and traffic inflow  
- `cfg.xml` — Combines network and route configurations  
- `randomTrips.py` — Used to generate randomized vehicle trips  

### Traffic Scenarios
| Scenario | Description |
|-----------|--------------|
| **Light Traffic** | Low vehicle inflow (avg. 20 sec between vehicles) |
| **Heavy Traffic** | High vehicle inflow (avg. 5 sec between vehicles) |

---

## 🔄 Training and Testing Protocol

1. **Training Phase**  
   - GA runs on one instance of light and heavy traffic route files.  
   - Each configuration’s performance is evaluated through SUMO.  

2. **Testing Phase**  
   - The best configuration is evaluated on **new, unseen** route files.  
   - Compared against a **baseline configuration** `[42, 3, 42, 3]`.

---

## 📊 Results

| Metric | Baseline | GA-Optimized | Improvement |
|---------|-----------|--------------|--------------|
| **Avg Waiting Time (Test Data)** | 8.49 sec | **3.60 sec** | **57.59% ↓** |

### Best Found Configuration
[11,4,19,4]

---

### Visual Insights
- GA convergence curve shows steady improvement.  
- Diversity plot indicates efficient exploration of search space.  
- Overall, GA outperforms static control across all traffic types.

---

###📈 Future Work
=Integration with Reinforcement Learning for adaptive control.
-Real-time updates from traffic sensors or IoT data.
-Support for multi-intersection networks.
-Optimization based on emissions and fuel consumption.

###📚 References
-R. P. Roess and E. S. Prassas, Time Optimization for Traffic Signal Control Using Genetic Algorithm.
-B. Park and J. D. Schneeberger, A Genetic Algorithm Approach for Traffic Signal Optimization with the TRANSYT Model, Transportation Research Part C, 2003.
-X. Zhang et al., An Intelligent Traffic Signal Control Based on Genetic Algorithm, Springer, 2024.

##🏁 Conclusion
This project demonstrates the potential of bio-inspired optimization for intelligent transportation systems.
By leveraging Genetic Algorithms and SUMO simulations, we achieved a 57.6% reduction in average vehicle waiting time — paving the way for smarter, adaptive urban traffic control.

🧩 Repository Maintainers

👨‍💻 Zohaib Aslam
👨‍💻 Ahsan Siddiqui
