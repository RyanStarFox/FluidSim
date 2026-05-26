Final Project Proposal 

**Project Title:** A Systematic Comparative Study of Particle-Grid Fluid Simulation Methods: FLIP vs. APIC vs. PolyPIC 

1. Introduction & Motivation 

Hybrid particle-grid methods are foundational to modern fluid simulation in computer graphics. While FLIP (Fluid-Implicit-Particle) is a widespread standard, it often suffers from visual noise. Advanced extensions like APIC (Affine Particle-In-Cell) and PolyPIC have been introduced to improve stability, reduce numerical dissipation, and better preserve angular momentum. The objective of this project is to systematically validate and benchmark these three existing simulators. By leveraging the high-quality, open-source implementations provided by the Taichi programming language, we will conduct a rigorous comparison to evaluate the algorithmic trade-offs between numerical accuracy, physical fidelity, and computational overhead. 

2. Proposed Methodology 

Aligning with Option 1, our project will focus on the experimental validation of existing simulators rather than reinventing the numerical solvers. We will utilize the Taichi/Python open-source ecosystem (with proper source acknowledgments in our final report). Our primary engineering focus will be on designing controlled experimental environments, ensuring identical boundary conditions, grid resolutions, and particle counts across all three solvers. This isolates the advection and projection steps, allowing for a fair and rigorous evaluation of each method's core mathematical formulations. 

3. Experimental Setup (Test Scenarios) 

We will design and implement the following standardized test cases to stress-test the simulators: 

* 
**Dam Break:** A classic benchmark to observe large-scale macroscopic fluid movement, boundary collisions, and energy retention over time. 


* 
**Liquid Pouring:** A scenario specifically designed to evaluate complex topological changes, splashing behaviors, and resting state stability. 



4. Evaluation Metrics 

To comprehensively validate the simulators, we will analyze and plot data across the following dimensions: 

* 
**Numerical Dissipation:** Observing the loss of fine-scale details and vorticity over time. 


* 
**Kinetic Energy Conservation:** Tracking and plotting the total kinetic energy of the system to objectively measure energy loss or artificial energy gain. 


* 
**Computational Efficiency:** Profiling the execution time per frame and memory footprint for each method, providing a critical algorithmic performance analysis. 


* 
**Visual Quality:** Rendering high-quality, side-by-side animations to qualitatively assess splash realism, noise reduction, and surface smoothness.
