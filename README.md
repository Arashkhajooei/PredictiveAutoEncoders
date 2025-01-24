graph TD
    A[MFCC Input] --> B[Input Coupler]
    B --> C[40D→128D Projection]
    C --> D[Fixed Reservoir]
    D --> E[Output Coupler]
    E --> F[128D→40D Readout]
    F --> G[Predicted MFCC]
