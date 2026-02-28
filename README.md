# MT2026_Monocular-Visual-Odometry_Muhtasim_Network_and_Vision
This project implements a basic Monocular Visual Odometry (MVO) system integrated with Inertial Measurement Unit (IMU) data.
The system:
1. Uses a single camera stream for visual motion estimation
2. Uses a smartphone as an IMU for orientation and scale estimation
3. Accumulates poses to generate a trajectory
4. Plots the trajectory in real-time

The pipeline consists of:

Feature Detection
Shi-Tomasi corner detection: Detects strong feature points in each frame

Feature Tracking
Lucas–Kanade Optical Flow: Tracks features between consecutive frames

Pose Estimation
Essential Matrix computation: Recovers relative rotation and translation

IMU Integration
Uses rotation vector and linear acceleration: Computes orientation and velocity

Pose Accumulation: Integrates incremental motion and builds a global trajectory

Trajectory Plotting: Real-time X–Z plane visualization

