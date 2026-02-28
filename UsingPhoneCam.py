import cv2
import numpy as np
import matplotlib.pyplot as plt
import websocket
import json
import threading
import time

# ================= CONFIG =================
PHONE_IP = "Ip address in sensor server"
IMU_PORT = 8080              

CAMERA_INDEX = 0
address = "IP address of the IP webcam"

MAX_FEATURES = 2000
REDETECT_THRESH = 80
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7

RANSAC_PROB = 0.999
RANSAC_THRESH = 1.0
PLOT_UPDATE_EVERY = 5


# ================= PHONE IMU =================
class PhoneIMU:
    def __init__(self, ip, port):
        self.rotation = np.eye(3)
        self.velocity = np.zeros((3,1))
        self.prev_timestamp = None

        url = f'ws://{ip}:{port}/sensors/connect?types=%5B%22android.sensor.rotation_vector%22,%22android.sensor.linear_acceleration%22%5D'

        self.ws = websocket.WebSocketApp(url, on_message=self.on_message)
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def quat_to_rot(self, qx, qy, qz, qw):
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])

    def on_message(self, ws, message):
        data = json.loads(message)
        sensor_type = data["type"]
        values = data["values"]
        timestamp = data["timestamp"]

        if sensor_type == "android.sensor.rotation_vector":
            qx, qy, qz, qw = values[:4]
            self.rotation = self.quat_to_rot(qx, qy, qz, qw)

        elif sensor_type == "android.sensor.linear_acceleration":
            if self.prev_timestamp is None:
                self.prev_timestamp = timestamp
                return

            dt = (timestamp - self.prev_timestamp) / 1e9
            self.prev_timestamp = timestamp

            acc = np.array(values).reshape(3,1)
            self.velocity += acc * dt


# ================= FEATURE TRACKER =================
class FeatureTracker:
    def detect(self, gray):
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_FEATURES,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE
        )
        if pts is None:
            return np.empty((0,1,2), dtype=np.float32)
        return pts.astype(np.float32)

    def track(self, prev_gray, gray, prev_pts):
        if len(prev_pts) == 0:
            return np.array([]), np.array([])

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        good_prev = prev_pts[status.flatten()==1]
        good_curr = curr_pts[status.flatten()==1]
        return good_prev, good_curr


# ================= POSE ESTIMATOR =================
class PoseEstimator:
    def __init__(self, K):
        self.K = K

    def estimate(self, prev_pts, curr_pts):
        if len(prev_pts) < 8:
            return None, None, 0

        E, _ = cv2.findEssentialMat(
            prev_pts, curr_pts, self.K,
            method=cv2.RANSAC,
            prob=RANSAC_PROB,
            threshold=RANSAC_THRESH
        )

        if E is None:
            return None, None, 0

        inliers, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.K)
        return R, t, inliers


# ================= VISUAL ODOMETRY =================
class VO:
    def __init__(self, K):
        self.tracker = FeatureTracker()
        self.estimator = PoseEstimator(K)
        self.prev_gray = None
        self.prev_pts = None
        self.t_total = np.zeros((3,1))
        self.trajectory = [(0.,0.,0.)]

    def process(self, frame, imu_velocity, imu_rotation):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_pts = self.tracker.detect(gray)
            self.prev_gray = gray
            return

        good_prev, good_curr = self.tracker.track(self.prev_gray, gray, self.prev_pts)

        if len(good_curr) < REDETECT_THRESH:
            self.prev_pts = self.tracker.detect(gray)
            self.prev_gray = gray
            return

        R, t, inliers = self.estimator.estimate(good_prev, good_curr)

        if R is not None and inliers > 10:
            direction = imu_rotation @ t
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                displacement = np.linalg.norm(imu_velocity) * 0.01
                self.t_total += displacement * direction
                self.trajectory.append(tuple(self.t_total.flatten()))

        self.prev_gray = gray
        self.prev_pts = good_curr.reshape(-1,1,2)


# ================= TRAJECTORY SMOOTHER =================
def smooth_trajectory(trajectory, window=15):
    traj = np.array(trajectory)
    smoothed = np.copy(traj)
    half = window // 2

    for i in range(len(traj)):
        s = max(0, i-half)
        e = min(len(traj), i+half+1)
        smoothed[i] = np.mean(traj[s:e], axis=0)

    return smoothed


# ================= TRAJECTORY PLOT =================
class TrajectoryPlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([],[])
        self.ax.set_title("Trajectory")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        plt.show()

    def update(self, traj):
        if len(traj)<2:
            return
        traj = np.array(traj)
        self.line.set_xdata(traj[:,0])
        self.line.set_ydata(traj[:,2])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(address)

    if not cap.isOpened():
        print("Cannot open laptop webcam.")
        return

    imu = PhoneIMU(PHONE_IP, IMU_PORT)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fx = w * 0.9
    cx = w / 2
    cy = h / 2

    K = np.array([[fx,0,cx],[0,fx,cy],[0,0,1]], dtype=np.float64)

    vo = VO(K)
    plot = TrajectoryPlot()

    frame_count = 0

    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vo.process(frame, imu.velocity, imu.rotation)

        if frame_count % PLOT_UPDATE_EVERY == 0:
            plot.update(vo.trajectory)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    smoothed = smooth_trajectory(vo.trajectory)
    plot.update(smoothed.tolist())
    plt.ioff()
    plt.show()

    print("[INFO] Finished.")


if __name__ == "__main__":
    main()