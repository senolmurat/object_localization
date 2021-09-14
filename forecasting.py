import cv2
import numpy as np


class Kalman3D:
    dt = 1.0
    v = dt  # Velocity
    a = 0.5 * np.power(dt, 2)  # Acceleration
    T = np.array([[1, 0, 0, v, 0, 0, a, 0, 0], [0, 1, 0, 0, v, 0, 0, a, 0], [0, 0, 1, 0, 0, v, 0, 0, a],
                  [0, 0, 0, 1, 0, 0, v, 0, 0], [0, 0, 0, 0, 1, 0, 0, v, 0], [0, 0, 0, 0, 0, 1, 0, 0, v],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
    M = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]], np.float32)
    kalman = cv2.KalmanFilter(9, 3)
    kalman.transitionMatrix = T
    kalman.measurementMatrix = M

    def correct(self, coords_3d):
        measured = np.array([[np.float32(coords_3d[0])], [np.float32(coords_3d[1])], [np.float32(coords_3d[2])]])
        self.kalman.correct(measured)
        return self.predict()

    def predict(self):
        predicted = self.kalman.predict()
        return predicted
