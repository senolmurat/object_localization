import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D

from triangulation import triangulate
from orthographic_localization import fit_to_origin, get_xyz_center
from forecasting import Kalman3D


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def combined_display(image_xy, image_zx, image_zy, img_3d):
    upper_body = np.concatenate((image_zx, image_zy), axis=1)
    lower_bdoy = np.concatenate((image_xy, img_3d), axis=1)
    final_set = np.concatenate((upper_body, lower_bdoy), axis=0)
    return final_set


def inference_and_output_pipeline(image, net, output_layers):
    temp = image.copy()
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = temp.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h, center_x, center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        selected_boxes = np.take(boxes, indexes, 0)[0]
        selected_classes = np.take(class_ids, indexes, 0)[0]
        selected_scores = np.take(confidences, indexes, 0)[0]
        return temp, selected_boxes, selected_classes, selected_scores
    else:
        return temp, None, None, None


def write_bbox_info_on_image(image, boxes, classes, class_ids, scores):
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    color = (120, 200, 160)
    font = cv2.FONT_HERSHEY_PLAIN
    if boxes is not None:
        for i in range(len(boxes)):
            # x, y, w, h, new_center_x, new_center_y, center_x, center_y = boxes[i]
            x, y, w, h, center_x, center_y = boxes[i]
            label = str(classes[class_ids[i]])
            score = int(scores[i] * 100)
            text = "{} ({}%) ({}, {}, {}, {})".format(label, score, center_x, center_y, w, h)
            # color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, text, (x, y + 30), font, 1.5, color, 2)
    return image


def main():
    # Projection matrices
    P_XY = np.array([[-0.0002, 1422.2220, -800.0000, 40000.0000],
                     [1599.9999, 0.0002, -600.0000, 30000.0000],
                     [0.0000, 0.0000, -1.0000, 50.0000]])
    P_XZ = np.array([[2222.2222, 799.9999, 0.0001, 31999.9961],
                     [0.0000, 600.0003, -2499.9998, 45250.0078],
                     [0.0000, 1.0000, 0.0000, 40.0000]])
    P_YZ = np.array([[799.9944, -2222.2180, 0.0020, 31999.7598],
                     [600.0063, -0.0015, -2499.9917, 45250.1797],
                     [1.0000, -0.0000, 0.0000, 40.0000]])

    origin_xy = np.array([[320], [240]], dtype=float)
    origin_zx = np.array([[320], [428]], dtype=float)
    origin_zy = np.array([[320], [428]], dtype=float)

    reference_xy = np.array([[320], [240]], dtype=float)
    reference_zx = np.array([[320], [248]], dtype=float)
    reference_zy = np.array([[320], [248]], dtype=float)

    object_size = 2.5    # Fixed visual size. Should be dynamic after size estimation

    net = cv2.dnn.readNet("yolov3_main.weights", "yolov3_testing.cfg")

    # device = 'cpu'
    # if device == 'cpu':
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #     print('Using CPU device.')
    # elif device == 'gpu':       # OpenCV needs to be manually built with DNN module compatible with CUDA backend
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #     print('Using GPU device.')

    classes = ["ball"]
    images_path_xy = sorted(glob.glob("renders_new/XY/*.jpg"))
    images_path_zx = sorted(glob.glob("renders_new/ZX/*.jpg"))
    images_path_zy = sorted(glob.glob("renders_new/ZY/*.jpg"))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    kalman = Kalman3D()

    import time
    for idx, img_path_xy in enumerate(images_path_xy):
        start = time.time()
        img_path_zx = images_path_zx[idx]
        img_path_zy = images_path_zy[idx]

        img_xy = cv2.imread(img_path_xy)
        img_zx = cv2.imread(img_path_zx)
        img_zy = cv2.imread(img_path_zy)

        img_xy, boxes_xy, classes_xy, scores_xy = inference_and_output_pipeline(img_xy, net, output_layers)
        img_zx, boxes_zx, classes_zx, scores_zx = inference_and_output_pipeline(img_zx, net, output_layers)
        img_zy, boxes_zy, classes_zy, scores_zy = inference_and_output_pipeline(img_zy, net, output_layers)

        detection_time = time.time()
        print(detection_time - start, "seconds (detection)")

        # Get centers of first bounding box, save as points (Convenience, doesn't work with multiple objects in camera)
        if boxes_xy is not None:
            point_xy = np.array([[boxes_xy[0][4]], [boxes_xy[0][5]]], dtype=float)
        else:
            point_xy = None
        if boxes_zx is not None:
            point_zx = np.array([[boxes_zx[0][4]], [boxes_zx[0][5]]], dtype=float)
        else:
            point_zx = None
        if boxes_zy is not None:
            point_zy = np.array([[boxes_zy[0][4]], [boxes_zy[0][5]]], dtype=float)
        else:
            point_zy = None
        # print("XY:   ", point_xy)
        # print("ZX:   ", point_zx)
        # print("ZY:   ", point_zy)

        # Prints all 3D coords obtained from XY, XZ, YZ combinations c(3, 2)
        coords_3d = triangulate(P_XY, P_XZ, P_YZ, point_xy, point_zx, point_zy)
        # triangulate(origin_xy, origin_zx, origin_zy)

        if coords_3d is not None:
            kalman.correct(coords_3d)
        else:
            predicted = kalman.predict()
            coords_3d = np.array([predicted[0][0], predicted[1][0], predicted[2][0]], dtype=float)

        localization_timr = time.time()
        print(localization_timr - detection_time, "seconds (localization)")

        # origin_xy = np.dot(P_XY, coord_xy_xz)
        # origin_xz = np.dot(P_XZ, coord_xy_xz)
        # origin_xy /= origin_xy[2]
        # origin_xz /= origin_xz[2]

        # Change box center origins (instead of top-left corner it is the corner corresponding to orange O's in images)
        # boxes_xy, center_xy = fit_to_origin(img_xy.shape, boxes_xy, dim="xy")
        # boxes_zx, center_zx = fit_to_origin(img_zx.shape, boxes_zx, dim="zx")
        # boxes_zy, center_zy = fit_to_origin(img_zy.shape, boxes_zy, dim="zy")

        # print(str(boxes_xy) + "   x, y")
        # print(str(boxes_zx) + "   x, z")
        # print(str(boxes_zy) + "   y, z")

        # Drawing on images
        img_xy = write_bbox_info_on_image(img_xy, boxes_xy, classes, classes_xy, scores_xy)
        img_zx = write_bbox_info_on_image(img_zx, boxes_zx, classes, classes_zx, scores_zx)
        img_zy = write_bbox_info_on_image(img_zy, boxes_zy, classes, classes_zy, scores_zy)
        # cv2.putText(img_xy, "O", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 127, 255), 2)
        # cv2.putText(img_zx, "O", (0, img_zx.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 127, 255), 2)
        # cv2.putText(img_zy, "O", (img_zy.shape[1] - 20, img_zy.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5,
        #             (0, 127, 255), 2)
        cv2.putText(img_xy, "Y", (img_xy.shape[1] // 2, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 127, 0), 2)
        cv2.putText(img_xy, "X", (20, img_xy.shape[0] // 2), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 127, 0), 2)
        cv2.putText(img_zx, "X", (img_zx.shape[1] // 2, img_zx.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 127, 0), 2)
        cv2.putText(img_zx, "Z", (20, img_zx.shape[0] // 2), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 127, 0), 2)
        cv2.putText(img_zy, "Y", (img_zy.shape[1] // 2, img_zy.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 127, 0), 2)
        cv2.putText(img_zy, "Z", (img_zy.shape[1] - 20, img_zx.shape[0] // 2), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 127, 0), 2)

        # 3D Plot configurations, currently plots ortographic localization coords and not triangulation outputs
        height, width, channels = img_xy.shape
        dpi = 256
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = plt.axes(projection='3d')
        ax.view_init(30, 30)
        # ax.set_xlim3d(0, height)
        # ax.set_ylim3d(0, width)
        # ax.set_zlim3d(0, img_zx.shape[0])
        origin_arrow_x = Arrow3D([-0.5, 25], [0, 0], [0, 0], mutation_scale=20, lw=1.5, arrowstyle="->", linestyle="-",
                                 color=(0.0, 0.0, 1.0, 0.2))
        origin_arrow_y = Arrow3D([0, 0], [-0.5, 25], [0, 0], mutation_scale=20, lw=1.5, arrowstyle="->", linestyle="-",
                                 color=(0.0, 1.0, 0.0, 0.2))
        origin_arrow_z = Arrow3D([0, 0], [0, 0], [-0.5, 25], mutation_scale=20, lw=1.5, arrowstyle="->", linestyle="-",
                                 color=(1.0, 0.0, 0.0, 0.2))
        ax.add_artist(origin_arrow_x)
        ax.add_artist(origin_arrow_y)
        ax.add_artist(origin_arrow_z)
        # x_data, y_data, z_data = get_xyz_center(center_xy, center_zx, center_zy)
        ax.set_xlim3d(20, -20)
        ax.set_ylim3d(20, -20)
        ax.set_zlim3d(0, 20)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.707, 1]))    # 1, 1, sqrt(2)/2 to scale
        if coords_3d is not None:
            x_data = coords_3d[0]
            y_data = coords_3d[1]
            z_data = coords_3d[2]
        else:
            x_data = None
            y_data = None
            z_data = None
        if x_data is not None and y_data is not None and z_data is not None:
            ax.scatter3D(x_data, y_data, z_data, s=50, color="r")
            center_txt = "   (" + str(round(x_data, 2)) + ", " + str(round(y_data, 2)) + ", " + str(round(z_data, 2)) \
                         + ")"
            ax.text(x_data, y_data, z_data, '%s' % center_txt, size=16, zorder=1,
                    color='g')
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = x_data + (np.cos(u) * np.sin(v)) * object_size
            y = y_data + (np.sin(u) * np.sin(v)) * object_size
            z = z_data + (np.cos(v)) * object_size
            ax.plot_surface(x, y, z, color="g", alpha=0.3)
            x_arrow = Arrow3D([20, -20], [y_data, y_data], [z_data, z_data], mutation_scale=2,
                              lw=1, arrowstyle="-", linestyle=":", color=(0.5, 0.0, 0.0, 0.3))
            y_arrow = Arrow3D([x_data, x_data], [20, -20], [z_data, z_data], mutation_scale=2,
                              lw=1, arrowstyle="-", linestyle=":", color=(0.5, 0.0, 0.0, 0.3))
            z_arrow = Arrow3D([x_data, x_data], [y_data, y_data], [20, 0], mutation_scale=2,
                              lw=1, arrowstyle="-", linestyle=":", color=(0.5, 0.0, 0.0, 0.3))
            ax.add_artist(x_arrow)
            ax.add_artist(y_arrow)
            ax.add_artist(z_arrow)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        # ax.grid(False)
        fig.canvas.draw()
        plot_bytes = fig.canvas.tostring_rgb()
        img_3d = np.frombuffer(plot_bytes, 'u1')
        img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        out_img = combined_display(img_xy, img_zx, img_zy, img_3d)
        w_width = 1200
        w_height = 900
        cv2.namedWindow("Inspection Window", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Inspection Window", w_width, w_height)
        cv2.imshow("Inspection Window", out_img)
        # cv2.imwrite('out.jpg', out_img)
        plt.close()
        end = time.time()
        print(end - localization_timr, "seconds (display)")
        print(end - start, "seconds (total)")
        if cv2.waitKey(0) & 0xFF == ord('q'):       # Press any key to move to the next image, press q to quit
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
