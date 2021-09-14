# zx = 1540, 913
# zy = 60, 913
# xy = 1367, 1167

def fit_to_origin(shape, boxes, dim="xy"):
    new_boxes = []
    height, width, channels = shape
    if dim == "xy":
        x_origin = 1367
        y_origin = 1167
    if dim == "zx":
        x_origin = 1540
        y_origin = 913
    if dim == "zy":
        x_origin = 60
        y_origin = 913
    if boxes is not None:
        for i in range(len(boxes)):
            x, y, w, h, center_x, center_y = boxes[i]
            new_center_x = abs(center_x - x_origin)
            new_center_y = abs(center_y - y_origin)
            new_boxes.append([x, y, w, h, new_center_x, new_center_y, center_x, center_y])
        return new_boxes, (new_center_x, new_center_y)  # Convenience tuple, major bug for multiple objects in image
    else:
        return None, None


def get_xyz_center(center_xy, center_zx, center_zy):
    if center_xy is not None and center_zx is not None and center_zy is not None:
        center_x = (center_xy[1] + center_zx[0]) // 2
        center_y = (center_xy[0] + center_zy[0]) // 2
        center_z = (center_zx[1] + center_zy[1]) // 2
        return center_x, center_y, center_z
    elif center_xy is None and center_zx is not None and center_zy is not None:
        center_x = center_zx[0]
        center_y = center_zy[0]
        center_z = (center_zx[1] + center_zy[1]) // 2
        return center_x, center_y, center_z
    elif center_xy is not None and center_zx is None and center_zy is not None:
        center_x = center_xy[1]
        center_y = (center_xy[0] + center_zy[0]) // 2
        center_z = center_zy[1]
        return center_x, center_y, center_z
    elif center_xy is not None and center_zx is not None and center_zy is None:
        center_x = (center_xy[1] + center_zx[0]) // 2
        center_y = center_xy[0]
        center_z = center_zx[1]
        return center_x, center_y, center_z
    else:
        return None, None, None
