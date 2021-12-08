from utils import *
import statistics


def ray_trace_segment(img, init, direction="y", to_origin=1):
    """Ray trace from the inital points to some end  like end of non zero pixels in image. Is useful for fruits that turns and make an S shape
    Inputs and outputs are different based on x or y. May be better to write a differnt function for both"""
    assert len(img.shape) == 2, "Should be Gray scale image"
    h, w = img.shape
    if direction == "y":
        # Go downwards
        increment = 1
        x = init
        y = 0
        segments = []
        # Go along the height of image to find segments
        while y < h:
            firstPoint = None
            secondPoint = None
            if img[y, x]:  # find first non zero point
                firstPoint = (x, y)
                while y < h and img[y, x]:
                    y += increment
                # find last non zero point for that x, Replace below 4 with some more sensible threshold
                if y - firstPoint[1] > 4:
                    secondPoint = (x, y)
                if firstPoint and secondPoint:
                    segments.append(
                        [firstPoint, secondPoint, secondPoint[1] - firstPoint[1]]
                    )
            y += increment
    elif direction == "x":
        x, y = init
        if to_origin:
            # Go Left
            increment = -1
        else:
            # Go Right
            increment = 1
        segments = []
        x += increment
        while x < w and x >= 0:
            if img[y, x]:
                segments.append((x, y))
            x += increment
    return segments


def min_max_segment(ref, xs, ys):
    """Min Max of fruit"""
    segments = []
    m_p = np.argwhere(xs == ref)
    y_p = ys[m_p]
    ymax = np.max(y_p)
    ymin = np.min(y_p)
    segments.append([(ref, ymin), (ref, ymax), ymax - ymin])
    return segments


def max_height(xs, ys):
    """Max height of fruit"""
    left_index = np.argwhere(xs == xs[0])
    y_lefts = set(ys[left_index].flatten())
    right_index = np.argwhere(xs == xs[-1])
    y_rights = set(ys[right_index].flatten())
    y_lefts = list(y_lefts)
    # print(f"The length of lengths is {len(y_lefts)}")
    y = y_lefts[len(y_lefts) // 2]
    height_points = [(xs[0], y), (xs[-1], y)]
    height = xs[-1] - xs[0]
    return height_points, height


def mid_height_width(tops, bottoms):
    """Mid height width of fruit"""
    mid_height = len(tops) // 2
    width_points = [tops[mid_height], bottoms[mid_height]]
    width = bottoms[mid_height][1] - tops[mid_height][1]
    return width_points, width


def mid_width_height(xs, ys):
    """Mid width height of fruit"""
    y_sorted = sorted(set(ys))
    mid_y = y_sorted[len(y_sorted) // 2]
    y_indexs = np.argwhere(ys == mid_y)
    matched_xs = xs[y_indexs]
    min_x = int(min(matched_xs))
    max_x = int(max(matched_xs))
    height_points = [(min_x, mid_y), (max_x, mid_y)]
    height = max_x - min_x
    # print(f"Height Points {height_points}")
    return height_points, height


def max_width(tops, bottoms):
    """Max width of fruit"""
    max_index, max_width = None, -1
    for i in range(len(tops)):
        width = bottoms[i][1] - tops[i][1]
        if width > max_width:
            max_width = width
            max_index = i
    width_points = [tops[max_index], bottoms[max_index]]
    return width_points, max_width


def curve_length(top, bottom, mask):
    """Curved backbone length of fruit"""
    c_length = []
    # curved length causes issues at the borders so ignoring length at borders
    ign_pct = 15 / 100
    for i in range(int(len(top) * ign_pct), int(len(top) * (1 - ign_pct))):
        c_length.append((top[i][0], round((top[i][1] + bottom[i][1]) / 2)))
    # Ray trace to extend the curve length on left
    segment_left = ray_trace_segment(mask, c_length[0], direction="x", to_origin=1)
    c_length = segment_left + c_length
    # Ray trace to extend the curve length on right
    segment_right = ray_trace_segment(mask, c_length[-1], direction="x", to_origin=0)
    c_length = c_length + segment_right
    return c_length


def ellipse_fitting(mask):
    """Ellipse fitting of fruit"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1, "More than one contours in ellipse mask"
    cont = contours[0]
    (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
    x, y = int(x), int(y)
    MA, ma = int(MA), int(ma)
    ellipse_img = np.zeros_like(mask)
    cv2.ellipse(ellipse_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, 255)
    elipse_contours, _ = cv2.findContours(
        ellipse_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    assert len(contours) == 1, "More than one contours in ellipse"
    elipse_contour = elipse_contours[0]
    error = cv2.matchShapes(cont, elipse_contour, 1, 0.0)
    return ellipse_img, error


def blockiness(top, bottom):
    """Blockiness at different points of fruits"""
    ign_pct = 15 / 100
    bottomindex = int(len(top) * ign_pct)
    midindex = int(len(top) // 2)
    topindex = int(len(top) * (1 - ign_pct))
    bottom_block = [top[bottomindex], bottom[bottomindex]]
    mid_block = [top[midindex], bottom[midindex]]
    top_block = [top[topindex], bottom[topindex]]
    return bottom_block, mid_block, top_block


def vis_phenotypes(
    perimeter,
    area,
    max_height,
    mid_width_height,
    max_width,
    mid_height_width,
    curved_length,
    bottom_block,
    mid_block,
    top_block,
    img,
    line_width=5,
):
    """Visualization of phenotypes"""
    fig = plt.figure(figsize=(10, 10))
    nrows = 6
    ncols = 2
    vis_imgs = []

    vis_img = img.copy()
    for i in range(len(perimeter)):
        cv2.circle(vis_img, perimeter[i], line_width, (255, 0, 0))
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, max_height[0], max_height[1], (255, 0, 0), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, max_width[0], max_width[1], (0, 255, 0), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, mid_width_height[0], mid_width_height[1], (0, 0, 255), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    for i in range(len(curved_length)):
        cv2.circle(vis_img, curved_length[i], line_width, (255, 0, 0))
    vis_imgs.append(vis_img)
    # Shape
    # max height to maximum width
    vis_img = img.copy()
    cv2.line(vis_img, max_height[0], max_height[1], (255, 0, 0), line_width)
    cv2.line(vis_img, max_width[0], max_width[1], (0, 255, 0), line_width)
    vis_imgs.append(vis_img)
    # Height mid width/ width mid height
    vis_img = img.copy()
    cv2.line(vis_img, mid_width_height[0], mid_width_height[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Top blockiness/ Mid height Width
    vis_img = img.copy()
    cv2.line(vis_img, top_block[0], top_block[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Bottom blockiness/ Mid height Width
    vis_img = img.copy()
    cv2.line(vis_img, bottom_block[0], bottom_block[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Upper blockiness/ Bottom blockiness
    vis_img = img.copy()
    cv2.line(vis_img, top_block[0], top_block[1], (0, 0, 255), line_width)
    cv2.line(vis_img, bottom_block[0], bottom_block[1], (255, 255, 0), line_width)
    vis_imgs.append(vis_img)

    labels = [
        "Perimeter",
        "Max height",
        "Max Width",
        "Mid Width Height",
        "Mid Height Width",
        "Curved Length",
        "max height/max width",
        "height mid-width/width mid-height",
        "Proximal Fruit Blokiness",
        "Distal Fruit Blokiness",
        "Fruit Shape Triangle",
    ]
    for i, patch in enumerate(vis_imgs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(labels[i])
        ax.imshow(patch[..., ::-1])
    return fig


def phenotype_measurement_pepper(img, patch, skip_outliers=None, pct_width=0.4):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert (
        len(contours) == 1
    ), "Outer countour should be one. Multiple contours per fruit not supported!!"
    # print(f"Shape of contours {contours[0].shape}")
    points = contours[0].squeeze()
    # points are perimeter
    points = np.array(sorted(points, key=lambda x: x[0]))
    # Area points
    area_points = np.where(img != 0)
    xs = points[:, 0]
    ys = points[:, 1]
    # sort xs
    u_xs = sorted(set(xs))
    top = []
    bottom = []
    widths = []
    # lengths
    max_height_p, max_height_v = max_height(xs, ys)
    mid_width_height_p, mid_width_height_v = mid_width_height(xs, ys)

    for i, x in enumerate(u_xs):
        # segments=ray_trace_segment(img,x)
        segments = min_max_segment(x, xs, ys)
        for segment in segments:
            top.append(segment[0])
            bottom.append(segment[1])
            widths.append(segment[2])

    assert (
        len(top) == len(bottom) == len(widths)
    ), "The top, bottom and width points are not equal"

    # widths and curve lengths
    max_width_p, max_midth_v = max_width(top, bottom)
    mid_height_width_p, mid_height_width_v = mid_height_width(top, bottom)
    c_length = curve_length(top, bottom, img)
    b_block, m_block, t_block = blockiness(top, bottom)
    fig = vis_phenotypes(
        points,
        area_points,
        max_height_p,
        mid_width_height_p,
        max_width_p,
        mid_height_width_p,
        c_length,
        b_block,
        m_block,
        t_block,
        patch,
    )
    # median_width=statistics.median(widths)
    return (
        points,
        area_points,
        max_height_v,
        mid_width_height_v,
        max_midth_v,
        mid_height_width_v,
        fig,
    )


def vis_dims(img, length, width):
    vis_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print(f"length of width points {len(width)}")
    for i in range(len(width)):
        cv2.circle(vis_img, width[i][0], 2, (0, 0, 255))
        cv2.circle(vis_img, width[i][1], 2, (255, 0, 255))
        cv2.line(vis_img, width[i][0], width[i][1], (0, 255, 255), 2)
    length = np.array(length, dtype=np.int32)
    vis_img = cv2.polylines(vis_img, [length], False, (0, 0, 255), 2)
    return vis_img


def curve_measurements(points, ppx, ppy):
    dist = 0
    for i in range(1, len(points)):
        dx2 = ((points[i][0] - points[i - 1][0]) * ppx) ** 2
        dy2 = ((points[i][1] - points[i - 1][1]) * ppy) ** 2
        dist += np.sqrt(dx2 + dy2)
    return dist
