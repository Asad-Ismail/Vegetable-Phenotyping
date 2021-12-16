import numpy as np
import cv2
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('All packages imported Successfully for pepepr Phenotyping!!')

def min_max_segment(ref, xs, ys):
    """Min Max of fruit"""
    segments = []
    m_p = np.argwhere(xs == ref)
    y_p = ys[m_p]
    ymax = np.max(y_p)
    ymin = np.min(y_p)
    segments.append([(ref, ymin), (ref, ymax), ymax - ymin])
    return segments

def find_max_height(xs, ys):
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


def ray_trace_segment(img, init, direction="y", to_origin=1):
    """Ray trace from the inital points to some end  like end of non zero pixels in image. Is useful for fruits that turns and make an S shape
    Inputs and outputs are different based on x or y. """
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
                # find last non zero point for that x, Replace below 4 with some more dynamic threshold
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

def find_mid_height_width(tops, bottoms):
    """Mid height width of fruit"""
    mid_height = len(tops) // 2
    width_points = [tops[mid_height], bottoms[mid_height]]
    width = bottoms[mid_height][1] - tops[mid_height][1]
    return width_points, width

def find_mid_width_height(xs, ys):
    """Mid width height of fruit"""
    y_sorted = sorted(set(ys))
    mid_y = y_sorted[len(y_sorted) // 2]
    y_indexs = np.argwhere(ys == mid_y)
    matched_xs = xs[y_indexs]
    min_x = int(min(matched_xs))
    max_x = int(max(matched_xs))
    height_points = [(min_x, mid_y), (max_x, mid_y)]
    height = max_x - min_x
    return height_points, height

def find_max_width(tops, bottoms):
    """Max width of fruit"""
    max_index, max_width = None, -1
    for i in range(len(tops)):
        width = bottoms[i][1] - tops[i][1]
        if width > max_width:
            max_width = width
            max_index = i
    width_points = [tops[max_index], bottoms[max_index]]
    return width_points, max_width

def find_curve_length(top, bottom, mask,ign_pct=15):
    """Curved backbone length of fruit ign some pct of fruit to have more accurate measurement"""
    c_length = []
    # curved length causes issues at the borders so ignoring length at borders
    ign_pct = ign_pct / 100
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
    #draw complete ellipse form angle 0 to 360
    cv2.ellipse(ellipse_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, 255)
    elipse_contours, _ = cv2.findContours(
        ellipse_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    assert len(contours) == 1, "More than one contours in ellipse"
    elipse_contour = elipse_contours[0]
    error = cv2.matchShapes(cont, elipse_contour, 1, 0.0)
    return (x,y,MA,ma,angle), error

def blockiness(top, bottom,ign_pct=20):
    """Blockiness at different points of fruits"""
    ign_pct = ign_pct / 100
    bottomindex = int(len(top) * ign_pct)
    midindex = int(len(top) // 2)
    topindex = int(len(top) * (1 - ign_pct))
    bottom_block = [top[bottomindex], bottom[bottomindex]]
    mid_block = [top[midindex], bottom[midindex]]
    top_block = [top[topindex], bottom[topindex]]
    return bottom_block, mid_block, top_block

def box_fit(mask):
    """Find bounding box of the detected cropped mask"""
    x, y, w, h = cv2.boundingRect(mask)
    return (x,y,w,h)

def eucledian_distance(points,ppx=None,ppy=None):
    length=0
    if ppx and ppy:
        for i in range(1,len(points)):
            length+=np.sqrt(((points[i][0]-points[i-1][0])**2)*ppx +  ((points[i][1]-points[i-1][1])**2)*ppy)
    else:
        for i in range(1,len(points)):
            length+=np.sqrt(((points[i][0]-points[i-1][0])**2) +  ((points[i][1]-points[i-1][1])**2))
    return length


def convert_pixels_to_measure(area,perimeter,mid_height_width_p,max_width_p,mid_width_height_p,max_height_p,c_length,widths,u_block,l_block,ppx=None,ppy=None):
    """Convert pixels to real units cmm/mm if ppx and ppy are defined otherwise find lengths of curves"""
    if ppx and ppy:
        area=area*ppx
        perimeter=perimeter*ppx
    mid_height_width=eucledian_distance(mid_height_width_p,ppx,ppy)
    max_width=eucledian_distance(max_width_p,ppx,ppy)
    mid_width_height=eucledian_distance(mid_width_height_p,ppx,ppy)
    max_height=eucledian_distance(max_height_p,ppx,ppy)
    c_length=eucledian_distance(c_length,ppx,ppy)
    if ppy:
        for i,w in enumerate(widths):
            widths[i]=w*ppy
    u_block=eucledian_distance(u_block,ppx,ppy)
    l_block=eucledian_distance(l_block,ppx,ppy)
        
    return (area,perimeter,mid_height_width,max_width,mid_width_height,max_height,c_length,widths,u_block,l_block)

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
    box,
    ellipse,
    img,
    line_width=5,
):
    """Visualization of phenotypes"""
    vis_imgs = []

    vis_img = img.copy()
    for i in range(len(perimeter)):
        cv2.circle(vis_img, tuple(perimeter[i]), line_width, (255, 0, 0))
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
    
   
    # Ellipse vis
    vis_img = img.copy()
    x,y,MA,ma,angle=ellipse
    cv2.ellipse(vis_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, (255,0,0),line_width)
    vis_imgs.append(vis_img)
    
    # Box vis
    vis_img = img.copy()
    x,y,w,h=box
    vis_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),line_width)
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
        "Elipse",
        "Rectangle",
    ]
    
    assert len(labels)==len(vis_imgs), f"Label length {len(labels)}  and vis images {len(vis_imgs)} not equal"
    return vis_imgs,labels

def phenotype_measurement_pepper(img, patch,ppx=None,ppy=None):
    """
    Input: Croped fruit mask, croped fruit images, ppx and ppy
    Returns:area,
        perimeter,
        mid_height_width,
        max_width,
        mid_width_height,
        max_height,
        c_length,
        widths,
        maxheight_to_maxwidth,
        mid_width_height_to_mid_height_width,
        c_length_to_mid_height_width,
        proximal_block,
        distal_block,
        triangle,
        ellipse_err,
        box_ratio,
        vis_imgs,
        labels 
        if number of points are greater than 10 otherwise returns 18 None
    """

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    assert (
        len(contours) == 1
    ), "Outer countour should be one. Multiple contours per fruit not supported!!"
    
    points = contours[0].squeeze()
    if points.size<10:
        print(f"Number of points are too less to calulate phenotpyes meaningfully Returning None!!")
        return tuple(None for _ in range(18))
    # points are perimeter
    points = np.array(sorted(points, key=lambda x: x[0]))
    # Perimeter fruit
    perimeter = cv2.arcLength(contours[0],True)
    # Area fruit
    area = cv2.contourArea(contours[0])
    area_points = np.where(img != 0)
    xs = points[:, 0]
    ys = points[:, 1]
    # sort xs
    u_xs = sorted(set(xs))
    top = []
    bottom = []
    widths = []
    # lengths
    max_height_p, max_height_v = find_max_height(xs, ys)
    mid_width_height_p, mid_width_height_v = find_mid_width_height(xs, ys)

    for i, x in enumerate(u_xs):
        # segments=ray_trace_segment(img,x)
        segments = min_max_segment(x, xs, ys)
        for segment in segments:
            top.append(segment[0])
            bottom.append(segment[1])
            widths.append(segment[2])

    assert (len(top) == len(bottom) == len(widths)), "The top, bottom and width points are not equal"

    # widths and curve lengths
    max_width_p, max_midth_v = find_max_width(top, bottom)
    mid_height_width_p, mid_height_width_v = find_mid_height_width(top, bottom)
    c_length = find_curve_length(top, bottom, img)
    b_block, m_block, t_block = blockiness(top, bottom)
    
    ellipse,ellipse_err=ellipse_fitting(img)
    
    box=box_fit(img)
    
    vis_imgs,labels = vis_phenotypes(
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
        box,
        ellipse,
        patch.copy(),
    )
    # median_width=statistics.median(widths)
    area,perimeter,mid_height_width,max_width,mid_width_height,max_height,c_length,widths,u_block,l_block=\
    convert_pixels_to_measure(area,perimeter,mid_height_width_p,max_width_p,mid_width_height_p,max_height_p,c_length,widths,t_block,b_block,ppx,ppy)
    # shape metrics
    #metric 8
    maxheight_to_maxwidth=max_height/max_width
    #metrix 9
    mid_width_height_to_mid_height_width=mid_width_height/mid_height_width
    # metric 10
    c_length_to_mid_height_width=c_length/mid_height_width
    
    proximal_block=u_block/mid_height_width
    distal_block=l_block/mid_height_width
    
    triangle=u_block/l_block
    
    return (
        area,
        perimeter,
        mid_height_width,
        max_width,
        mid_width_height,
        max_height,
        c_length,
        widths,
        maxheight_to_maxwidth,
        mid_width_height_to_mid_height_width,
        c_length_to_mid_height_width,
        proximal_block,
        distal_block,
        triangle,
        ellipse_err,
        box[2]/box[3],
        vis_imgs,
        labels
    )


# Uncomment to run it as standalone lambda function
# def lambda_handler(event, context):
#     """
#     Accepts an action and a number, performs the specified action on the number,
#     and returns the result.
#     :param event: The event dict that contains the parameters sent when the function
#                   is invoked.
#     :param context: The context in which the function is called.
#     :return: The result of the specified action.
#     """
#     logger.info('Event: %s', event)
#     #change mask and fruit_patch to input mask and RGB detected fruit
#     # Image is a temp image mask of the detected fruit in current directory can be changed to the requsested Image
#     image_name="corrected_mask.jpg"
#     mask=cv2.imread(image_name,-1)
#     #binarize the mask
#     ret, mask = cv2.threshold(mask, 127, 255, 0)
#     # Fruit patch is the cropped fruit used for visualization
#     fruit_patch=mask.copy()
#     assert len(mask.shape) == 2, "Should be Gray scale Mask"
#     if len(fruit_patch.shape)==2:
#         fruit_patch=cv2.cvtColor(fruit_patch, cv2.COLOR_GRAY2BGR)      
#     results={}
#     (
#     area,
#     perimeter,
#     mid_height_width,
#     max_width,
#     mid_width_height,
#     max_height,
#     c_length,
#     widths,
#     maxheight_to_maxwidth,
#     mid_width_height_to_mid_height_width,
#     c_length_to_mid_height_width,
#     proximal_block,
#     distal_block,
#     triangle,
#     ellipse_err,
#     square_aspect_ratio,
#     vis_imgs,
#     labels
#     ) = phenotype_measurement_pepper(mask, fruit_patch)
#     #write results to dictioanry
#     if "Image" not in results: 
#         results["Image"]=[image_name]
#         results["Area"]=[area]
#         results["Perimeter"]=[perimeter]
#         results["Mid_Width"]=[mid_height_width]
#         results["Max_Width"]=[max_width]
#         results["Mid_Height"]=[mid_width_height]
#         results["Max_Height"]=[max_height]
#         results["Curved_Height"]=[c_length]
#         results["widths"]=[widths]
#         results["maxheight_to_maxwidth"]=[maxheight_to_maxwidth]
#         results["midheight_to_midwidth"]=[mid_width_height_to_mid_height_width]
#         results["curveheight_to_midwidth"]=[c_length_to_mid_height_width]
#         results["proximal_block"]=[proximal_block]
#         results["distal_block"]=[distal_block]
#         results["triangle"]=[triangle]
#         results["ellipse_err"]=[ellipse_err]
#         results["square_aspect_ratio"]=[square_aspect_ratio]
#     else:
#         results["Image"].append(image_name)
#         results["Area"].append(area)
#         results["Perimeter"].append(perimeter)
#         results["Mid_Width"].append(mid_height_width)
#         results["Max_Width"].append(max_width)
#         results["Mid_Height"].append(mid_width_height)
#         results["Max_Height"].append(max_height)
#         results["Curved_Height"].append(c_length)
#         results["widths"].append([widths])
#         results["maxheight_to_maxwidth"].append(maxheight_to_maxwidth)
#         results["midheight_to_midwidth"].append(mid_width_height_to_mid_height_width)
#         results["curveheight_to_midwidth"].append(c_length_to_mid_height_width)
#         results["proximal_block"].append(proximal_block)
#         results["distal_block"].append(distal_block)
#         results["triangle"].append(triangle)
#         results["ellipse_err"].append(ellipse_err)
#         results["square_aspect_ratio"].append(square_aspect_ratio)
    
#     results_df=pd.DataFrame.from_dict(results,orient='index').transpose()
#     results_json = results_df.to_json(orient="index")
#     logger.info('Calculated result %s', results)
#     logger.info('Calculated result Json format %s', results_json)
#     #response = {'result': results}
#     return results_json
