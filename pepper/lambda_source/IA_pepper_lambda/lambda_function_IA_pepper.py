import json
from aws import AWS
from colormath.color_objects import LabColor, LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
import cv2
import numpy as np
import math
from phenotypes_utils import *

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def crop_mask(image, mask):
    [rows, columns] = np.where(mask)
    white = np.zeros(image.shape)
    mask = np.reshape(mask, mask.shape + (1,))
    masked = np.where(mask, image, white).astype(np.uint8)
    cropped = masked[min(rows, default=0):max(rows, default=0), min(
        columns, default=0): max(columns, default=0)]
    return cropped

def skeletonize (mask, image):
    show = False
    m = cv2.convertScaleAbs(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rect = cv2.minAreaRect(contours[0])
    if rect[1][0] > rect[1][1]:
        image  = rotate_bound(image, abs(rect[2] + 90))
        mask  = rotate_bound(mask, abs(rect[2] + 90))
    else:
        image  = rotate_bound(image, abs(rect[2]+180))
        mask  = rotate_bound(mask, abs(rect[2]+180))

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approx = cv2.approxPolyDP(mask_contours[0], 0.02 * cv2.arcLength(mask_contours[0], True), True)
    approx_rect = cv2.minAreaRect(approx)
    curvature = 0

    if len(approx) > 5 and not((float(approx_rect[1][0]/approx_rect[1][1]) > 0.7) and (float(approx_rect[1][0]/approx_rect[1][1]) < 1.3)):
        
        skel = cv2.ximgproc.thinning(mask.astype(np.uint8), None, cv2.ximgproc.THINNING_ZHANGSUEN)
        skel_contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        biggest_contour = max(skel_contours, key = cv2.contourArea)
        biggest_contour = biggest_contour[:int(len(biggest_contour)/2)]
        extend_mask = np.zeros(mask.shape)
        def find_new_point(point1, point2, top):
            if point1[0][0] - point2[0][0] != 0:
                slope = (point1[0][1] - point2[0][1]) / (point1[0][0] - point2[0][0]) 
            else:
                slope = (point1[0][1] - point2[0][1])
            if slope > 0 and top:
                x = point1[0][0] - 100
            elif slope < 0 and top:
                x = point1[0][0] + 100
            elif slope > 0 and not top:
                x = point1[0][0] + 100
            else:
                x = point1[0][0] - 100
            y = slope * (x - point2[0][0]) + point1[0][1]
            return (int(x), int(y))
        cv2.line(extend_mask, (biggest_contour[0][0][0], biggest_contour[0][0][min(len(biggest_contour[0][0]) - 1 ,2)]), find_new_point(biggest_contour[0], biggest_contour[min(len(biggest_contour) - 1 ,10)], True), (255, 255, 255), 1)
        cv2.line(extend_mask, (biggest_contour[-1][0][0], biggest_contour[-1][0][min(len(biggest_contour[-1][0]) - 1 ,2)]), find_new_point(biggest_contour[-1], biggest_contour[max(len(biggest_contour) -10, -10)], False), (255, 255, 255), 1)
        extend_mask = cv2.bitwise_and(extend_mask, extend_mask, mask = mask)
        extend_contours, _ = cv2.findContours(extend_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(skel,  extend_contours, -1, (255, 0, 255), thickness=cv2.FILLED)
        skel_contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        biggest_contour = max(skel_contours, key = cv2.contourArea)
        backbone = biggest_contour[:int(len(biggest_contour)/2)]
        length = cv2.arcLength(backbone, False)
        
        approx_mask = np.zeros(mask.shape)
        pts = np.array([backbone[0][0], backbone[-1][0]],np.int32)
        cv2.fillPoly(approx_mask, [pts], 255)
        approx_contours, _ = cv2.findContours(approx_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        x_mid=[x[0][0] for x in backbone]
        x_pred=[x[0][0] for x in approx_contours[0]]
       
        err_x=[np.abs(x1-x2) for x1,x2 in zip(x_pred,x_mid)]
        curvature = sum(err_x)/len(err_x)
        ranges = {}
        distance = 0
        interval = length / 10
        num = 0
        
        for ind, i in enumerate(backbone):
            if  ind != 0:
                distance += math.hypot(i[0][0] - backbone[ind-1][0][0], i[0][1] - backbone[ind-1][0][1])
                if distance > interval * num:
                    num += 1 
                    if ind > 10 and (ind + 10) < len(backbone):
                        x1, y1 = backbone[ind-10][0] * 200
                        x2, y2 = backbone[ind+10][0] * 200
                        lines_mask = np.zeros(mask.shape)
                        cv2.line(lines_mask, (i[0][0] +  y2 -  y1, i[0][1] - x2 + x1), (i[0][0] -  y2 +  y1,i[0][1] +x2 -x1), (255, 0, 255), 1)
                        lines_mask = cv2.bitwise_and(lines_mask, lines_mask, mask = mask)
                        line_contours, _ = cv2.findContours(lines_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        line_contours = line_contours[0][:int(len(line_contours[0])/2)]
                        ranges[str((num-1) * 10 )+ '%' ] = cv2.arcLength(line_contours, False)
                        if show:
                            cv2.drawContours(image,  line_contours, -1, (255, 0, 255), thickness=cv2.FILLED)
                        
        width = ranges['50%']

        if show:
            cv2.drawContours(image,  backbone, -1, (255, 0, 255), thickness=cv2.FILLED)
            cv2.drawContours(image,  approx_contours, -1, (255, 255, 255), 1)
    else:
        center = approx_rect[0]
        top_bottom = np.array([x for x in mask_contours[0] if int(x[0][0]) == int(center[0])], dtype=np.int32)
        top = min([x[0][1] for x in top_bottom])
        bottom = max([x[0][1] for x in top_bottom])
        left_right =  np.array([x for x in mask_contours[0] if int(x[0][1]) == int(center[1])], dtype=np.int32)
        right = max([x[0][0] for x in left_right ])
        left = min([x[0][0] for x in left_right ])
        length = max(bottom - top, right - left)
        width = min(bottom - top, right - left)
        ranges = {}
        length_interval_distance = int(length / 10)
        for i in range(9):
            points =[x[0] for x in mask_contours[0] if int(x[0][1]) == int(top + length_interval_distance * (i + 1))]
            distance = max(points[0][0], points[1][0]) - min(points[0][0], points[1][0])
            ranges[str((i + 1) * 10 )+ '%' ] = distance
        width = ranges['50%']
        if show:
            cv2.drawContours(image, np.array([[ top_bottom[0][0], top_bottom[1][0]]]), -1, (0, 0, 255
                    ), 3)
            cv2.drawContours(image, np.array([[ left_right[0][0],  left_right[1][0]]]), -1, (0, 255, 0), 3)
    if show:
        cv2.drawContours(image, mask_contours, -1, (255, 0, 255
            ), 3) 
        cv2.imshow("Skeleton",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    return length, width, ranges, curvature

def get_size(mask):
    m = cv2.convertScaleAbs(mask)
    contours, _ = cv2.findContours(m.copy(), 1, 1)
    if len(contours) > 0:
        rect = cv2.minAreaRect(contours[0])
        area = cv2.contourArea(contours[0])
        (x, y), (w, h), a = rect
        width = min(w, h)
        length = max(w, h)
        return (width, length, a, area)
    else:
        return (0, 0, 0, 0)
def round_number(num):
    return str(round(num, 3))

def lambda_handler(event, context):
    aws = AWS()
    aws.set()
    meta = aws.get_meta(event['Name'], event['ProjName'])

    annotations = aws.get_annotation(event['Name'], event['ProjName'])['annotations']
    if '608__4chan' in meta['keys']:
        aws.get_file("/tmp/image.jpg", meta['keys']['608__4chan'])
    elif 'calibrated' in meta and meta['calibrated'] == True:
        aws.get_file("/tmp/image.jpg", meta['keys']['608_calibration'])
    else:
        aws.get_file("/tmp/image.jpg", meta['keys']['608'])
    image = cv2.imread("/tmp/image.jpg", cv2.IMREAD_UNCHANGED)
    ratio = float((meta['original_height'] / image.shape[0]))

    if 'px_cm' not in meta:
        refrence_tags = {}
        project = list(filter(lambda p: p["Name"] == meta["project"],  aws.get_project(meta["taxonomy"])['projects']))[0]
        for tag in project['tags']:
            for subTag in tag["subTags"]:
                if subTag["type"] == "constant" and (subTag["Name"] == "Length" or subTag["Name"] == "Width"):
                    if tag["Name"] not in refrence_tags:
                        refrence_tags[tag["Name"]] = {} 
                    unit = ""
                    if "unit" in subTag["constant"]:
                        unit = subTag["constant"]["unit"]
                       
                    refrence_tags[tag["Name"]][subTag["Name"]] = {'value': subTag[
                        "constant"]["value"], 'unit': unit}
        refrence = {'width': 1, 'length': 1, 'w_unit': "px", 'l_unit': "px", 'angle': 0}
        for annotation in annotations:
            if annotation["data"]['tagName'] in refrence_tags:
                points = np.array(list(map(lambda x: [int(x["x"] * image.shape[1] / 100), int(x["y"] * image.shape[0] / 100)], annotation["geometry"][
                                    "points"])), dtype=np.int32)
                mask = np.zeros([int(image.shape[0]), int(image.shape[1]) ],dtype=np.uint8)
                cv2.fillPoly(mask, pts=[points], color=(255))
                (ref_width, ref_length, ref_Angle, ref_area) = get_size(mask)
                refrence['angle'] = ref_Angle
                if "Width" in refrence_tags[annotation["data"]['tagName']]:
                    refrence['width'] = float(refrence_tags[annotation["data"]['tagName']]["Width"]['value']) / ref_width / ratio
                    refrence['w_unit'] = refrence_tags[annotation["data"]['tagName']]["Width"]['unit']
                if "Length" in refrence_tags[annotation["data"]['tagName']]:
                    refrence['length'] = float(refrence_tags[annotation["data"]['tagName']]["Length"]['value']) / ref_length / ratio
                    refrence['l_unit'] = refrence_tags[annotation["data"]['tagName']]["Length"]['unit']
    else:
        refrence = {'width': float(meta['px_cm']), 'length': float(meta['px_cm']), 'w_unit': "cm", 'l_unit': "cm", 'angle': 0}
    
    message = []
    height = 0
    meta = aws.get_meta(event['Name'], event['ProjName'])
    for annotation in annotations:
        points = np.array(list(map(lambda x: [float(x["x"]) * image.shape[1] / 100, float(x["y"]) * image.shape[0] / 100], annotation["geometry"]["points"])), dtype=np.int32)
        if len(points) > 1:
            mask = np.zeros([int(image.shape[0]), int(image.shape[1]) ],dtype=np.uint8)
            cv2.fillPoly(mask, pts=[points], color=(255))
            (width, length, angle, area) = get_size(mask)
            if image.shape[2] > 3:
                alpha_channel = image[:,:,3]
              
                img_mask =alpha_channel[np.where(mask == 255)]
                height = np.max(img_mask) / 10 - np.min(alpha_channel) / 10

            cropped = crop_mask(image, mask).astype('uint8')
            # Assuming the mask and image are original size images crop the relevant fruit part
            x, y, w, h = cv2.boundingRect(mask)
            croped_Img = image[y:y+h, x:x+w]
            croped_mask=mask[y:y+h, x:x+w]
            
            # Added for Pepper phenotyping defined in utils. Works on mask and image patch Assuming length is across x and width is across y will produce wring results
            # if  the conditions are not met, Some outputs are not used since they are alöready included can be used for debugging
            (
            area_rec,
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
            square_aspect_ratio,
            vis_imgs,
            labels
            ) = phenotype_measurement_pepper(croped_mask, croped_Img,ppx=refrence['length'],ppy=refrence['width'])
            
          
            try:
                mid_length, mid_width, ranges, curvature = skeletonize(mask, image) 
            except Exception as e:
                print(e)
                curvature = 0
                mid_length = length
                mid_width = width
                ranges = False

            bgr =list( cv2.mean(image, mask=mask)[:3])
            sRgb = sRGBColor(bgr[2], bgr[1], bgr[0], is_upscaled=True)
            lab = convert_color(sRgb, LabColor)
            lch = convert_color(sRgb, LCHabColor)
            if angle - refrence['angle'] > 45:
                if ranges != False:
                    for r in ranges:
                        ranges[r] *= refrence['length']
                width *= refrence['length']
                length *= refrence['width']
            else:
                width *= refrence['width']
                length *= refrence['length']
                if ranges != False:
                    for r in ranges:
                        ranges[r] *= refrence['width']
          
            res = {
                'id': str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName'],
                'size': {
                    'Width': round_number(min(width, length) * ratio) + " " + refrence['w_unit'], 
                    'Length': round_number(max(width,length)* ratio) + " " + refrence['l_unit'], 
                    'Angle': round_number(angle), 
                    'Area': round_number(area * ratio * ratio * refrence['width']* refrence['length']),
                    "Mid_Length": round_number(max(mid_width, mid_length)* ratio * refrence['length']) + " " + refrence['l_unit'],
                    "Mid_Width": round_number(min(mid_width, mid_length) * ratio* refrence['width'] )+ " " + refrence['w_unit'],
                    "Curvature": round_number(curvature),
                    # write results from above addtion of shape and size
                    "Mid_Height_Width": round_number(mid_height_width),
                    "Max_Width": round_number(max_width),
                    "Mid_Width_Height": round_number(mid_width_height),
                    "Max_Height": round_number(max_height),
                    "Curved_Height": round_number(c_length),
                },
                'color': {
                    'RGB': str(bgr[::-1]), 'LAB': str([lab.lab_l, lab.lab_a, lab.lab_b]),  'LCH': str([lch.lch_l, lch.lch_c, lch.lch_h])
                },
                # Write results from pepper phenotypes
                'shape':
                    {
                    "Maxheight_to_Maxwidth": maxheight_to_maxwidth,
                    "Mid_Width_Height_to_Mid_Height_Width": mid_width_height_to_mid_height_width,
                    "C_Height_to_Mid_Height_Width": c_length_to_mid_height_width,
                    "Proximal_Block": proximal_block,
                    "Distal_Block": distal_block,
                    "Triangle": triangle,
                    "Ellipse_Err": ellipse_err,
                    "Square_Aspect_Ratio": square_aspect_ratio,
                        
                    }
            }
            
            if ranges != False:
                for r in ranges:
                    ranges[r] = round_number(ranges[r] * ratio) + " " + refrence['w_unit']
                res['size']['Ranges'] = str(ranges)
            message.append(res)
            
            if 'subTags' not in meta:
                 meta["subTags"] = {}

            if str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName'] not in meta["subTags"]:
                meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']] ={}
            if height != 0:
                res["size"]["Height"] = round_number(height) + " cm"
                meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Height"] = res["size"]["Height"] 
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Mid_Width"] = res["size"]["Mid_Width"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Mid_Length"] = res["size"]["Mid_Length"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Length"] = res["size"]["Length"]
            if ranges != False:
                meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Ranges"] = res["size"]["Ranges"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Width"] = res["size"]["Width"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Area"] = res["size"]["Area"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Curvature"] = res["size"]["Curvature"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Angle"] = res["size"]["Angle"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["RGB"] = res["color"]["RGB"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["LAB"] = str(res["color"]["LAB"])
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["LCH"] = str(res["color"]["LCH"])
            
            # Write pepper phenotyes  size to metadata
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Mid_Height_Width"] = res["size"]["Mid_Height_Width"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Max_Width_Length"] = res["size"]["Mid_Width_Height"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Max_Length"] = res["size"]["Max_Height"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Curved_Height"] = res["size"]["Curved_Height"]
            # Write pepper phenotyes  shape to metadata              
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["MaxLength_to_MaxWidth"] = res["shape"]["Maxheight_to_Maxwidth"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Mid_Width_Length_to_Mid_Length_Width"] = res["shape"]["Mid_Width_Height_to_Mid_Height_Width"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["C_Length_to_Mid_Length_Width"] = res["shape"]["C_Height_to_Mid_Height_Width"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Proximal_Block"] = res["shape"]["Proximal_Block"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Distal_Block"] = res["shape"]["Distal_Block"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Triangle"] = res["shape"]["Triangle"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Ellipse_Error"] = res["shape"]["Ellipse_Err"]
            meta["subTags"][str(annotation["data"]['id']) +"__"+ annotation["data"]['tagName']]["Square_Aspect_Ratio"] = res["shape"]["Square_Aspect_Ratio"]
            
    
    if len(annotation) > 0:
        meta["annotations"] = True
        if "run_id" in meta['meta_data'] and 'weight_uom' in meta['meta_data']:
            pictures = aws.get_pictures(event['ProjName'], meta['meta_data']['run_id'])
            total_area = 0
            for pic in pictures:
                if pic["Name"] != event["Name"]:
                    if "subTags" in pic:
                        for s, i in pic['subTags'].items():
                            if 'Area' in i:
                                total_area += float(i["Area"])
            for s, i in meta["subTags"].items():
                if "Area" in i:
                    total_area += float(i["Area"])
            conv = float(meta["meta_data"]["weight"]) / total_area
            for s, i in meta["subTags"].items():
                if "Area" in i  and 'Weight' in meta['subTags'][s]:
                    meta['subTags'][s]["Weight"] = str(round_number(float(i["Area"]) * conv)) + " "+ meta['meta_data']['weight_uom']
                for subTag in message:
                    if subTag["id"] == s and 'Weight' in meta['subTags'][s]:
                        subTag['size']["Weight"] = meta['subTags'][s]["Weight"]
    else:
        meta["annotations"] = False
    aws.save_meta(meta)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(message)
    }



# lambda_handler({
#     "Name":"123_21_rgb_009.jpg",
#     "ProjName":"Pepper/grab cut test"
# }, False)
