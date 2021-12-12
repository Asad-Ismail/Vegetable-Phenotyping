import cv2
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from inference import detectroninference
import openpyxl
import pickle


def resize_image(img,scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def find_orientation(cucumber):
    """horizontal or vertical cucumbers true if vertical else false"""
    xs=cucumber[:,0]
    ys=cucumber[:,1]
    xmin=np.min(xs)
    xmax=np.max(xs)
    ymin=np.min(ys)
    ymax=np.max(ys)
    return ymax-ymin> xmax-xmin


def left_right_bound(img):
    cuc=cv2.findNonZero(img).squeeze()
    line_points=[]
    r_data={}
    vertical=find_orientation(cuc)
    if vertical:
        xs=cuc[:,0]
        ys=cuc[:,1]
    else:
        xs=cuc[:,1]
        ys=cuc[:,0]
    # loop through all the unique ys
    r_data["orient"]=vertical
    u_ys=sorted(set(ys))
    for y in u_ys:
        tmp=[]
        m_p=np.argwhere(ys==y)
        x_p=xs[m_p]
        if vertical:
            tmp.append((np.min(x_p),y))
            tmp.append((np.max(x_p),y))
        else:
            tmp.append((y,np.min(x_p)))
            tmp.append((y,np.max(x_p)))
        line_points.append(tmp)
    r_data["data"]=line_points
    return r_data



def draw_bounds(img,data,color=(255,0,0)):
    for point in data:
        cv2.circle(img,point,2,color)




def colsest_to_cb(cuc_boxes,masks,color_box,org_img_cuc,debug=False):
    min_so_far=1e12
    closest_index=None
    x1c,x2c,y1c,y2c=color_box
    xc,yc=0,0
    h,w,_=org_img_cuc.shape
    for index,box in enumerate(cuc_boxes):
       for i in range(0,4,2):
           #if w>h:
           dis=np.abs(x1c-box[i])
           #else:
           #dis=np.abs(y1c-box[i+1])
           if (dis<min_so_far):
               min_so_far=dis
               closest_index=index
               xc=box[i]
               yc=box[i+1]
    if debug:
        x1, y1, x2, y2=cuc_boxes[closest_index]
        closest_mask=masks[closest_index].astype(np.uint8)
        closest_mask=cv2.cvtColor(closest_mask,cv2.COLOR_GRAY2BGR)
        vis_img=cv2.rectangle(org_img_cuc,(x1,y1),(x2,y2),(0,255,0),3)
        vis_img=cv2.line(org_img_cuc,(x1c,y1c),(x1c,y2c),(0,255,0),3)
        alpha=0.5
        beta = (1.0 - alpha)
        vis_img = cv2.addWeighted(vis_img, alpha, closest_mask, beta, 0.0)
        cv2.imshow("Closest bbox",resize_image(vis_img,30))
        #cv2.imwrite("TestImage.png",vis_img)
        #cv2.waitKey(0)
    return closest_index


def dims_per_pixels(color_box):
    real_dims=(6,9)
    x1c,x2c,y1c,y2c=color_box
    length_per_pixel=float(real_dims[0])/(np.abs(y2c-y1c))
    width_per_pixel=float(real_dims[1])/(np.abs(x2c-x1c))
    return length_per_pixel,width_per_pixel


def get_area(mask,box,l_pp,w_pp):
    data=left_right_bound(mask)
    lower_bound=[point[0] for point in data["data"]]
    upper_bound=[point[1] for point in data["data"]]

    assert len(lower_bound)==len(upper_bound)
    actual_area=0
    for i in range(0,len(lower_bound)-1):
        x1,y1=lower_bound[i]
        x2,y2=upper_bound[i+1]
        #cv2.rectangle(bound_img,(x1,y1),(x2,y2),(0,255,0))
        width=np.abs(x2-x1)*w_pp
        length=np.abs(y2-y1)*l_pp
        actual_area+=width*length
     
    x1, y1, x2, y2=box
    #cv2.rectangle(bound_img,(x1,y1),(x2,y2),(0,255,0))
    rough_area=(np.abs(x2-x1)*w_pp)*(np.abs(y2-y1)*l_pp)
    assert rough_area>=actual_area
    return actual_area
    #print(f"Rough Area is {rough_area} actual area {actual_area}")
    #cv2.imshow("Image",bound_img)
    #cv2.waitKey(0)


def get_length(mask,l_pp):
    data=left_right_bound(mask)
    lower_bound_y=[point[0][1] for point in data["data"]]
    print(lower_bound_y[0])
    print(lower_bound_y[-1])
    length=(lower_bound_y[-1]-lower_bound_y[0])*l_pp
    return length



def evaluate(cuc_boxes,masks,color_box,org_img_cuc,gts,filename,debug=False):
    gt_l,gt_weight=gts
    closest_index=colsest_to_cb(cuc_boxes,masks,color_box,org_img_cuc,debug)
    l_pp,w_pp=dims_per_pixels(color_box)
    # Get cucumber closest to the color board
    x1, y1, x2, y2=cuc_boxes[closest_index]
    #pred_length=np.abs(y2-y1)*l_pp
    pred_length=get_length(masks[closest_index],l_pp)
    error=pred_length-gt_l
    print(f"Predicted Length of {filename} cucumber is {pred_length}")
    print(f"Error of length {error}")
    total_area=0
    for i in range(len(masks)):
        total_area+=get_area(masks[i],cuc_boxes[i],l_pp,w_pp)

    return pred_length,gt_l,np.abs(error),total_area
    
dictnames={"041621":0,"030221":0}  

def get_gt_values(sheet,image_name):
    #image_value=image_name.split(".")[0]
    image_value=image_name.split("_")[0]
    dt=image_name.split("_")[-1].split(".")[0][:6]
    #colNames = sheet['F']
    all_matches=[]
    colNames = sheet['G']
    if dt=="041621":
        gt_lengths = sheet['S']
        gt_weights = sheet['W']
        dictnames["041621"]+=1
    elif dt=="030221" or dt=="030121":
    #elif dt=="030221":
    #elif dt=="030221":
        gt_lengths = sheet['R']
        gt_weights = sheet['V']
        dictnames["030221"]+=1
    else:
        if dt not in dictnames:
            dictnames[dt]=1
        else:
            dictnames[dt]+=1
        print(f"File {image_name} not found xls")
        return None
    for i,colname in  enumerate(colNames):
        if colname.value is None:
            continue
        value_name=str(colname.value)
        if value_name==image_value:
            #shutil.copyfile("/media/asad/ADAS_CV/cuc/axel_april_evaluate/"+image_name,"/media/asad/ADAS_CV/cuc/axel_april_evaluate_gt/"+image_name)
            all_matches.append(i)
    if len(all_matches)>0:
        gt_l=float(gt_lengths[all_matches[0]].value)
        gt_w=float(gt_weights[all_matches[0]].value)/1000
        return gt_l,gt_w
    if len(all_matches)>2:
        print("More than 1 Matches Do Something")
    else:
        return None      

def get_parameters(model):
    param=sum([p.numel() for p in model.parameters() if p.requires_grad]) 
    print(param*4*1e-6)   

images_length_error={}

imgs_error=['P00000000879558649071959_1116_  _032621141143.jpg', 'P00000000879558637815811_1079_  _032621133201.jpg', 'P00000000879558649071949_1106_  _022421102307.jpg', 'P00000000879558645484175_1051_  _032621102435.jpg', 'P00000000879558649071976_1133_  _032621143606.jpg', 'P00000000879558645484145_1021_  _032621094022.jpg', 'P00000000879558649071962_1119_  _032621141636.jpg', 'P00000000879558645484169_1045_  _032621101606.jpg', 'P00000000879558645484153_1029_  _032621095210.jpg', 'P00000000879558645484170_1046_  _032621101728.jpg', 'P00000000879558637815782_1064_  _032621104345.jpg', 'P00000000879558645484177_1053_  _032621102659.jpg', 'P00000000879558649071952_1109_  _032621140135.jpg', 'P00000000879558645484150_1026_  _032621094817.jpg', 'P00000000879558637815870_1103_  _032621135445.jpg', 'P00000000879558637815832_1090_  _022421093842.jpg', 'P00000000879558645484164_1040_  _032621100912.jpg', 'P00000000879558637815852_1100_  _032621135125.jpg', 'P00000000879558637815870_1103_  _022421095158.jpg', 'P00000000879558649071970_1127_  _032621142744.jpg', 'P00000000879558649071955_1112_  _022421103346.jpg', 'P00000000879558649071954_1111_  _022421103242.jpg', 'P00000000879558645484158_1034_  _032621095934.jpg', 'P00000000879558645484161_1037_  _032621100537.jpg']
#imgs_error=['P00000000879558850448882_V26_269_030221101709.jpg', 'P00000000879558850449160_V26_547_041621094424.jpg', 'P00000000879558850448659_V26_46_030121171505.jpg', 'P00000000879558850449165_V26_552_041621094657.jpg', 'P00000000879558850449153_V26_540_041621094156.jpg', 'P00000000879558850449067_V26_454_041621090210.jpg', 'P00000000879558850449127_V26_514_030221123402.jpg', 'P00000000879558850448899_V26_286_030221101954.jpg', 'P00000000879558850449059_V26_446_030221111013.jpg', 'P00000000879558850448926_V26_313_030221102825.jpg', 'P00000000879558850448939_V26_326_030221103306.jpg', 'P00000000879558850448661_V26_48_030121171604.jpg', 'P00000000879558850448643_V26_30_030121162027.jpg', 'P00000000879558850449112_V26_499_041621092414.jpg', 'P00000000879558850449154_V26_541_030221124319.jpg', 'P00000000879558850449076_V26_463_041621090612.jpg', 'P00000000879558850449039_V26_426_030221110335.jpg', 'P00000000879558850449073_V26_460_041621090443.jpg', 'P00000000879558850449157_V26_544_041621094309.jpg', 'P00000000879558850448740_V26_127_030121174644.jpg', 'P00000000879558850448745_V26_132_030121175000.jpg', 'P00000000879558850449139_V26_526_041621093603.jpg', 'P00000000879558850448925_V26_312_030221102750.jpg', 'P00000000879558850449149_V26_536_041621093953.jpg', 'P00000000879558850449109_V26_496_030221112636.jpg', 'P00000000879558850448627_V26_14_030121160651.jpg', 'P00000000879558850449148_V26_535_041621093911.jpg', 'P00000000879558850448650_V26_37_030121170824.jpg', 'P00000000879558850449163_V26_550_041621094505.jpg', 'P00000000879558850449111_V26_498_041621092204.jpg', 'P00000000879558850449164_V26_551_041621094541.jpg', 'P00000000879558850449113_V26_500_041621092453.jpg', 'P00000000879558850449141_V26_528_030221123823.jpg', 'P00000000879558850449093_V26_480_041621091410.jpg', 'P00000000879558850449130_V26_517_041621093254.jpg', 'P00000000879558850448622_V26_9_030121160041.jpg', 'P00000000879558850449092_V26_479_041621091337.jpg', 'P00000000879558850449109_V26_496_041621092039.jpg']
if __name__=="__main__":
    train=True
    cuc=detectroninference("/media/asad/ADAS_CV/cuc/output_axelMay/model_final.pth")
    print(get_parameters(cuc.predictor.model))
    print(cuc.predictor.model.parameters())
    book = openpyxl.load_workbook('/media/asad/8800F79D00F79104/v26/V26_221_DH0-20210416113457.xlsx')
    sheet=book.active
    #img_path="/home/asad/axel/complete"
    img_path="/media/asad/8800F79D00F79104/v26/v26_rotated"
    times=[]
    preds_l=[]
    gts_l=[]
    errors=[]
    areas=[]
    gt_weights=[]
    total_fruits=0
    total_images=0
    skipped=0
    input_images=os.listdir(img_path)
    for file_index,filename in enumerate(input_images):
        try:
            gt_l,gt_weight=get_gt_values(sheet,filename)
        except:
            continue
        if gt_l is None or gt_weight is None:
            exit("Error GT not found!!")
        f_path=os.path.join(img_path,filename)
        org_img=cv2.imread(f_path)
        #original image with 60% scaling
        #org_img=resize_image(org_img,30)
        org_img_cuc=org_img.copy()
        detected_cucumber,all_masks,all_patches,boxes,classes,scores=cuc.pred(org_img)
        if filename in imgs_error:
            debug=True
        else:
            debug=False
        pred,gt_l,error,area=evaluate(boxes,all_masks,cb_box,org_img_cuc,(gt_l,gt_weight),filename=filename,debug=debug)
        if (abs(error)>7):
            skipped+=1
            continue
            #print("something is wrong!!")
        font= cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (detected_cucumber.shape[0]//2,detected_cucumber.shape[1]//11)
        fontScale = 1
        fontColor= (0,0,255)
        lineType = 3
        detected_cucumber=cv2.putText(detected_cucumber.copy(),"Pred: {:.2f}".format(pred), topLeftCornerOfText,font, fontScale,fontColor,lineType)
        fontColor= (0,0,255)
        topLeftCornerOfText = ((detected_cucumber.shape[0]//2)+300,detected_cucumber.shape[1]//11)
        detected_cucumber=cv2.putText(detected_cucumber.copy(),"GT: {:.2f}".format(gt_l), topLeftCornerOfText,font, fontScale,fontColor,lineType)

        if debug:
            cv2.imshow("Detected Cucumber and color",resize_image(detected_cucumber,50))
            #cv2.imwrite(os.path.join("/media/asad/ADAS_CV/cuc/debug_tests/",filename+".png"),detected_cucumber)
            cv2.waitKey(0)
            #exit(-1)
        else:
            total_fruits+=len(boxes)
            total_images+=1
            areas.append(area)
            gt_weights.append(gt_weight)
            preds_l.append(pred)
            gts_l.append(gt_l)
            errors.append(error)
        print(f"Processed {file_index+1} files")
        images_length_error[filename]=abs(error)

    print(f"Total images {total_images}")
    print(f"Total fruits {total_fruits}")
    print(f"Skipped {skipped} files")
    print(dictnames)
    l_err_mean=np.mean(errors)
    l_err_median= np.median(errors)
    l_err_std= np.std(errors)
    l_err_perc=np.mean(np.abs(np.array(gts_l)-np.array(preds_l))/np.array(gts_l))
    print(f"Mean error Length: {l_err_mean} Median error Length: {l_err_median} Std error Length: {l_err_std} Percentage error Length: {l_err_perc*100}")
    print(f"*"*40)
    if train:
        model_weight = np.polyfit(areas, gt_weights, 1)
        print(f"Model is {model_weight}")
        pred_model_weight=np.poly1d(model_weight)
        w_preds=pred_model_weight(areas)
    else:
        pred_model_weight=np.poly1d([0.0029999,-0.078212])
        w_preds=pred_model_weight(areas)

    
    w_m_err_mean=np.mean(np.abs(gt_weights-w_preds))
    w_m_err_perc=np.mean(np.abs(gt_weights-w_preds)/np.array(gt_weights))
    w_m_err_median=np.median(np.abs(gt_weights-w_preds))
    w_m_err_std=np.std(np.abs(gt_weights-w_preds))
    print(f"Mean error Weight grams: {w_m_err_mean} Median error Weight grams: {w_m_err_median} Std error Weight grams: {w_m_err_std} Percentage mean: {w_m_err_perc*100}")
    plt.scatter(gt_weights,areas)
    plt.xlabel("Ground truth weights")
    plt.ylabel("Cucumber area")
    figures,axes=plt.subplots(1,3,figsize=(20,20))
    f_axes=axes.ravel()
    f_axes[0].scatter(range(len(errors)),errors)
    f_axes[0].set_title("Length error")
    f_axes[1].scatter(gt_weights,w_preds)
    f_axes[1].set_title("Weight error")
    f_axes[2].scatter(gt_weights,areas)
    f_axes[2].set_title("Weight VS Area")
    #plt.scatter(gts,preds)
    plt.show()

