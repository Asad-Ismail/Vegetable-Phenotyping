import cv2
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from inference import detectroninference

def intersec_over_union(img1,img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_points=cv2.findNonZero(img1).squeeze()
    img1_points=set(img1_points.flatten())
    img2_points=cv2.findNonZero(img2).squeeze()
    img2_points=set(img2_points.flatten())
    inter=img1_points.intersection(img2_points)
    union=img1_points.union(img2_points)
    return len(inter)/len(union)


def distance(point_one, point_two):
    return ((point_one[0] - point_two[0]) ** 2 +
            (point_one[1] - point_two[1]) ** 2) ** 0.5

def total_distance(points):
    return sum(distance(p1, p2) for p1, p2 in zip(points, points[1:]))


def find_orientation(cucumber):
    """horizontal or vertical cucumbers true if vertical else false"""
    xs=cucumber[:,0]
    ys=cucumber[:,1]
    xmin=np.min(xs)
    xmax=np.max(xs)
    ymin=np.min(ys)
    ymax=np.max(ys)
    return ymax-ymin> xmax-xmin


def shape_uniformity(img1):
    cuc=cv2.findNonZero(img1).squeeze()
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



def curvature_plane(data):
    x=np.array([*data])[:,0]
    # Reverse y for curvature becuase of differnce of image and plot lib
    y=-1*np.array([*data])[:,1]
    x_p=np.diff(x)
    x_pp=np.diff(x_p)
    x_pp=np.append(x_pp,0)
    y_p=np.diff(y)
    y_pp=np.diff(y_p)
    y_pp=np.append(y_pp,0)
    curv_num=(x_p*y_pp)-(y_p*x_pp)
    curv_den=((x_p**2)+(y_p**2))**1.5
    curv=curv_num/curv_den
    #remove curvature if greater than 2
    curv=[x for x in curv if abs(x)<=1.2]
    return curv


def sd_calculate(r_data):
    orient=r_data["orient"]
    data=r_data["data"]
    if orient:
        lower_bound=[point[0][0] for point in data]
        upper_bound=[point[1][0] for point in data]
        upper_bound=[point[1][1] for point in data]
    lower_var=np.var(np.diff(lower_bound))
    upper_var=np.var(np.diff(upper_bound))
    print(f"Lower bound std {lower_var}")
    print(f"Upper bound std {upper_var}")

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def color_palette(color):
    if color=="green":
        all_colors=[]
        g_ll=40
        g_hl=90
        for i in range(g_ll,g_hl,5):
            color=(i,100,100)
            all_colors.append(color)
        img=np.zeros((len(all_colors)*100,400,3),dtype=np.uint8)
        for i,c in enumerate(all_colors):
            #print(img[100*i:100*(i+1),:].shape)
            min_r=i*100
            max_r=100*(i+1)
            img[min_r:max_r,:]=c
            img[max_r-1,:]=(255,255,255)
            cv2.putText(img, str(c[0]), (10,min_r+30), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,0,255), 2, cv2.LINE_AA) 
        bgrimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return bgrimg
        

def calculate_color_hist(patch):
    hsvimg = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    roi=np.nonzero(patch)
    color_channels_h=hsvimg[roi[0],roi[1]][:,0]
    return color_channels_h


def calculate_dimensions(lower_bound,upper_bound,distances,drawing_image=None):
    #Only cacluatng for lower bound not sure why it should be different for upper
    lower_bound=np.array([*lower_bound])
    xs_l=lower_bound[:,0]
    ys_l=lower_bound[:,1]
    if(np.max(xs_l)-np.min(xs_l)< np.max(ys_l)-np.min(ys_l)):
        #Veritcal cuc
        if distances is not None:
            length=(np.max(ys_l)-np.min(ys_l))*distances[1]
        else:
            length=np.max(ys_l)-np.min(ys_l)
            if drawing_image is not None:
                drawing_image=cv2.line(drawing_image, (np.max(xs_l),np.min(ys_l)), (np.max(xs_l),np.max(ys_l)), (255,255,0), 2) 
        
    else:
        #Horzontal cucumber
        if distances is not None:
            length=(np.max(xs_l)-np.min(xs_l))*distances[0]
        else: 
            length=np.max(xs_l)-np.min(xs_l)
        if drawing_image is not None:
            drawing_image=cv2.line(drawing_image, (np.min(xs_l),np.max(ys_l)), (np.max(xs_l),np.max(ys_l)), (255,255,0), 2) 
    return length

def cal_dims(lower_bound,upper_bound,distances,drawing_image,draw_width=True):
    ##Draw lengths for the color board
    lengths=[]
    lower_bound=np.array([*lower_bound])
    upper_bound=np.array([*upper_bound])
    xs_l=lower_bound[:,0]
    ys_l=lower_bound[:,1]
    xs_u=upper_bound[:,0]
    ys_u=upper_bound[:,1]
    if(np.max(xs_l)-np.min(xs_l)< np.max(ys_l)-np.min(ys_l)):
        #Veritcal cuc
        if distances is not None:
            widths=np.abs(xs_l-xs_u)*distances[0]
        else:
            widths=np.abs(xs_l-xs_u)
    else:
        #Horizontal cucumber
        if distances is not None:
            widths=np.abs(ys_l-ys_u)*distances[1]
        else:
            widths=np.abs(ys_l-ys_u)
    if draw_width:
        for i,(start,end) in enumerate(zip(lower_bound,upper_bound)):
            lengths.append(((start[0]+end[0])/2.0,(start[1]+end[1])/2.0))
            cv2.circle(drawing_image,(int((start[0]+end[0])/2.0),int((start[1]+end[1])/2.0)),1,(255,255,0))
            if (i%10)==0:
                drawing_image=cv2.line(drawing_image, tuple(start), tuple(end), (0,255,0), 1) 
    m_w=np.mean(widths)
    s_w=np.std(widths)
    length=total_distance(lengths)
    return widths,m_w,s_w,length


if __name__=="__main__":
    #cuc_cls=cuc_class(WEIGHTS_DIR="/home/asad/od/cuc/mrcnn/models/resnet38")
    #cuc=detectroninference("/home/asad/dev/detectron2/output_resnet101/model_final.pth")
    #cuc=detectroninference("/home/asad/dev/detectron2/output_resnet101_scalecam/model_0006999.pth")
    cuc=detectroninference("/home/asad/dev/detectron2/output_resnet101_scalecam/model_0006999.pth")
    #cuc_cls=cuc_class()
    #img_path="/home/asad/annotated_700_cucumber/t_cu"
    #img_path="/media/asad/ADAS_CV/scalecam/corrected"
    img_path="/media/asad/8800F79D00F79104/aws_data/aws_color_correct"

    #img_path="/home/asad/rand_imgs"
    times=[]
    for file_index,filename in enumerate(os.listdir(img_path)):
        f_path=os.path.join(img_path,filename)
        org_img=cv2.imread(f_path)
        correct_img=org_img.copy()
        distances=None

        detected_cucumber,all_masks,all_patches=cuc.pred(org_img)
        resize_shape=correct_img.shape[:-1]
        resize_shape=tuple((int(x*0.5) for x in resize_shape))
        # reverse x and y for size tuple
        resize_shape= resize_shape[::-1] 
        image_shape=correct_img.shape[:-1][::-1]
        resize_ratio=(image_shape[0]/resize_shape[0],image_shape[1]/resize_shape[1])
        if distances is not None:
            distances=[x*y for x,y in zip(distances,resize_ratio) ]
        #distances=None
        print(distances)
        for ind,mask in enumerate(all_masks):
            #Mask
            img=cv2.resize(correct_img.astype(np.uint8),resize_shape)
            mask=cv2.resize(mask.astype(np.uint8),resize_shape)
            cuc_patch=cv2.resize(all_patches[ind].astype(np.uint8),resize_shape)
            imgray=mask
            data=shape_uniformity(imgray)
            lower_bound=[point[0] for point in data["data"]]
            upper_bound=[point[1] for point in data["data"]]
            c_l=curvature_plane(lower_bound)
            c_u=curvature_plane(upper_bound)
            #smoothing for curvature
            smooth=60
            c_l=moving_average(c_l,smooth)
            c_u=moving_average(c_u,smooth)
            #color palette test
            #palette=color_palette("green")
            #Hue values 
            h_color=calculate_color_hist(cuc_patch)
            h_color_mean=np.mean(h_color)
            h_color_median=np.median(h_color)
            h_color_std=np.std(h_color)
            #Width of cucumbers
            bound_img=np.zeros_like(img)
            draw_bounds(bound_img,lower_bound)
            draw_bounds(bound_img,upper_bound,(0,0,255))
            wd,m_wd,sd_wd,length=cal_dims(lower_bound,upper_bound,distances,bound_img)
            width=np.mean(wd)
            ratio=length/np.mean(wd)
            # Find if curved or not
            #is_curved,probs=cuc_cls(cuc_patch[:,:,::-1])
            fig,axis=plt.subplots(4,2,figsize=(10,10))

            # Draw cuc patch and color palette
            axis[0,0].imshow(correct_img[:,:,[2,1,0]])
            axis[0,0].set_yticklabels([])
            axis[0,0].set_xticklabels([])
            axis[0,1].imshow(detected_cucumber[:,:,::-1])
            axis[0,1].set_yticklabels([])
            axis[0,1].set_xticklabels([])
            axis[1,0].imshow(cuc_patch[:,:,::-1])
            axis[1,0].set_yticklabels([])
            axis[1,0].set_xticklabels([])
            axis[1,1].imshow(bound_img[:,:,[2,1,0]])
            axis[1,1].set_yticklabels([])
            axis[1,1].set_xticklabels([])
            axis[2,0].plot(c_l,'b',label='Lower boundary')
            axis[2,0].plot(c_u,'r', label='Upper boundary')
            axis[2,0].set_ylabel("Curvature")
            axis[2,0].set_xlabel("Points")
            axis[2,0].legend(prop={'size': 2})
            #Draw color histograms
            #Bins are for green color hsv range
            bins=np.arange(20,110,10)
            weights = np.ones_like(h_color)/float(len(h_color))
            H, bins=np.histogram(h_color,bins=bins, weights=weights)
            axis[2,1].bar(bins[:-1],H,width=5,label=f'Mean: {h_color_mean:.2f}, Std: {h_color_std:.2f}')
            #plt.bar(bins[:-1],H,width=5,label=f'Mean: {h_color_mean:.2f}, Std: {h_color_std:.2f}')
            #plt.show()
            axis[2,1].set_xticks(bins)
            axis[2,1].set_xlabel("Hue")
            axis[2,1].legend(prop={'size': 4})
            # Draw widths
            axis[3,0].plot(wd,label=f"Mean: {m_wd:.2f}, Std: {sd_wd}")
            axis[3,0].set_ylabel("width")
            axis[3,0].legend(prop={'size': 4})
            #Draw dimensions
            #axis[3,1].text(0.2, 0.2, f"Curved?: {cuc_cat}",bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',verticalalignment='center')
            #axis[3,1].text(0.5, 0.2, f"P_Curv {cuc_prob[1].item():.2f}",bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',verticalalignment='center')
            axis[3,1].text(0.2, 0.5, f'Length: {length:.2f}',bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',verticalalignment='center')
            axis[3,1].text(0.52, 0.5, f'Width: {width:.2f}',bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',verticalalignment='center')
            axis[3,1].text(0.8, 0.5, f'AR: {ratio:.2f}',bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',verticalalignment='center')
            plt.tight_layout()
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry = (250, 120, 1280, 1024)
            plt.draw()
            plt.waitforbuttonpress(0) # this will wait for indefinite time
            plt.close(fig)


