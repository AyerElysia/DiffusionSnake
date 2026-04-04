
import math
#import torch
import numpy as np

from scipy.spatial.distance import pdist

class Vertebrae(object):
    def __init__(self, 
                 label=0, 
                 keypoints=np.zeros((6, 2)), 
                 score=1.0, 
                 visible=True,
                 feature=[]):
        if label>-1:
            self.label = int(label)
        elif len(feature):
            self.label = np.argmax(feature)
        else:
            self.label = 0
        self.keypoints = np.array(keypoints[:, :2]).astype(np.float32).reshape(-1, 2) 
        self.score = score
        self.visible = visible             
        if len(feature)<1:
            self.feature = np.zeros((1, 28))
            self.feature[0, label] = 1.0
        else:
            self.feature = np.array(feature).astype(np.float32).reshape(1, -1)     

    @property
    def bbox(self, mode="xyxy"):
        kpts = self.keypoints.reshape(-1, 2)    
        box = np.array([np.min(kpts[:,0]), np.min(kpts[:,1]), np.max(kpts[:,0]), np.max(kpts[:,1])])    
        if mode == "tlwh":
            box = np.concatenate((box[:2], box[2:]-box[:2]), axis=-1)        
        elif mode == "xywh":
            box = np.concatenate(((box[:2]+box[2:])/2, box[2:]-box[:2]), axis=-1)  
        return box

    @property
    def polygon_area(self):
        A = 0.0
        pt1 = self.keypoints[-1]
        for pt2 in self.keypoints:
            A += (pt1[1]*pt2[0]-pt1[0]*pt2[1])
            pt1 = pt2
        return abs(A)/2

    @property
    def bbox_area(self):
        return (self.bbox[2] - self.bbox[0])*(self.bbox[3] - self.bbox[1])    

    @property
    def centroid(self):
        A = self.polygon_area+np.spacing(1)
        cx, cy = 0.0, 0.0  
        pt1 = self.keypoints[-1]
        for pt2 in self.keypoints:
            cx += ((pt2[0]+pt1[0])*(pt2[1]*pt1[0]-pt2[0]*pt1[1]))
            cy += ((pt2[1]+pt1[1])*(pt2[1]*pt1[0]-pt2[0]*pt1[1]))
            pt1 = pt2
        if cx==0 and cy==0:
            cx = np.mean(self.keypoints[:, 0])*6*A
            cy = np.mean(self.keypoints[:, 1])*6*A
            # print(self.keypoints)
        return cx/(6*A), cy/(6*A)

    @property
    def AnteriorHeight(self):
        x1, y1 = self.keypoints[0]  
        x2, y2 = self.keypoints[-1]         
        return ((x2-x1)**2+(y2-y1)**2)**0.5

    @property
    def MiddleHeight(self):
        x1, y1 = self.keypoints[1]  
        x2, y2 = self.keypoints[-2]         
        return ((x2-x1)**2+(y2-y1)**2)**0.5
    
    @property
    def PosteriorHeight(self):
        x1, y1 = self.keypoints[2]  
        x2, y2 = self.keypoints[-3]         
        return ((x2-x1)**2+(y2-y1)**2)**0.5    

    @property
    def AverageHeight(self):              
        #return (self.AnteriorHeight+self.MiddleHeight+self.PosteriorHeight)/3
        return (self.AnteriorHeight+self.PosteriorHeight)/2

    def Contains(self, pt_in=[0, 0]):
        #判断一个点是否包含在这个多边形中
        result = False
        x, y = pt_in
        x1, y1 = self.keypoints[-1]
        for (x2, y2) in self.keypoints:
            if (x==x1 and y==y1) or (x==x2 and y==y2):
                return True
            if (y2>y) != (y1>y):
                x3 = x2+(y-y2)*(x1-x2)/(y1-y2)
                if x==x3:
                    return True
                if x<x3:
                    result = not result
            x1, y1 = x2, y2
        return result

    def IOU(self, bbox):
        #返回指定bbox与本实例的交并比
        return IOU(self.bbox, bbox, iou_type="iou")

    def Move(self, value=[0, 0]):
        #value=(x, y), x<0是向左移动，x>0是向右移动，y<0是向上移动， y>0是向下移动
        #value = torch.tensor(value)
        self.keypoints += value

    def Resize(self, value=[0, 0]):
        #value = torch.tensor(value)
        self.keypoints[:, 0] = self.keypoints[:, 0]*value[0]
        self.keypoints[:, 1] = self.keypoints[:, 1]*value[1]

def IOU(a, b, iou_type = "iou", mode = "xyxy"):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        iou_type: str, ["iou","giou","diou","ciou","eiou"]
        Returns
        -------
        IoUs: (N, K) ndarray of overlap between boxes and query_boxes
        """

        a = a.reshape(-1,4)
        b = b.reshape(-1,4)
        a = ConvertCoords(a, mode="{}2xyxy".format(mode))
        b = ConvertCoords(b, mode="{}2xyxy".format(mode))

        a_w,a_h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]   #[N]
        b_w,b_h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]   #[K]  

        a_area = a_w*a_h    #[N]
        b_area = b_w*b_h    #[K]

        i_w = np.minimum(a[:, 2].reshape(-1,1), b[:, 2]) - np.maximum(a[:, 0].reshape(-1, 1), b[:, 0]) #[N,K]
        i_h = np.minimum(a[:, 3].reshape(-1, 1), b[:, 3]) - np.maximum(a[:, 1].reshape(-1, 1), b[:, 1]) #[N,K]
        i_w = i_w.clip(0)
        i_h = i_h.clip(0)
        i_area = i_w * i_h  #[N,K]
        u_area = np.expand_dims(a_area, 1) + b_area - i_area #[N,K]
        u_area = u_area.clip(1e-8) 

        IoUs = i_area / u_area #[N,K]

        if iou_type == "diou" or iou_type == "ciou":
            c_w = np.maximum(a[:, 2].reshape(-1, 1), b[:, 2]) - np.minimum(a[:, 0].reshape(-1, 1), b[:, 0])  #最小闭包区域的width
            c_h = np.maximum(a[:, 3].reshape(-1, 1), b[:, 3]) - np.minimum(a[:, 1].reshape(-1, 1), b[:, 1])  #最小闭包区域的height
            c2 = c_w**2 + c_h**2
            rho2 = ((b[:,0]+b[:,2]-a[:, 0].reshape(-1, 1)-a[:, 2].reshape(-1, 1))**2+(b[:,1]+b[:,3]-a[:, 1].reshape(-1, 1)-a[:, 3].reshape(-1, 1))**2)/4  #中心距离
            alpha = 1e-8
            if iou_type == "ciou":
                v = (4/math.pi**2)*np.power(np.arctan(b_w/b_h)-np.arctan(a_w/a_h).reshape(-1,1),2)
                alpha = v**2/(v-IoUs+(1+1e-8))
            IoUs = IoUs - rho2/(c2+alpha)
        elif iou_type == "giou":
            cw = np.maximum(a[:, 2], b[:, 2]) - np.minimum(a[:, 0], b[:, 0])  #最小闭包区域的width
            ch = np.maximum(a[:, 3], b[:, 3]) - np.minimum(a[:, 1], b[:, 1])  #最小闭包区域的height
            c_area = cw*ch
            IoUs = IoUs - (c_area-u_area)/c_area       

        return np.array(IoUs)

def SIM(a, b, metric = "cosine"):
        """
        Parameters
        ----------
        a: list
        b: list
        metric: str, "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule".
        Returns
        -------
        sims: (N, K) ndarray of similarity between a and b
        """
        sims = np.zeros((len(a),len(b)))
        for i, va in enumerate(a):
            for j, vb in enumerate(b):
                data = np.vstack([va.feature, vb.feature])
                sims[i, j] = pdist(data, "cosine")
      
        return 1-sims

def OKS(dks, gks, gt_bboxes, sigmas):
    """
    parameters
    ----------
    ka: (n, m) list [[x1,y1,x2,y2,x3,y3,...]]
    kb: (k, m) list [[x1,y1,x2,y2,x3,y3,...]]
    returns
    -------
    oks: (n, k) compute oks between each detection and ground truth object
    """
            
    # dimention here should be NxK

    oks = np.zeros((len(dks), len(gks)))   #[N, K]
    vars = (sigmas * 2)**2
    k = len(sigmas)

    # compute oks between each detection and ground truth object
    for j, gk in enumerate(gks):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gk)
        xg = g[0::2]; 
        yg = g[1::2]; 
        
        bb = gt_bboxes[j]
        gt_area = (bb[0] - bb[2])*(bb[1] - bb[3])
        for i, dk in enumerate(dks):
            d = np.array(dk)
            xd = d[0::2]; 
            yd = d[1::2]
            
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
           
            e = (dx**2 + dy**2) / vars / (gt_area+np.spacing(1)) / 2
            #exp = np.exp(-e)
            value = np.sum(np.exp(-e)) / e.shape[0]
            oks[i, j] = value if value > 0.0001 else 0
    return oks

def ConvertCoords(bboxes, mode="xyxy2tlwh"):
    assert mode in ["xyxy2tlwh","tlwh2xyxy","xyxy2xywh","xywh2xyxy","tlwh2xywh","xywh2tlwh","xyxy2xyxy","xywh2xywh","tlwh2tlwh"], "mode is not recognize"

    if bboxes.shape[0]:
        bboxes = bboxes.reshape(-1, 4)
        if mode == "xyxy2tlwh":
            bboxes = np.concatenate((bboxes[...,:2], bboxes[...,2:]-bboxes[...,:2]),axis=-1) 
        elif mode == "tlwh2xyxy":
            bboxes = np.concatenate((bboxes[...,:2], bboxes[...,:2]+bboxes[...,2:]),axis=-1)
        elif mode == "xyxy2xywh":
            bboxes = np.concatenate(((bboxes[...,:2]+bboxes[...,2:])/2, bboxes[...,2:]-bboxes[...,:2]),axis=-1) 
        elif mode == "xywh2xyxy":
            bboxes = np.concatenate((bboxes[...,:2]-bboxes[...,2:]/2, bboxes[...,:2]+bboxes[...,2:]/2),axis=-1)
        elif mode == "tlwh2xywh":
            bboxes = np.concatenate((bboxes[...,:2]+bboxes[...,2:]/2, bboxes[...,2:]),axis=-1)
        elif mode == "xywh2tlwh":
            bboxes = np.concatenate((bboxes[...,:2]-bboxes[...,2:]/2, bboxes[...,2:]),axis=-1)
        else:
            bboxes = bboxes 
    else:
        bboxes = bboxes 
    if type=="ndarray":
        bboxes = np.array(bboxes)
    return bboxes