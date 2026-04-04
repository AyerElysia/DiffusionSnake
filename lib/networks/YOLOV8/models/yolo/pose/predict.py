# Ultralytics YOLO 🚀, AGPL-3.0 license

from ....engine.results import Results
from ..detect.predict import DetectionPredictor
from ....utils import DEFAULT_CFG, LOGGER, ops

###duan
import os
import cv2
import numpy as np
import torch
###duan

class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results


    #########duan20231002  
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        ratio_pad = []
        im_list = []
        imgsz = self.imgsz[0] if isinstance(self.imgsz, list) else self.imgsz
        for im0 in im:
            #如果图像尺寸小于模型默认尺寸就先放大到模型指定尺寸，然后再padding
            height, width = im0.shape[:2]       #[slice, height, width]
            if max(height, width)<imgsz:
                r = imgsz / max(height, width)

                new_w, new_h = int(round(width * r)), int(round(height * r))
                im0 = cv2.resize(im0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                new_w, new_h = width, height
            max_w = max(imgsz, new_w + (32 - new_w%32)%32)
            max_h = max(imgsz, new_h + (32 - new_h%32)%32)

            #padding
            dw, dh = (max_w - new_w)/2, (max_h - new_h)/2  # wh padding
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            
            ratio_pad.append([(new_w/width, new_h/height), (left, top)])  #记录下来图像的变化，用于复原     [(x_factor, y_factor)] 

            im0 = cv2.copyMakeBorder(im0, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # add border
            im0 = im0.astype(np.float32)/255.0
            
            im_list.append(im0[np.newaxis, ...])  # Channel first

        im = np.stack(im_list)
        im = np.ascontiguousarray(im).copy()  ###duan  contiguous
        im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        return im, ratio_pad    #duan20240720  
    
    def postprocess_medicalimage(self, preds, img, orig_imgs, ratio_pad):
        """Return detection results for a given input image or list of images."""
        
        #img是pad之后的图像，orig_imgs是pad之前的图像（spacing=1），但不是原始图像
        preds, feats = ops.nms(preds,
                                self.args.conf,
                                self.args.iou,
                                agnostic=self.args.agnostic_nms,
                                max_det=self.args.max_det,
                                classes=self.args.classes,
                                nc=len(self.model.names))
        isDebug = 0
        results = {}
        path, _, gold_data = self.batch
        for i, pred in enumerate(preds):            
            img_path = path[i] if isinstance(path, list) else path
            img_file, slice = img_path.rsplit("_", 1)            
            img_name = os.path.basename(img_file)
            sliceid = "slice_{}".format(slice)
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            ###############
            golds = []
            if gold_data is not None:
                gold = gold_data[i]
                image_size = gold["image_size"]              #[width, height, slice]
                image_spacing = gold["image_spacing"]        #[x, y, z]        
                gold_labels = gold["labels"]
                for gi, glabel in enumerate(gold_labels):
                    kpts = gold["keypoints"][gi]    
                    bbox = gold["bboxes"][gi]  ###np.array([np.min(kpts[:,0]), np.min(kpts[:,1]), np.max(kpts[:,0]), np.max(kpts[:,1])])    

                    golds.append({"annot_type": "polygon",
                                    "category": self.model.names[int(glabel)],
                                    "visible": True,
                                    "keypoint": kpts.tolist(),
                                    "bbox": bbox.tolist(),
                                    "bbox_mode": "xyxy",
                                    "score": 1.0,}
                    )
            else:
                # image_size = orig_img.shape
                image_spacing = [1,1,1]
            ###############
            # pad_left, pad_top = ratio_pad[i][1]
            image_size = orig_img.shape   #[height, width, 1]
            # ratio_pad[i][0] = (img.shape[2] / image_size[0], img.shape[3] / image_size[1])

            pred_bbox = pred[:, :4]             #xyxy           
            pred_bbox = ops.scale_boxes(img.shape[2:], pred_bbox, image_size, ratio_pad[i]).round()
            pred_bbox[:, ::2] /= image_spacing[0]
            pred_bbox[:, 1::2] /= image_spacing[1]

            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]        
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, image_size, ratio_pad[i])
            pred_kpts[..., 0] /= image_spacing[0]
            pred_kpts[..., 1] /= image_spacing[1]

            detects = []
            for pi, p in enumerate(pred):
                if int(p[5])>0:
                    kpts = pred_kpts[pi].cpu().reshape(-1, 2)    
                    bbox = pred_bbox[pi].cpu()    
                    detects.append({"annot_type": "polygon",
                                    "category": self.model.names[int(p[5])],
                                    "visible": True,
                                    "keypoint": kpts.tolist(),
                                    "bbox": bbox.tolist(),
                                    "bbox_mode": "xyxy",
                                    "score": float(p[4]),
                                    "feature": feats[i][pi].cpu().tolist(),}
                    )

            if img_name not in results.keys():
                results[img_name]={"image_size": image_size,
                                   "image_spacing": image_spacing,
                                   "image_scale": [1, 1, 1],
                                   "gold_data":{},
                                   "detect_data":{}}
            if len(detects):
                results[img_name]["detect_data"][sliceid] = detects
            if len(golds):
                results[img_name]["gold_data"][sliceid] = golds

            ########################check image
            
            if isDebug and len(detects) and len(golds):
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import numpy as np
                fig,ax = plt.subplots(figsize = (12,12), ncols = 2, nrows = 1)   
                ax[0].set_title(f"postprocess_medicalimage({img_name}) and labels and bboxes and keypoints") 
                ax[0].imshow(orig_img[...,0], cmap="gray")                 
                ax[1].imshow(orig_img[...,0], cmap="gray")   
                for ki, key in enumerate(["gold_data", "detect_data"]):
                    for r, result in enumerate(results[img_name][key][sliceid]):
                        if key == "gold_data":
                            colour = "yellow"
                        else:
                            colour = "white"

                        if result["visible"]:                                                    
                            linestyle = "-"
                        else:                                                                    
                            linestyle = "- -"
                        label = result["category"]
                        score = result["score"]                                 
                        kpts = np.array(result["keypoint"])
                        # if key=="gold_data":
                        kpts *= np.array(image_spacing)[:2]      
                        if len(kpts)==2:
                            ax[ki].add_patch(patches.Rectangle(tuple(kpts[0]), kpts[1][0]-kpts[0][0], kpts[1][1]-kpts[0][1], fill= False, edgecolor = colour, linestyle = linestyle))   
                        else:                       
                            ax[ki].add_patch(patches.Polygon(kpts, fill= False, edgecolor = colour, linestyle = linestyle))      
                        xy = kpts[ki]
                        xytext = (xy[0]+100, xy[1])
                        ax[ki].annotate("{}:({})({:.4f})".format(key, label, score),
                                        xy=xy,
                                        xytext=xytext,
                                        size=10,
                                        va="center",
                                        ha="left",
                                        arrowprops=dict(color=colour,arrowstyle="simple",connectionstyle="arc3,rad=0.2",), 
                                        color = colour, 
                                        bbox=dict(boxstyle='round,pad=0.5', facecolor="blue", alpha=0.5))   
                isDebug = 0   
                plt.show()
                plt.close()
            ########################  
        return results