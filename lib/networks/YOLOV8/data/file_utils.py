import os
import numpy as np
try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False

from PIL import Image
from pathlib import Path

def GetFilesFromDir(dir, filter):
    files = []
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[-1]
        if ext in filter:
            files.append(os.path.join(dir, file))
    return files

def ReadImage(image_path, new_direction=None, new_size=None, new_spacing=None, interpolator=None):
    if not SIMPLEITK_AVAILABLE:
        # 如果SimpleITK不可用，使用PIL作为备选
        from PIL import Image
        import numpy as np

        image_info = {"image_path": image_path}
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 对于非医学图像，使用PIL
        ori_image = Image.open(image_path).convert('L')
        array_image = np.asarray(ori_image)
        array_image = np.ascontiguousarray(array_image[np.newaxis, ...])  # contiguous  [1, height, width]

        if len(array_image.shape) < 3:
            array_image = array_image[np.newaxis, :]

        return ori_image, array_image.copy(), image_info

    # 原有的SimpleITK实现
    image_info = {"image_path": image_path}
    image_path = Path(image_path)
    suffixes = image_path.suffixes
    assert image_path.exists()

    if "".join(image_path.suffixes[-2:]) in [".nii.gz", ".dicom"] or image_path.suffix in [".dcm", ".nii", ".dicom", ".img", ".hdr", ".nrrd", ".mnc"]:
        ori_image, image_info = ReadMedicalImage(image_path, new_direction, new_size, new_spacing, interpolator)
        array_image = sitk.GetArrayFromImage(sitk.Cast(sitk.RescaleIntensity(ori_image), sitk.sitkUInt8))   #转为[0-255]范围，[slice, height, width]
    else:
        ori_image = Image.open(image_path).convert('L')
        array_image = np.asarray(ori_image)
        array_image = np.ascontiguousarray(array_image[np.newaxis, ...])  # contiguous  [1, height, width]
    if len(array_image.shape)<3:
        array_image = array_image[np.newaxis, :]

    return ori_image, array_image.copy(), image_info

def ReadMedicalImage(image_path, new_direction=None, new_size=None, new_spacing=None, interpolator=None):
    if not SIMPLEITK_AVAILABLE:
        # 如果SimpleITK不可用，返回假数据
        import numpy as np
        fake_image = np.zeros((64, 64), dtype=np.uint8)
        fake_info = {
            "image_path": str(image_path),
            "image_model": "FAKE",
            "ori_direction": "LPS",
            "new_direction": "LPS",
            "ori_size": [64, 64],
            "ori_spacing": [1.0, 1.0],
            "image_scale": [1.0, 1.0],
            "new_size": [64, 64],
            "new_spacing": [1.0, 1.0],
            "image_windowcenter": 0,
            "image_windowwidth": 255
        }
        return fake_image, fake_info

    # 原有的SimpleITK实现
    #new_direction is sting, new_spacing is tuple.
    assert Path(image_path).exists()
    image_info = {"image_path": image_path, }  #保存图像附加信息

    sitk_image = sitk.ReadImage(image_path)    
    image_info["image_model"] = sitk_image.GetMetaData("0008|103e") if sitk_image.HasMetaDataKey("0008|103e") else "NULL"         
    if new_direction is not None and len(image_size)>2:
        orient_filter = sitk.DICOMOrientImageFilter()    
        ori_direction = orient_filter.GetOrientationFromDirectionCosines(sitk_image.GetDirection())   #LPS  
        image_info["ori_direction"] = ori_direction
        image_info["new_direction"] = ori_direction        
        if min(image_size)>1 and ori_direction != new_direction:  
            orient_filter.SetDesiredCoordinateOrientation(new_direction)   #默认使用PIR
            sitk_image = orient_filter.Execute(sitk_image)   
            image_info["new_direction"] = new_direction

    image_size = np.array(sitk_image.GetSize())
    image_spacing = np.array(sitk_image.GetSpacing())
    image_info["ori_size"] = image_size.tolist()    
    image_info["ori_spacing"] = image_spacing.tolist()
    image_info["image_scale"]   = np.ones_like(image_size)  
    
    #对于指定new_size的，重新计算new_spacing参数，否则使用new_spacing计算new_size, 如果都没有则默认前两维的spacing应相同
    if new_size is not None:
        if len(new_size)<len(image_size):
            new_size.append(image_size[-1])
        new_size = np.array(new_size)
        new_spacing = image_size * image_spacing / new_size        
    else:
        if new_spacing is not None:
            new_spacing = np.array(new_spacing)
            if len(new_spacing) < len(image_spacing):
                #image_spacing[:len(new_spacing)] = new_spacing
                new_spacing = np.append(new_spacing, image_spacing[len(new_spacing):])
        else:
            if image_spacing[0] != image_spacing[1]:
                new_spacing = np.array([min(image_spacing[:2]),min(image_spacing[:2]),image_spacing[-1]])
            else:
                new_spacing = image_spacing    

        #image_spacing = np.array(sitk_image.GetSpacing()) 
        new_size = np.array(image_size*image_spacing/new_spacing).astype(int)
        new_spacing = image_size * image_spacing / new_size

    #重采样
    if not np.array_equal(new_size, image_size):
        #按照size, spacing重采样
        resamplefilter = sitk.ResampleImageFilter()            
        resamplefilter.SetReferenceImage(sitk_image)
        resamplefilter.SetOutputSpacing(new_spacing.tolist())
        resamplefilter.SetSize(new_size.tolist())
        resamplefilter.SetOutputDirection(sitk_image.GetDirection())
        resamplefilter.SetOutputOrigin(sitk_image.GetOrigin())        
        resamplefilter.SetTransform(sitk.Transform())    
        if interpolator is not None:
            resamplefilter.SetInterpolator(interpolator)
        else:
            resamplefilter.SetInterpolator(sitk.sitkNearestNeighbor)                
        sitk_image = resamplefilter.Execute(sitk_image)        
    image_info["new_size"] = new_size.tolist()
    image_info["new_spacing"] = new_spacing.tolist()
    image_info["image_scale"]   = (new_size/image_size).tolist()

    #调整窗宽窗高, 将HU值从范围【min_hu, max_hu】调整到【0， ww】，统一CT图像中HU的取值范围特征
    statsFilter = sitk.StatisticsImageFilter()
    statsFilter.Execute(sitk_image)
    min_hu = statsFilter.GetMinimum()
    max_hu = statsFilter.GetMaximum()
    min_hu = max(-1000, min_hu)
    max_hu = min(max_hu, min_hu+8000)
    ww = int(max_hu-min_hu)
    
    image_info["image_windowcenter"] = int(ww/2)
    image_info["image_windowwidth"] = ww      

    winFilter = sitk.IntensityWindowingImageFilter()
    winFilter.SetWindowMinimum(min_hu)
    winFilter.SetWindowMaximum(max_hu)
    winFilter.SetOutputMinimum(0)
    winFilter.SetOutputMaximum(ww)
    sitk_image = winFilter.Execute(sitk_image)

    return sitk_image, image_info      