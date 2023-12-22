import cv2
import numpy as np
import os

def readImagesAndTimes():
  
    times = np.array([1/30.0, 0.25, 2.5, 15.0], dtype=np.float32)
  
    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        if im is None:
            print(f"Error loading image {filename}")
            exit()
        images.append(im)
  
    return images, times

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    print("Reading images ... ")
    images, times = readImagesAndTimes()
  
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
  
    print("Calculating Camera Response Function (CRF) ... ")
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)
  
    print("Merging images into one HDR image ... ")
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

    ensure_dir("output")
    cv2.imwrite("output/hdrDebevec.hdr", hdrDebevec)
    print("saved hdrDebevec.hdr ")
  
    print("Tonemaping using Drago's method ... ")
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("output/ldr-Drago.jpg", ldrDrago * 255)
    print("saved ldr-Drago.jpg")

# This algorithm is patented and is excluded in this configuration  
    # print("Tonemaping using Durand's method ... ")
    # tonemapDurand = cv2.xphoto.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    # ldrDurand = tonemapDurand.process(hdrDebevec)
    # ldrDurand = 3 * ldrDurand
    # cv2.imwrite("output/ldr-Durand.jpg", ldrDurand * 255)
    # print("saved ldr-Durand.jpg")
  
    print("Tonemaping using Reinhard's method ... ")
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite("output/ldr-Reinhard.jpg", ldrReinhard * 255)
    print("saved ldr-Reinhard.jpg")
  
    print("Tonemaping using Mantiuk's method ... ")
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite("output/ldr-Mantiuk.jpg", ldrMantiuk * 255)
    print("saved ldr-Mantiuk.jpg")
