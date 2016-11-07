import os
import cv2
import numpy as np
import sys
import json
from sampleGenerator.sample import sample



input_folder = '/home/frivas/pentalo/sample_generator/sample/data1/images/'
output_folder = './out/'


def getIfPropertyToTest(sampleNumber, testRatio):
        return  sampleNumber % 10 < testRatio*10



def main():
    minPoints=80
    index=0
    imagesList=[]
    fileLists= os.listdir(input_folder+'/camera1/')
    imagesList += [each for each in fileLists if each.endswith('.png')]
    sorted_files = sorted(imagesList, key=lambda x: int(x.split('.')[0]))

    print 'source ', input_folder +  '/camera1/' + sorted_files[1]

    sourceImage=cv2.imread(input_folder + '/camera1/' +  sorted_files[1])
    sourceImage=cv2.cvtColor(sourceImage,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",sourceImage)

    sourceGray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    sourceGray = cv2.GaussianBlur(sourceGray, (21, 21), 0)

    (H,W,c) = np.shape(sourceImage)
    cv2.waitKey(0)

    warps=[0,90,180,270]
    scales=[1,1.5,2]
    testRatio=0.3
    imagesCounter=0
    for image in sorted_files[2:-1]:
        testImage=False
        for warp in warps:
            print "w: ", warp
            for scale in scales:
                imagePath = input_folder +  '/camera1/' + image
                currImage=cv2.imread(imagePath)
                currImage=cv2.cvtColor(currImage,cv2.COLOR_BGR2RGB)


                gray = cv2.cvtColor(currImage, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                frameDelta = cv2.absdiff(sourceGray, gray)
                thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

                # dilate the thresholded image to fill in holes, then find contours
                # on thresholded image
                # thresh = cv2.dilate(thresh, None, iterations=2)
                kernel = np.ones((5,5),np.uint8)
                thresh=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                cv2.imshow("th",thresh)




                nPoints = cv2.countNonZero(thresh)
                if nPoints < minPoints * scale:
                    nPoints=0
                if nPoints:
                    filename, file_extension = os.path.splitext(image)
                    if (warp == warps[0] and scale == scales[0]):
                        imagesCounter+=1
                    testImage=getIfPropertyToTest(imagesCounter,testRatio)
                    print testImage
                    if testImage:
                        cv2.imwrite("test/" + image,currImage)

                    points = cv2.findNonZero(thresh)
                    roi = cv2.boundingRect(points)
                    s = sample(currImage,roi)
                    s.applyScale(scale)
                    s.applyWarp(warp)
                    s.printResult()
                    maskedImage = s.getImageWithData()
                    cv2.imshow("current",maskedImage)
                    if testImage:
                        cv2.imwrite("test/" + filename + '-mask.png',maskedImage)
                    else:
                        cv2.imwrite(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) +'_' + filename + "-mask.png",maskedImage)
                        cv2.imwrite(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) + '_' + filename + '.png',s.getImage())

                        f = open(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) + '_' + filename + '.txt', 'w')
                        f.write(s.getResultStr())
                        f.close()

                    cv2.waitKey(1)
                if testImage:
                    break
            if testImage:
                break


    print imagesCounter

if __name__ == "__main__":
    main()