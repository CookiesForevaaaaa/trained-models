#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

<<<<<<< HEAD
parser.add_argument("input", type=str, default="/dev/video2", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="rtp://192.168.1.18:1234", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet.onnx", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.2, help="minimum detection threshold to use") 
=======
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.25, help="minimum detection threshold to use") 
>>>>>>> 1df50cb37b50fbab4f7625aadb00abbef7b1a608
parser.add_argument("--input-codec", type=str, default="h264", help="codec of the input stream (e.g., h264, h265)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource("/dev/video2", argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
<<<<<<< HEAD


=======
>>>>>>> 1df50cb37b50fbab4f7625aadb00abbef7b1a608
	
# load the object detection network
#net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
<<<<<<< HEAD
net = detectNet(model="ssd-mobilenet.onnx", labels="labels.txt", 
=======
net = detectNet(model="models/eye/ssd-mobilenet.onnx", labels="models/eye/labels.txt", 
>>>>>>> 1df50cb37b50fbab4f7625aadb00abbef7b1a608
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                threshold=args.threshold)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()
<<<<<<< HEAD
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)
    # Get resolution
#    width = input.GetWidth()
#    height = input.GetHeight()
#    print(f"Stream Resolution: {width}x{height}")

################## DETEKCIJA IN OBDELAVA #######################

# Resolucija videa je 1280x720 .. polovica X = 640 ampak damo še 100 v levo in 100 v desno zraven
    middle_left = 540   # 640 - 100
    middle_right = 740  # 640 + 100
    for detection in detections:
        if (detection.ClassID == 2): # ID = 1 je za oke
            #print("Na sliki je: ")
            #print(net.GetClassDesc(detection.ClassID))


            if middle_left <= detection.Center[0] <= middle_right:
               print("%s je na sredini" % net.GetClassDesc(detection.ClassID))
            elif detection.Center[0] < middle_left:
                print("%s je na levi" % net.GetClassDesc(detection.ClassID))
            else:
                print("%s je na desni" % net.GetClassDesc(detection.ClassID))
                #print("Slika je velika W:%s H:%s" % (camera.GetWidth(), camera.GetHeight()))
                #print(detections)
    if img is None: # timeout
        continue

    # print the detections
#    print("detected {:d} objects in image".format(len(detections)))

#    for detection in detections:
#        print(detection)
=======

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)
>>>>>>> 1df50cb37b50fbab4f7625aadb00abbef7b1a608

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
