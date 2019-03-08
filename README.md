# :oncoming_automobile: :taxi: :bus: Smart-Driver-Drowsiness-Detection :blue_car: :articulated_lorry: :minibus:

Detecting the drowsiness level of the driver and notify or alert him if he reaches the limit of the drowsiness hazard level
We will follow three main techniques to detect the drowsiness that are
1. Monitoring vehicle movements :man:
2. Monitoring the vehicale movements. :eyes: :oncoming_bus:
3. Monitoring the pulse rate :heart:

## Tracking eye movements :eyes:

  We use the Convolutional Neural Netwoks algorithms to detect and track the eye. The CNN algorithm implements from scratch according to VGG-16 architecture. 

## Monitering the vehicle movements.:oncoming_police_car:
  Road lane detection is a supported evidence for detecting the drowsiness. In here I use openCV libraries to do the image processing. First get the frames from the video. then process frame by frame. In each frame filter the edges of the objects using color range of the image which is basically filter yellow and white colors. Get the range of interest(roi) by cropping the frame. Then using Hough lines transformation draw lines on the frame according to that edges in roi. To calculate the angle, first divide the frame in to 2 vertical halfs and using houghlines function to each frame half filter the angle value. Using angle values if 1 side angle reducing and other side angle increasing consider it as driving away from the road lines, or if only one angle detecting if that angle reducing or increasing consider as driving away from the road. If not detecting any road line, ignore the result. If the result is driving away from the road confirm it by eye movement tracking to detect drowsiness.
  
  ## Monitering the pulse rate :chart_with_upwards_trend:
  check the heart pulse rate whether it exceeds normal pulse rate and if so confirm it by considering previous 2 data and alert drowsy state.
