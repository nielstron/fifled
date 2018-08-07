# FlowDetect

This is a small holiday project trying to detect what humans describe as "objects" by inspecting visual flow in an image.

## Idea

All this program does is computing the image flow in a camera image. Then it groups pixels based on whether they are pyhsically near and moving. The groups are marked in the output by their corresponding convex hull.

As this only uses a minimal amount of algorithms, the resulting output is really fast. Yet it also resembles a lot how humans detect objects. At least this is assumed, as there is nothing else about objects than "belonging" together. This is identified by them sharing a direction of movement and being physically near. Among frames objects can be grouped by near objects in the preceding frames. Thus different views of the same object get grouped together.

## Libraries

For image processing OpenCV is used. If you want to build this yourself, make sure OpenCV is set up correctly on your system and in your IDE.

## Demo Video

[![Watch the demo video](thumb.png "Click to get redirected to the Youtube Video")](https://youtu.be/l5aenMUADbg)

## TO-DOs

 - Group images based on whether they move in the same direction.
 - Train a machine learning based system on what pixels belong together in one object group. Take into account possible tracking results and time and location proximity
