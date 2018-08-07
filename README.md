# flowdetect

This is a small holiday project trying to detect what humans describe as "objects" by inspecting visual flow in an image.

## Idea of the project

All this program does is computing the image flow in a camera image. Then it groups pixels based on whether they are pyhsically near and moving. The groups are marked in the output by their corresponding convex hull.

## Libraries

For image processing OpenCV is used. If you want to build this yourself, make sure OpenCV is set up correctly on your system and in your IDE.

## TO-DOs

 - Group images based on whether they move in the same direction.
 - Train a machine learning based system on what pixels belong together in one object group. Take into account possible tracking results and time and location proximity
