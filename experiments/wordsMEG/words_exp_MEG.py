"""
==========================
Localizer - Words, Faces, Objects, Scramble
==========================
"""

import numpy as np
from expyfun import visual, ExperimentController
from PIL import Image
import os
import csv

# Directory where image files are
imagedir = "/mnt/diskArray/projects/KNK/readingD/readingD1B/"
# List of images
imagelist = sorted(os.listdir(imagedir))

# Load up event onset times
onsets = []
with open('/mnt/diskArray/projects/KNK/readingD/readingOnsets1.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        onsets.extend(row)
onsets = map(int, onsets)
# Load up even numbers associated with the onsets
eventnums = []
with open('/mnt/diskArray/projects/KNK/readingD/onsetsEventNums.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        eventnums.extend(row)
eventnums = map(int, eventnums)

# Experiment controller can not handle 22 event numbers so we will loop over
# each event number and recode as 2 numbers
c = -1
for ii in eventnums:
    c = c+1
    if ii <= 12:
        eventnums[c] = [1,ii]
    else:
        eventnums[c] = [2,ii-12]

# with ExperimentController('ReadingExperiment', full_screen=True) as ec:
with ExperimentController('ReadingExperiment', full_screen=False, window_size=[800,800]) as ec:
    # Set the background color to gray
    ec.set_background_color([180/255.,180/255.,180/255.,1])
    # load up the image stack
    img=[]
    for im in imagelist:
     img_buffer = np.array(Image.open(imagedir+im))/255.
     img.append(visual.RawImage(ec, img_buffer, scale=2.))

    # do the drawing, then flip
    frametimes=[]
    buttons   =[]
    ec.listen_presses()
    last_flip = -1
    # Initiate a counter
    c = -1
    for obj in img:
     c = c+1
     # If this frame is the beginning of a new event type then we mark it in
     # The MEG data file
     if c in onsets:
      ec.call_on_next_flip(ec.stamp_triggers,eventnums[onsets.index(c)], check='int4')
       
     # Draw the image and flip the screen 
     obj.draw()
     last_flip=(ec.flip(last_flip+0.2))
     frametimes.append(last_flip)

    pressed = ec.get_presses()  # relative_to=0.0
