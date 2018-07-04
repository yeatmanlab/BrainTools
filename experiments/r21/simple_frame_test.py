
"""
==========================
Display images for MEG
==========================
"""
import numpy as np
from expyfun import visual, ExperimentController
from expyfun.io import write_hdf5
import time
from PIL import Image


import os
from os import path as op
import glob

# background color2
testing = True
test_trig = [15]
if testing:
    bgcolor = [0., 0., 0., 1.]
else:
    bgcolor = [127/255., 127/255., 127/255., 1]

# Paths to images
#basedir = '/home/jyeatman/projects/MEG/images/'
basedir = os.path.join('C:\\Users\\neuromag\\Desktop\\jason\\wordStim')
if not os.path.isdir(basedir):
    basedir = os.path.join('/home/jyeatman/git/wmdevo/megtools/stim/wordStim')

imagedirs = ['word_c254_p20', 'word_c254_p50', 'word_c137_p20',
             'word_c254_p80', 'word_c137_p80', 'bigram_c254_p20',
             'bigram_c254_p50', 'bigram_c137_p20']

nimages = [20, 20, 20, 20, 20, 7, 7, 7]  # number of images in each category
if len(nimages) == 1:
    nimages = np.repeat(nimages, len(imagedirs))
# ISIs to be used. Must divide evenly into nimages
isis = np.arange(.62, .84, .02)
imduration = 1.  # Image duration
s = .5  # Image scale

# Create a vector of ISIs in a random order. One ISI for each image
rng = np.random.RandomState(int(time.time()))
ISI = np.repeat(isis, sum(nimages)/len(isis))
rng.shuffle(ISI)

# Creat a vector of dot colors for each ISI
c = ['g', 'b', 'y', 'c', '#6B8E23', '#F0E68C', '#8FBC8F']
dcolor = np.tile(c, np.ceil(float(len(ISI))/len(c)))
dcolor2 = np.tile(c, np.ceil(float(len(ISI))/len(c)))
dcolor3 = np.tile(c, np.ceil(float(len(ISI))/len(c)))
# Now insert the target trials
dcolor[:7] = 'r'
dcolor2[:7] = 'r'
dcolor3[:7] = 'r'
# becasue we rounded up in terms of the number we needed we now remove extras
dcolor = dcolor[:len(ISI)]
dcolor2 = dcolor2[:len(ISI)]
dcolor3 = dcolor3[:len(ISI)]

# Shuffle the order of the fixation dot
rng.shuffle(dcolor)
rng.shuffle(dcolor2)

# Create a vector denoting when to display each image
imorder = np.arange(sum(nimages))
# Create a vector marking the category of each image
imtype = []
for i in range(1, len(imagedirs)+1):
    imtype.extend(np.tile(i, nimages[i-1]))

# Shuffle the image order
ind_shuf = range(len(imtype))
rng.shuffle(ind_shuf)
imorder_shuf = []
imtype_shuf = []
for i in ind_shuf:
    imorder_shuf.append(imorder[i])
    imtype_shuf.append(imtype[i])

# Build the path structure to each image in each image directory. Each image
# category is an entry into the list. The categories are in sequential order 
# matching imorder, but the images within each category are random
imagelist = []
imnumber = []
c = -1
for imname in imagedirs:
    c = c+1
    # Temporary variable with image names in order
    tmp = sorted(glob.glob(os.path.join(basedir, imname, '*')))
    # Randomly grab nimages from the list
    n = rng.randint(0, len(tmp), nimages[c])
    tmp2 = []
    for i in n:
        tmp2.append(tmp[i])
    # Add the random image list to an entry in imagelist
    imagelist.extend(tmp2)
    # record the image number
    imnumber.extend(n)
    assert len(imagelist[-1]) > 0

# Start instance of the experiment controller
with ExperimentController('ShowImages', full_screen=True) as ec:
    #write_hdf5(op.splitext(ec.data_fname)[0] + '_trials.hdf5',
    #           dict(imorder_shuf=imorder_shuf,
    #                imtype_shuf=imtype_shuf))
    fr = 1/ec.estimate_screen_fs()  # Estimate frame rate
    adj = fr/2  # Adjustment factor for accurate flip
    # Wait to fill the screen
    ec.set_visible(False)
    # Set the background color to gray
    ec.set_background_color(bgcolor)

    # load up the image stack. The images in img_buffer are in the sequential 
    # non-shuffled order
    img = []
    for im in imagelist:
        img_buffer = np.array(Image.open(im), np.uint8) / 255.
        if img_buffer.ndim == 2:
            img_buffer = np.tile(img_buffer[:, :, np.newaxis], [1, 1, 3])
        img.append(visual.RawImage(ec, img_buffer, scale=s))
        ec.check_force_quit()

    # make a blank image
    blank = visual.RawImage(ec, np.tile(bgcolor[0], np.multiply([s, s, 1], img_buffer.shape)))
    bright = visual.RawImage(ec, np.tile([1.], np.multiply([s, s, 1], img_buffer.shape)))
    # Calculate stimulus size
    d_pix = -np.diff(ec._convert_units([[3., 0.], [3., 0.]], 'deg', 'pix'), axis=-1)

    # do the drawing, then flip
    ec.set_visible(True)
    frametimes = []
    buttons = []
    ec.listen_presses()
    last_flip = -1

    # Create a fixation dot
    fix = visual.FixationDot(ec, colors=('k', 'k'))
    fix.set_radius(4, 0, 'pix')
    fix.draw()

    # Display instruction (7 seconds).
    # They will be different depending on the run number
    if int(ec.session) % 2:
        t = visual.Text(ec,text='Button press when the dot turns red - Ignore images',pos=[0,.1],font_size=40,color='k')
    else:
        t = visual.Text(ec,text='Button press for fake word',pos=[0,.1],font_size=40,color='k') 
    t.draw()
    ec.flip()
    ec.wait_secs(1.0)

    # Show images
    frame = 0
    trigger = 0
    # The iterable 'trial' randomizes the order of everything since it is
    # drawn from imorder_shuf
    t0 = time.time()
    while frame < 2400:
#        assert len(dcolor) == len(ISI)
        # Insert a blank after waiting the desired image duration
        # Change the fixation dot color
#        fix.set_colors(colors=(dcolor2[trial], dcolor2[trial]))
#        blank.draw(), fix.draw()
#        ec.write_data_line('dotcolorFix', dcolor2[trial])

#        last_flip = ec.flip(last_flip+imduration/2-adj)
        # Draw a fixation dot of a random color
#        fix.set_colors(colors=(dcolor[trial], dcolor[trial]))
        # Stamp a trigger when the image goes on
#        trig = imtype[trial] if not testing else test_trig
#        ec.call_on_next_flip(ec.stamp_triggers, trig, check='int4')
        
#        bright.draw()
#        ec.stamp_triggers(1, check='int4')
#        t1 = ec.flip()
        # Draw the image

        if np.mod(frame,40) == 0:
            bright.draw()
            trigger = 1
#            ec.stamp_triggers(1,check='int4')
        else:
            trigger = 0
            blank.draw()
#            ec.stamp_triggers(2,check='int4')
#            count = (count + 1) % 2
#            blank.draw()
#        else:
#            img[trial].draw()
#        fix.draw()

        # Mark the log file of the trial type
#        ec.write_data_line('imnumber', imnumber[trial])
#        ec.write_data_line('imtype', imtype[trial])
#        ec.write_data_line('dotcolorIm', dcolor[trial])

        # The image is flipped ISI milliseconds after the blank
        if trigger:
            ec.stamp_triggers(1,check='int4',wait_for_last=False)
        last_flip = ec.flip()
        
#        last_flip - t1
#        ec.flip()
       # Change the fixation dot color mid trial
#        fix.set_colors(colors=(dcolor3[trial], dcolor3[trial]))
#        img[trial].draw(), fix.draw()
#        ec.write_data_line('dotcolorFix', dcolor2[trial])
#        last_flip = ec.flip(last_flip+imduration/2-adj)
        
        ec.get_presses()
        ec.listen_presses()
        frametimes.append(last_flip)
        ec.check_force_quit()
        
        frame += 1
print "\n\n Elasped time: %0.4f secs" % (time.time()-t0)
    # Now the experiment is over and we show 5 seconds of blank
#    blank.draw(), fix.draw()
#    ec.flip()
#    ec.wait_secs(5.0)
#    pressed = ec.get_presses()  # relative_to=0.0