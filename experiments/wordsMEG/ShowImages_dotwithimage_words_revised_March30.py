"""
==========================
Display images for MEG
==========================
"""
import numpy as np
from expyfun import visual, ExperimentController
from expyfun.io import write_hdf5

from PIL import Image
import os
from os import path as op
import glob

# background color
testing = False
test_trig = [15]
if testing:
    bgcolor = [0., 0., 0., 1.]
else:
    bgcolor = [127/255., 127/255., 127/255., 1]
# These should be inputs
#basedir = '/home/jyeatman/projects/MEG/images/'
basedir = os.path.join('C:\\Users\\neuromag\\Desktop\\jason\\wordStim')
#basedir = os.path.join('/home/jyeatman/git/wmdevo/megtools/stim/wordStim')
#imagedirs = ['faces_localizer', 'limb_localizer', 'objects_localizer', 'place_localizer']
imagedirs = ['stim001', 'stim007', 'stim008', 'stim009', 'stim014', 'stim015', 'stim016']
imagedirs = ['word_c254_p20', 'word_c254_p50','word_c137_p20','word_c141_p20',
             'word_c254_p80', 'bigram_c254_p20', 'bigram_c254_p50',
             'bigram_c137_p20','bigram_c141_p20']

nimages = [20, 20, 20, 20, 20, 5, 5, 5, 5]  # number of images in each category
if len(nimages) == 1: nimages = np.repeat(nimages,len(imagedirs))
isis = np.arange(.25,.44,.02)  # ISIs to be used. Must divide evenly into nimages
isis = [.25, .3, .35, .4, .45, .5, .55, .6, .65, .7]
imduration = 1
s = .7  # Image scale

# Create a vector of ISIs in a random order. One ISI for each image
rng = np.random.RandomState()
ISI = np.repeat(isis, sum(nimages)/len(isis))
rng.shuffle(ISI)

# Create a vector that will denote times to change the color of the fixation
# dot. This vector denotes the point within the ISI to change the dot color.
# I bounded this at .2 and .8 because I don't want dot changes occuring
# immediately before or after image presentation
dotchange = rng.uniform(.2, .8, len(ISI))
# Creat a vector of dot colors for each ISI
c = ['r', 'g', 'b', 'y', 'c']
dcolor = np.tile(c, len(ISI)/len(c))
dcolor2 = np.tile(c, len(ISI)/len(c))
rng = np.random.RandomState()
rng.shuffle(dcolor)
rng = np.random.RandomState()
rng.shuffle(dcolor2)
# Create a vector denoting when to display each image
imorder = np.arange(sum(nimages))
# Create a vector marking the category of each image
imtype = []
for i in range(1, len(imagedirs)+1):
    imtype.extend(np.tile(i, nimages[i-1]))

# Shuffle the image order
ind_shuf = range(len(imtype))
rng = np.random.RandomState()
rng.shuffle(ind_shuf)
imorder_shuf = []
imtype_shuf = []
for i in ind_shuf:
    imorder_shuf.append(imorder[i])
    imtype_shuf.append(imtype[i])

# Build the path structure to each image in each image directory. Each image
# category is an entry into the list
imagelist = []
imnumber = []
c = -1
rng = np.random.RandomState()
for imname in imagedirs:
    c = c+1
    # Temporary variable with image names in order
    tmp = sorted(glob.glob(os.path.join(basedir, imname, '*')))
    # Randomly grab nimages from the list
    n = rng.randint(0, len(tmp), nimages[c])
    tmp2 = []
    for i in n: tmp2.append(tmp[i])  # This line is slow... Python must have a faster way to index
    # Add the random image list to an entry in imagelist
    imagelist.extend(tmp2)
    # record the image number
    imnumber.extend(n)
    assert len(imagelist[-1]) > 0

# Start instance of the experiment controller
#with ExperimentController('ReadingExperiment') as ec:
#with ExperimentController('ShowImages', full_screen=False, window_size=[800, 800]) as ec:
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

    # load up the image stack
    img = []
    for im in imagelist:
        #img_buffer = np.array(Image.open(os.path.join(imagedir,im)))/255.
        #img.append(visual.RawImage(ec, img_buffer, scale=1.))
        img_buffer = np.array(Image.open(im), np.uint8) / 255.
        if img_buffer.ndim == 2:
            img_buffer = np.tile(img_buffer[:, :, np.newaxis], [1, 1, 3])
        img.append(visual.RawImage(ec, img_buffer, scale=s))
        ec.check_force_quit()

    # make a blank image
    blank = visual.RawImage(ec, np.tile(bgcolor[0], np.multiply([s, s, 1],img_buffer.shape)))
    bright = visual.RawImage(ec, np.tile([1.], np.multiply([s, s, 1],img_buffer.shape)))
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
    fix.draw()
    # Instructions
    if int(ec.session) %2:
        t = visual.Text(ec,text='Button press when the dot turns red - Ignore images',pos=[0,.1],font_size=40,color='k')
    else:
        t = visual.Text(ec,text='Button press for fake word',pos=[0,.1],font_size=40,color='k')
        
    t.draw()
    ec.flip()
    ec.wait_secs(7.0)

    # Show images
    count = 0
    for trial in imorder_shuf:
        # Insert a blank after waiting the desired image duration
        fix.set_colors(colors=(dcolor2[trial], dcolor2[trial]))
        blank.draw(), fix.draw()
        ec.write_data_line('dotcolorFix', dcolor2[trial])

        last_flip = ec.flip(last_flip+imduration)
        # Draw a fixation dot of a random color a random portion of the time
        # through the ISI
        fix.set_colors(colors=(dcolor[trial], dcolor[trial]))
        # Stamp a trigger when the image goes on
        trig = imtype[trial] if not testing else test_trig
        ec.call_on_next_flip(ec.stamp_triggers, trig, check='int4')
        # Draw the image
        if testing:
            if count == 0:
                bright.draw()
            else:
                blank.draw()
            count = (count + 1) % 2
        else:
            img[trial].draw()
        fix.draw()
        # Mark the log file of the trial type
        ec.write_data_line('imnumber', imnumber[trial])
        ec.write_data_line('imtype', imtype[trial])
        ec.write_data_line('dotcolorIm', dcolor[trial])
        # The image is flipped ISI milliseconds after the blank
        last_flip = ec.flip(last_flip+ISI[trial]-adj)
        ec.get_presses()
        ec.listen_presses()
        frametimes.append(last_flip)
        ec.check_force_quit()
        
    blank.draw(), fix.draw()    
    ec.wait_secs(5.0)
    pressed = ec.get_presses()  # relative_to=0.0