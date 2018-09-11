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

# background color
testing = False
test_trig = [15]
if testing:
    bgcolor = [0., 0., 0., 1.]
else:
    bgcolor = [127/255., 127/255., 127/255., 1]

# Paths to images
#basedir = '/home/jyeatman/projects/MEG/images/'
basedir = os.path.join('C:\\Users\\neuromag\\Desktop\\jason\\wordStim')
if not os.path.isdir(basedir):
    basedir = os.path.join('/mnt/diskArray/projects/MEG/wordStim')

""" Words, False fonts (Korean), Faces, Objects """
imagedirs = ['word_c254_p0', 'word_c254_p50', 'word_c254_p80', 'bigram_c254_p20']

nimages = [30, 30, 30, 30]  # number of images in each category
if len(nimages) == 1: # Does nothing....
    nimages = np.repeat(nimages, len(imagedirs))
n_totalimages = sum(nimages)
# ISIs to be used. Must divide evenly into nimages
isis = np.arange(1., 1.51, 0.1) #np.arange(.62, .84, .02)
imduration = 0.8  # Image duration 800 ms
s = .5  # Image scale

# Create a vector of ISIs in a random order. One ISI for each image
rng = np.random.RandomState(int(time.time()))
ISI = np.tile(isis, int(np.ceil(n_totalimages/len(isis)))+1)
rng.shuffle(ISI)
ISI = ISI[:(n_totalimages)]

total_time = sum(ISI) + sum(nimages)*imduration # in second

n_flickers = int(total_time*2)+1 # every 500 ms
n_target = int(0.2*n_flickers)

fix_seq = np.zeros(n_flickers) 
fix_seq[:n_target] = 1
rng.shuffle(fix_seq)
for i  in range(0,len(fix_seq)-2):
    if (fix_seq[i] + fix_seq[i+1]) == 2:
        fix_seq[i+1] = 0
        if fix_seq[i+3] == 0:
            fix_seq[i+3] = 1
        elif fix_seq[i+4] == 0:
            fix_seq[i+4] = 1
            
# Creat a vector of dot colors for each ISI
c = ['g', 'b', 'y', 'c']
k = 0
m = 0
fix_color = []
for i in range(0,len(fix_seq)): 
    if fix_seq[i] == 1:
        fix_color.append('r')
    else:
        fix_color.append(c[k])
        k += 1
        k = np.mod(k,4)
    if k == 0:
        rng.shuffle(c)
        while fix_color[i] == c[0]:
            rng.shuffle(c)

# Create a vector marking the category of each image
imtype = []
for i in range(1, len(imagedirs)+1):
    imtype.extend(np.tile(i, nimages[i-1]))
    rng.shuffle(imtype)

# Build the path structure to each image in each image directory. Each image
# category is an entry into the list. The categories are in sequential order 
# matching imorder, but the images within each category are random
templist = []
tempnumber = []
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
    templist.extend(tmp2)
    # record the image number
    tempnumber.extend(n)
    assert len(templist[-1]) > 0
temp_list = np.arange(0,len(templist))
rng.shuffle(temp_list)

imagelist = []
imnumber = []
for i in temp_list:
    imagelist.append(templist[i])
    imnumber.append(tempnumber[i])
    
# Start instance of the experiment controller
with ExperimentController('ShowImages', full_screen=True, version='dev') as ec:
    #write_hdf5(op.splitext(ec.data_fname)[0] + '_trials.hdf5',
    #           dict(imorder_shuf=imorder_shuf,
    #                imtype_shuf=imtype_shuf))
    fr = 1/ec.estimate_screen_fs()  # Estimate frame rate
    realRR = ec.estimate_screen_fs()
    realRR = round(realRR)
    adj = fr/2  # Adjustment factor for accurate flip
    # Wait to fill the screen
    ec.set_visible(False)
    # Set the background color to gray
    ec.set_background_color(bgcolor)

    n_frames = round(total_time * realRR)
    img_frames = round(imduration*realRR)
    jitter = np.arange(0,realRR*0.2) # 0~200 ms jitter
    
    temp_flicker = np.arange(0,n_frames,int(realRR/2)) # Get temp_flicker frames: every .5 s
    delay = []
    for i in np.arange(0,len(temp_flicker)):
        rng.shuffle(jitter)
        delay.append(jitter[0])
    frame_flicker = temp_flicker + delay # 
    frame_img = [0]
    for i in np.arange(0,len(ISI)):
        frame_img.append(frame_img[i] + img_frames + int(ISI[i]*realRR))
    frame_img = frame_img[:-1]
    
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
    ec.wait_secs(5.0)

    # Show images
    count = 0 # This is for testing...
    
    # Initial blank
    init_blanktime = 1.
    fix.set_colors(colors=('k', 'k'))
    blank.draw(), fix.draw()
    ec.write_data_line('dotcolorFix', 'k')
    last_flip = ec.flip()

    # The iterable 'trial' randomizes the order of everything since it is
    # drawn from imorder_shuf
    trial = 0
    frame = 0
    flicker = 0
    imageframe = []
    t0 = time.time()
    while frame < n_frames-1:
        if frame == frame_flicker[flicker]:
            fix.set_colors(colors=(fix_color[flicker],fix_color[flicker]))
            ec.write_data_line('dotcolorFix', fix_color[flicker])
            if flicker < len(frame_flicker)-2:
                flicker += 1
                
        if frame >= frame_img[trial] and frame < frame_img[trial] + img_frames:
            if frame == frame_img[trial]:
                ec.write_data_line('imnumber', imnumber[trial])
                ec.write_data_line('imtype', imtype[trial])
                trig = imtype[trial]
                ec.stamp_triggers(trig, check='int4', wait_for_last = False)
                
            fix.set_colors(colors=(fix_color[flicker],fix_color[flicker]))
            
            img[trial].draw()
            
            imageframe.append(frame)
            if frame == frame_img[trial] + img_frames - 1:
                if trial < len(imnumber)-2:
                    trial += 1
        else:
            fix.set_colors(colors=(fix_color[flicker],fix_color[flicker]))
            blank.draw()
        
        fix.draw()
        
        last_flip = ec.flip()
        frame += 1
        ec.get_presses()
        frametimes.append(last_flip)
        ec.check_force_quit()

    # Now the experiment is over and we show 5 seconds of blank
    print "\n\n Elasped time: %0.2d mins %0.2d secs" % divmod(time.time()-t0, 60)
    print "\n\n Targeted time: %0.2d mins %0.2d secs" % divmod(total_time, 60)
    blank.draw(), fix.draw()
    ec.flip()
    ec.wait_secs(5.0)
    pressed = ec.get_presses()  # relative_to=0.0
    
    