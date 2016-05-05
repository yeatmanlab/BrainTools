
def nlr_organizeMEG_mnefun(raw_dir=None, out_dir=None, subs=None, sess=None):
# TODO: WE DO NOT YET DEAL WITH THE FACT THAT THERE MAY BE MISSING RUNS FOR
# SOME SUBJECTS. IN THIS CASE, WE RENAME THE FILES IMPROPERLY. WE NEED TO
# RECORD THE RUN NUMBERS BEFORE COPYING THE FILES
    import os
    import shutil
    import glob
    import numpy as np

    # define subjects and sessions
    if raw_dir == None:
        raw_dir = '/mnt/diskArray/projects/MEG/nlr/raw'
    if subs == None:
        # subs = ['202_dd', '203_am']
        subs = glob.glob(os.path.join(raw_dir, '1*_*'))
        subs.extend(glob.glob(os.path.join(raw_dir, '2*_*')))
        # remove base path
        for n, s in enumerate(subs):
            subs[n] = os.path.basename(subs[n])
    if sess == None:
        # sess = [['150827', '150919', '151013', '151103'],
        #        ['150831', '150922', '151009', '151029']]
        sess = []
        for n, s in enumerate(subs):
            sess.append(glob.glob(os.path.join(raw_dir, s, '*')))
            for ii, i in enumerate(sess[n]):
                sess[n][ii] = os.path.basename(sess[n][ii])
    if out_dir == None:
        out_dir = '/mnt/diskArray/scratch/NLR_MEG'
    # We will make a list of subjects where each subject+session combo is an
    # entry in the list
    subjects = []
    # Loop over the subjects
    for n, s in enumerate(subs):
        # Loop over the sessions for subject n
        for ss in sess[n]:
            # Make a new directory for the subject (s) session (ss) combo
            sdir = os.path.join(out_dir, s+ss)          
            # List all directory fif files in raw directory
            fiflist = glob.glob(os.path.join(raw_dir, s, ss, '*_raw.fif'))
            # Sort the list
            fiflist.sort()
            if len(fiflist) >= 8:
                # make a directory for the raw files to be saved
                rfifdir = os.path.join(sdir, 'raw_fif')
                subjects.append(s+ss)  # append subjects with the subjectsession
                # Make directories if they do not exist
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                if not os.path.isdir(rfifdir):
                    os.mkdir(rfifdir)
                # Loop over the list of raw fif files
                for fnum, fname in enumerate(fiflist):
                    if not runnumber(fname):
                        # Skip this file if it doesn't match the naming scheme
                        # Defined in runnumber
                        print('Skipping ' + fname)
                    else:
                        fifdest = os.path.join(rfifdir, s + ss + '_' + runnumber(fname))
                        # WE ONLY COPY THE FILE OVER IF IT DOESN'T EXIST!
                        if not os.path.isfile(fifdest):
                            # Copy this fif file to the destination and rename it to
                            # have the subject name. This is the assumption in mnefun
                            shutil.copy(os.path.join(raw_dir, s, ss, fname), fifdest)
                        else:
                            print(fifdest + ' already exists: Skpping')
                        # Make the prebad file that mnefun wants
                        prebad = os.path.join(rfifdir, s+ss+'_prebad.txt')
                        if not os.path.isfile(prebad):
                            open(prebad, 'w').close()

    # Return a list of all the subjects
    return(subjects)

def runnumber(fifname):
    fifend = []
    runnums =['1_raw.fif', '2_raw.fif', '3_raw.fif', '4_raw.fif',
    '5_raw.fif', '6_raw.fif', '7_raw.fif', '8_raw.fif']
    for rn in runnums:
        if fifname.find(rn) > -1:
            fifend = rn
    return fifend            
            
 