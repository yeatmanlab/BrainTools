#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:24:51 2018

@author: sjjoo
"""

#%%
figureDir = '%s/figures/iowa' % raw_dir
os.chdir(figureDir)

""" Left STG """
nReps = 3000
boot_pVal = 0.05

M = np.mean(np.mean(X11[aud_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[aud_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[aud_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[aud_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[aud_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[aud_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[aud_vertices_l,:,:,:],axis=0)

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)
C = np.mean(X11[aud_vertices_l,:,:,0],axis=0) - np.mean(X11[aud_vertices_l,:,:,3],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('STG_dot_all.png',dpi=300)

plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)

plt.savefig('STG_dot_good.png',dpi=300)

plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)

plt.savefig('STG_dot_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('STG_dot_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('STG_dot_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('STG_dot_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 0, 1, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 0, 1, 3, all_subject, boot_pVal)

plt.savefig('STG_dot_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 0, 1, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 0, 1, 3, good_readers, boot_pVal)

plt.savefig('STG_dot_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 0, 1, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 0, 1, 3, poor_readers, boot_pVal)

plt.savefig('STG_dot_poor_allstim.png',dpi=300)
 
###############################################################################

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)
C = np.mean(X11[aud_vertices_l,:,:,5],axis=0) - np.mean(X11[aud_vertices_l,:,:,8],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('STG_lexical_all.png',dpi=300)

plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)

plt.savefig('STG_lexical_good.png',dpi=300)

plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)

plt.savefig('STG_lexical_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('STG_lexical_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('STG_lexical_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('STG_lexical_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 5,6,8, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 5,6,8, all_subject, boot_pVal)

plt.savefig('STG_lexical_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 5,6,8, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 5,6,8, good_readers, boot_pVal)

plt.savefig('STG_lexical_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 5,6,8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 5,6,8, poor_readers, boot_pVal)

plt.savefig('STG_lexical_poor_allstim.png',dpi=300)
 
###############################################################################

#%%
""" Left Broca """

M = np.mean(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[broca_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[broca_vertices_l,:,:,:],axis=0)

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)
C = np.mean(X11[broca_vertices_l,:,:,0],axis=0) - np.mean(X11[broca_vertices_l,:,:,3],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('Broca_dot_all.png',dpi=300)

plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)

plt.savefig('Broca_dot_good.png',dpi=300)

plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)

plt.savefig('Broca_dot_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Broca_dot_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Broca_dot_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Broca_dot_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 0, 1, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 0, 1, 3, all_subject, boot_pVal)

plt.savefig('Broca_dot_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 0, 1, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 0, 1, 3, good_readers, boot_pVal)

plt.savefig('Broca_dot_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 0, 1, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 0, 1, 3, poor_readers, boot_pVal)

plt.savefig('Broca_dot_poor_allstim.png',dpi=300)
 
###############################################################################

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)
C = np.mean(X11[broca_vertices_l,:,:,5],axis=0) - np.mean(X11[broca_vertices_l,:,:,8],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('Broca_lexical_all.png',dpi=300)

plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)

plt.savefig('Broca_lexical_good.png',dpi=300)

plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)

plt.savefig('Broca_lexical_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Broca_lexical_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Broca_lexical_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Broca_lexical_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 5,6,8, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 5,6,8, all_subject, boot_pVal)

plt.savefig('Broca_lexical_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 5,6,8, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 5,6,8, good_readers, boot_pVal)

plt.savefig('Broca_lexical_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 5,6,8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 5,6,8, poor_readers, boot_pVal)

plt.savefig('Broca_lexical_poor_allstim.png',dpi=300)
 
###############################################################################

#%%
""" Left TPJ """

M = np.mean(np.mean(X11[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[tpj_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[tpj_vertices_l,:,:,:],axis=0)

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)
C = np.mean(X11[tpj_vertices_l,:,:,0],axis=0) - np.mean(X11[tpj_vertices_l,:,:,3],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('TPJ_dot_all.png',dpi=300)

plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)

plt.savefig('TPJ_dot_good.png',dpi=300)

plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)

plt.savefig('TPJ_dot_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('TPJ_dot_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('TPJ_dot_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('TPJ_dot_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 0, 1, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 0, 1, 3, all_subject, boot_pVal)

plt.savefig('TPJ_dot_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 0, 1, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 0, 1, 3, good_readers, boot_pVal)

plt.savefig('TPJ_dot_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 0, 1, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 0, 1, 3, poor_readers, boot_pVal)

plt.savefig('TPJ_dot_poor_allstim.png',dpi=300)
 
###############################################################################

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)
C = np.mean(X11[tpj_vertices_l,:,:,5],axis=0) - np.mean(X11[tpj_vertices_l,:,:,8],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('TPJ_lexical_all.png',dpi=300)

plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)

plt.savefig('TPJ_lexical_good.png',dpi=300)

plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)

plt.savefig('TPJ_lexical_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('TPJ_lexical_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('TPJ_lexical_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('TPJ_lexical_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 5,6,8, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 5,6,8, all_subject, boot_pVal)

plt.savefig('TPJ_lexical_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 5,6,8, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 5,6,8, good_readers, boot_pVal)

plt.savefig('TPJ_lexical_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 5,6,8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 5,6,8, poor_readers, boot_pVal)

plt.savefig('TPJ_lexical_poor_allstim.png',dpi=300)
 
###############################################################################

#%%
""" Left Motor """

M = np.mean(np.mean(X11[motor_vertices_l,:,:,:],axis=0),axis=1)
errM = np.std(np.mean(X11[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(all_subject))

temp1 = X11[:,:,good_readers,:]
M1 = np.mean(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1)
errM1 = np.std(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(good_readers))
del temp1

temp1 = X11[:,:,poor_readers,:]
M2 = np.mean(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1)
errM2 = np.std(np.mean(temp1[motor_vertices_l,:,:,:],axis=0),axis=1) / np.sqrt(len(poor_readers))
del temp1

# For calculating p-values
X = np.mean(X11[motor_vertices_l,:,:,:],axis=0)

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 0, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 0, 3, all_subject, boot_pVal)
C = np.mean(X11[motor_vertices_l,:,:,0],axis=0) - np.mean(X11[motor_vertices_l,:,:,3],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('Motor_dot_all.png',dpi=300)

plotit2(times, M1, errM1, 0, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 0, 3, good_readers, boot_pVal)

plt.savefig('Motor_dot_good.png',dpi=300)

plotit2(times, M2, errM2, 0, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 0, 3, poor_readers, boot_pVal)

plt.savefig('Motor_dot_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Motor_dot_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Motor_dot_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=3, color_num = 12)

plt.savefig('Motor_dot_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 0, 1, 3, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 0, 1, 3, all_subject, boot_pVal)

plt.savefig('Motor_dot_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 0, 1, 3, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 0, 1, 3, good_readers, boot_pVal)

plt.savefig('Motor_dot_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 0, 1, 3, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 0, 1, 3, poor_readers, boot_pVal)

plt.savefig('Motor_dot_poor_allstim.png',dpi=300)
 
###############################################################################

#####################  Word vs. noise  ###################################### 
plotit2(times, M, errM, 5, 8, yMin=0, yMax=2.5, subject = 'all')
plotsig2(times,nReps,X, 5, 8, all_subject, boot_pVal)
C = np.mean(X11[motor_vertices_l,:,:,5],axis=0) - np.mean(X11[motor_vertices_l,:,:,8],axis=0)
plotcorr3(times, C, twre_index)
#plotcorr3(times, C, brs, 2.4)
#plotcorr3(times, C, age, 2.5)

plt.savefig('Motor_lexical_all.png',dpi=300)

plotit2(times, M1, errM1, 5, 8, yMin=0, yMax=2.5, subject = 'typical')
plotsig2(times,nReps,X, 5, 8, good_readers, boot_pVal)

plt.savefig('Motor_lexical_good.png',dpi=300)

plotit2(times, M2, errM2, 5, 8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig2(times,nReps,X, 5, 8, poor_readers, boot_pVal)

plt.savefig('Motor_lexical_poor.png',dpi=300)

################## Noise only #################################################
plotit(times, M, errM, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Motor_lexical_all_noise.png',dpi=300)

plotit(times, M1, errM1, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Motor_lexical_good_noise.png',dpi=300)

plotit(times, M2, errM2, yMin=0, yMax=2.5, title = 'abc', task=8, color_num = 12)

plt.savefig('Motor_lexical_poor_noise.png',dpi=300)

###################   All stim   ##############################################
plotit3(times, M, errM, 5,6,8, yMin=0, yMax=2.5, subject = 'all')
plotsig3(times,nReps,X, 5,6,8, all_subject, boot_pVal)

plt.savefig('Motor_lexical_all_allstim.png',dpi=300)

plotit3(times, M1, errM1, 5,6,8, yMin=0, yMax=2.5, subject = 'typical')
plotsig3(times,nReps,X, 5,6,8, good_readers, boot_pVal)

plt.savefig('Motor_lexical_good_allstim.png',dpi=300)

plotit3(times, M2, errM2, 5,6,8, yMin=0, yMax=2.5, subject = 'struggling')
plotsig3(times,nReps,X, 5,6,8, poor_readers, boot_pVal)

plt.savefig('Motor_lexical_poor_allstim.png',dpi=300)
 
###############################################################################