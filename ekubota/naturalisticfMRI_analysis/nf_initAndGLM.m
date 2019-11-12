function nf_initAndGLM(subnum,datadir)

% initializes vistasoft session and runs GLM
nf_initialize_vista(subnum,datadir)
nf_fmri_glm(datadir)
clx
