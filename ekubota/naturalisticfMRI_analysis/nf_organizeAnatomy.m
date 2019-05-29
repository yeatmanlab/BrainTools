function nf_organizeAnatomy(sublist)
% Takes in list of subjects and puts t1_class, t1_acpc, and ribbon file in 
% mrVista_Anat folder for fMRI analysis.

for fi = 1:length(sublist)
    nf_organizeIndividualAnatomy(sublist{fi})
end      
