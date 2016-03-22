s=1:7;
sge=1;
splitShells = 0;
dwiData{1} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB7_20130712/20130712_1717/4_1_DWI_2mm_108dir_b1000_2000/5040_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB7_20130802/20130802_1028/4_1_DWI_2mm_108dir_b1000_2000/5191_4_1.nii.gz'}
dwiData{2} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB8_20130723/20130723_1304/4_1_DWI_2mm_108dir_b1000_2000/5110_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB8_20130813/20130813_1214/4_1_DWI_2mm_108dir_b1000_2000/5269_4_1.nii.gz'};
dwiData{3} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB9_20130711/20130711_1616/11_1_DWI_2mm_108dir_b1000_2000/5027_11_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB9_20130807/20130807_1547/4_1_DWI_2mm_108dir_b1000_2000/5229_4_1.nii.gz'};
dwiData{4} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB10_20130703/20130703_1148/4_1_DWI_2mm_108dir_b1000_2000/4977_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB10_20130717/20130717_1334/4_1_DWI_2mm_108dir_b1000_2000/5067_4_1.nii.gz'};
dwiData{5} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB12_20130717/20130717_1743/4_1_DWI_2mm_108dir_b1000_2000/5070_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB12_20130806/20130806_1315/5_1_DWI_2mm_108dir_b1000_2000/5216_5_1.nii.gz'};
dwiData{6} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB15_20130729/20130729_1325/4_1_DWI_2mm_108dir_b1000_2000/5153_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB15_20131017/20131017_1635/4_1_DWI_2mm_108dir_b1000_2000/5660_4_1.nii.gz'};
dwiData{7} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB16_20130722/20130722_1744/4_1_DWI_2mm_108dir_b1000_2000/5105_4_1.nii.gz'};

bravo{1} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB7_20130712/20130712_1717/14_1_Ax_FSPGR_BRAVO/5040_14_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB7_20130802/20130802_1028/16_1_Ax_FSPGR_BRAVO/5191_16_1.nii.gz'};
bravo{2} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB8_20130723/20130723_1304/16_1_Ax_FSPGR_BRAVO/5110_16_1.nii.gz'
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB8_20130813/20130813_1214/15_1_Ax_FSPGR_BRAVO/5269_15_1.nii.gz'};
bravo{3} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB9_20130711/20130711_1616/3_1_Ax_FSPGR_BRAVO/5027_3_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB9_20130807/20130807_1547/16_1_Ax_FSPGR_BRAVO/5229_16_1.nii.gz'};
bravo{4} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB10_20130703/20130703_1148/17_2_Ax_FSPGR_BRAVO/4977_17_2.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB10_20130717/20130717_1334/16_1_Ax_FSPGR_BRAVO/5067_16_1.nii.gz'};
bravo{5} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB12_20130717/20130717_1743/16_1_Ax_FSPGR_BRAVO/5070_16_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB12_20130806/20130806_1315/16_1_Ax_FSPGR_BRAVO/5216_16_1.nii.gz'};
bravo{6} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB15_20130729/20130729_1325/16_1_Ax_FSPGR_BRAVO/5153_16_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB15_20131017/20131017_1635/20_1_Ax_FSPGR_BRAVO/5660_20_1.nii.gz'};
bravo{7} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB16_20130722/20130722_1744/13_1_Ax_FSPGR_BRAVO/5105_13_1.nii.gz'};

%% Acpc aligned t1
t1Path = {'/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB7/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB8/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB9/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB10/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB12/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB15/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB16/t1.nii.gz'};


%% Split DWI shells
if splitShells == 1
    for ii = s
        % Split the dwi data into multiple shells
        for jj = 1:length(dwiData{ii})
            bval = dir(fullfile(fileparts(dwiData{ii}{jj}),'*.bval'));
            bvec = dir(fullfile(fileparts(dwiData{ii}{jj}),'*.bvec'));
            bvalFile = fullfile(fileparts(dwiData{ii}{jj}),bval.name);
            bvecFile = fullfile(fileparts(dwiData{ii}{jj}),bvec.name);
            [b1000{ii}{jj}, b2000{ii}{jj}, bvals1{ii}{jj},bvals2{ii}{jj}, bvecs1{ii}{jj}, bvecs2{ii}{jj}]=...
                splitDWIShells(dwiData{ii}{jj},bvecFile,bvalFile);
        end
    end
    save /biac4/wandell/data/Lindamood_Bell/MRI/child/control/dataPaths b1000 b2000 bvals1 bvals2 bvecs1 bvecs2
else
    load /biac4/wandell/data/Lindamood_Bell/MRI/child/control/dataPaths
end
%% Process DWI data
params = dtiInitParams;
params.fitMethod = 'rt';params.noiseCalcMethod = 'b0';params.eddyCorrect=0;
params.clobber = 1;
for ii = s
    for jj = 1:length(dwiData{ii})
        % Fit b=2000 data
        if sge==0
            dtiInit(b2000{ii}{jj},t1Path{ii},params);
        elseif sge==1
            % Name the job
            jobname = sprintf('dt%d_%d',ii,round(rand*1000));
            % Process this subject on the grid
            sgerun2('dtiInit(b2000{ii}{jj},t1Path{ii},params);',jobname,1);
        else
            fprintf('\nSelect sge (1) or not (0)')
        end
        % Fit b=1000 data
        %dtiInit(b1000{ii}{jj},t1Path{ii},params);
    end
end