s=3:4;
sge=0;
splitShells = 0;
concat = 0;
dwiData{1} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB1_20130630/20130630_1437/4_1_DWI_2mm_108dir_b1000_2000/4957_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB1_20130716/20130716_1606/4_1_DWI_2mm_108dir_b1000_2000/5059_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB1_20130730/20130730_1004/4_1_DWI_2mm_108dir_b1000_2000/5160_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB1_20130818/20130818_1602/4_1_DWI_2mm_108dir_b1000_2000/5307_4_1.nii.gz'};

dwiData{2} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB2_20130628/20130628_1812/11_1_DWI_2mm_108dir_b1000_2000/4954_11_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB2_20130715/20130715_1805/4_1_DWI_2mm_108dir_b1000_2000/5053_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB2_20130729/20130729_1738/5_1_DWI_2mm_108dir_b1000_2000/5156_5_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB2_20130808/20130808_1746/11_1_DWI_2mm_108dir_b1000_2000/5239_11_1.nii.gz'}

dwiData{3} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB4_20130807/20130807_1120/raw/5_1_DWI_2mm_108dir_b1000_2000/5225_5_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB4_20130906/20130906_1538/raw/6_1_DWI_2mm_108dir_b1000_2000/5425_6_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB4_20130927/20130927_1512/raw/5_1_DWI_2mm_108dir_b1000_2000/5532_5_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB4_20131120/20131120_1054/raw/4_1_DWI_2mm_108dir_b1000_2000/5852_4_1.nii.gz'}

dwiData{4} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB11_20130709/20130709_1008/5_1_DWI_2mm_108dir_b1000_2000/5002_5_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB11_20130731/20130731_1805/4_1_DWI_2mm_108dir_b1000_2000/5179_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB11_20130819/20130819_1012/4_1_DWI_2mm_108dir_b1000_2000/5311_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB11_20130909/20130909_1617/6_1_DWI_2mm_108dir_b1000_2000/5433_6_1.nii.gz'}

dwiData{5} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB17_20130728/20130728_1345/4_1_DWI_2mm_108dir_b1000_2000/5144_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB17_20130813/20130813_1910/27_1_DWI_2mm_108dir_b1000_2000/5275_27_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB17_20130827/20130827_1835/5_1_DWI_2mm_108dir_b1000_2000/5374_5_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB17_20130910/20130910_1907/4_1_DWI_2mm_108dir_b1000_2000/5444_4_1.nii.gz'};

dwiData{6} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB18_20130805/20130805_1025/4_1_DWI_2mm_108dir_b1000_2000/5205_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB18_20130820/20130820_1438/4_1_DWI_2mm_108dir_b1000_2000/5328_4_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB18_20130904/20130904_1540/6_7_Concat_DWI_2mm_108dir/Concat_6_7.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB18_20131016/20131016_1443/5_1_DWI_2mm_108dir_b1000_2000/5652_5_1.nii.gz'}

bravo{1} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB1_20130630/20130630_1437/17_1_Ax_FSPGR_BRAVO/4957_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB1_20130716/20130716_1606/17_1_Ax_FSPGR_BRAVO/5059_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB1_20130730/20130730_1004/17_1_Ax_FSPGR_BRAVO/5160_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB1_20130818/20130818_1602/16_1_Ax_FSPGR_BRAVO/5307_16_1.nii.gz'};

bravo{2} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB2_20130628/20130628_1812/3_1_Ax_FSPGR_BRAVO/4954_3_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB2_20130715/20130715_1805/15_1_Ax_FSPGR_BRAVO/5053_15_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB2_20130729/20130729_1738/17_1_Ax_FSPGR_BRAVO/5156_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB2_20130808/20130808_1746/3_1_Ax_FSPGR_BRAVO/5239_3_1.nii.gz'};

bravo{3}={'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB4_20130927/20130927_1512/17_1_Ax_FSPGR_BRAVO/5532_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB4_20131120/20131120_1054/raw/16_1_Ax_FSPGR_BRAVO/5852_16_1.nii.gz'};

bravo{4} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB11_20130731/20130731_1805/17_1_Ax_FSPGR_BRAVO/5179_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB11_20130819/20130819_1012/20_1_Ax_FSPGR_BRAVO/5311_20_1.nii.gz'};


bravo{5} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB17_20130910/20130910_1907/16_1_Ax_FSPGR_BRAVO/5444_16_1.nii.gz'};

bravo{6} = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB18_20130805/20130805_1025/14_1_Ax_FSPGR_BRAVO/5205_14_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB18_20130820/20130820_1438/17_1_Ax_FSPGR_BRAVO/5328_17_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB18_20130904/20130904_1540/19_1_Ax_FSPGR_BRAVO/5406_19_1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB18_20131016/20131016_1443/15_1_Ax_FSPGR_BRAVO/5652_15_1.nii.gz'};
t1Path = {'/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB1/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB2/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB4/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB11/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB17/t1.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/anatomy/LB18/t1.nii.gz'};

catDir = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB1'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB2'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB4'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB11'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB17'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/all/LB18'}
%% Split DWI shells
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

%% concatenate into one huge volume
if concat == 1
    for ii = s
        dwiIm = [];
        bvals = [];
        bvecs = [];
        for jj = 1:4
            if jj == 1
                dwiIm = readFileNifti(b2000{ii}{jj});
                bvals = dlmread(bvals2{ii}{jj});
                bvecs = dlmread(bvecs2{ii}{jj});
                
            else
                tmp = readFileNifti(b2000{ii}{jj});
                dwiIm.data = cat(4,dwiIm.data,tmp.data);
                tmp = dlmread(bvals2{ii}{jj});
                bvals = horzcat(bvals,tmp);
                tmp = dlmread(bvecs2{ii}{jj});
                bvecs = horzcat(bvecs,tmp);
            end
        end
        dwiIm.dim = size(dwiIm.data);
        dwiCat{ii} = fullfile(catDir{ii},'raw','b2000cat.nii.gz');
        dwiIm.fname = dwiCat{ii};
        writeFileNifti(dwiIm);
        dlmwrite(fullfile(catDir{ii},'raw','b2000cat.bvec'),bvecs);
        dlmwrite(fullfile(catDir{ii},'raw','b2000cat.bval'),bvals);
        clear dwiIm
    end
    
    %% process dwi data
    params = dtiInitParams;
    params.fitMethod = 'rt';params.noiseCalcMethod = 'b0';params.eddyCorrect=0;
    params.clobber = 1;
    params.dt6BaseName = fullfile(catDir{ii},'dtcat');
    for ii = s
        dtiInit(dwiCat{ii},t1Path{ii},params);
    end
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