function X = nf_makeDesignMatrixFromParfile(sessDir, parfile, TR)
 
home = pwd;
parPath = strcat(sessDir, '/Stimuli', '/parfiles');
cd(parPath)
[onsets, conds, labels] = readParFile(parfile);
 
%create matrix of zeros with dimensions #TRs x nConditions
cd(sessDir) 
nTRs = 98;
nConds = 3;

X = zeros(nTRs, nConds);
 
%fill in design matrix with 1's indicating event onset and type (e.g. event
%at tr = 6 that was condition 4 would be indicated by a 1 in row 6, col 4)
for i = 1:length(onsets)
    cur_tr = (onsets(i)/TR) + 1;
    cur_cond = conds(i) + 1;
    X(round(cur_tr), cur_cond) = 1; %% things are weird because TRs are not an integer -ECK. 
end
 
%check;
figure; imagesc(X);
colormap(gray); colorbar;
xlabel('Conditions');
ylabel('Time points');

cd(home)