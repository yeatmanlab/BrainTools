function X1 = nf_makeDesignMatrix_jdy(sessDir, parfile, TR)


home = pwd;
parPath = strcat(sessDir, '/Stimuli', '/parfiles');
cd(parPath)
[onsets, conds, labels] = readParFile(parfile);

% Allocate space for design matrix in seconds
nsecs = 196;
nConds = 3;
X = zeros(nsecs, nConds);
cols = conds+1; % column numbers corresponding to conditions
endblank = 6; % Number of blank seconds at the end
onsets(end+1) = onsets(end) + endblank; % Add in blanks
TRlength = 2; % TR length in seconds
nTRs = 98; % Number of TRs for the whole experiment
extraSecs = 0; % CHECK THIS!! THERE ARE EXTRA SECS AT THE END
for ii = 1:(length(onsets) - 1) % Onsets-1 because of last blank offset being coded
   rows = (onsets(ii) +1):onsets(ii+1);
   X(rows, cols(ii)) = 1;
end

% CHECK THIS!
% X(end: end+extraSecs,1) = 1;

% Visualize design matrix
imagesc(X);xlabel('conditions');ylabel('seconds')
set(gca,'xtick',1:nConds,'xticklabels',unique(labels))

% Interpolate the design matrix at the granularity of the TR length which
% is 2.2s in our case
newX = size(X,1)/TRlength;

X1 = imresize(X,[newX,nConds],'bilinear');
figure;colormap('gray')
subplot(1,2,1)
imagesc(X);xlabel('conditions');ylabel('seconds')
set(gca,'xtick',1:nConds,'xticklabels',unique(labels));title('Design in Seconds');
subplot(1,2,2)
imagesc(X1);xlabel('conditions');ylabel('seconds')
set(gca,'xtick',1:nConds,'xticklabels',unique(labels));title('Design in TRs');

%check;
figure; imagesc(X);
colormap(gray); colorbar;
xlabel('Conditions');
ylabel('Time points');

cd(home)