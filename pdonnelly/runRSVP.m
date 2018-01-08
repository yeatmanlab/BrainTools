function runRSVP(sub_num, trial)

% Set number of runs 
nruns = 1; 

% Preallocate a cell array to collect data. 
C = cell((nruns*36),5);

 
% Perform standard setup for Psychtoolbox
PsychDefaultSetup(2);

% Define black, white, and gray
black = BlackIndex(0);
white = WhiteIndex(0);
gray = white / 2;


ListenChar(2);

% Open the window 
PsychImaging('PrepareConfiguration');
PsychImaging('AddTask', 'General', 'UseRetinaResolution');
[window, rect] = PsychImaging('OpenWindow', 0, gray);
%[0 0 1280 600]);
HideCursor;

% Get the center coordinates of the screen
[centerX, centerY] = RectCenter(rect);    
 
% Get the size of the screen window in pixels
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% % Disable all keys except for space bar, down arrow, right arrow
oldenablekeys = RestrictKeysForKbCheck([66,117,115]); %([44,79,81]);
 

% Display instructions for the task
instructions = 'Press the right arrow if you saw a letter and press the left arrow if you did not\n Press space to begin.\n';
Screen('TextFont', window, 'Courier');
Screen('TextSize', window, 40);
DrawFormattedText(window, instructions, 'center','center', 0, [], [], [], 1.5);
Screen('Flip', window);
 
% Wait until user presses a key
[~, ~, ~] = KbWait([], 2); 

stim = {'%', '#', '$', '@', '*', '^', '~', '!', '>', '?', ...
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', ...
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};



% Generate condition matrix in random order. 1-36 symbolizes each posisble stimuli, and
% nruns is how many instances of each stimuli we want. 
conditionMatrix = []; % preallocate 
for ii = 1:nruns 
    conditionMatrix = [conditionMatrix randperm(36)];
end 


for ti = 1:length(conditionMatrix) 
    Screen('TextFont', window, 'Courier');
    Screen('TextSize', window, 80);
    DrawFormattedText(window, stim{conditionMatrix(ti)}, 'center','center', 0, [], [], [], 1.5);
    
  % Save the time the screen was flipped
    stimulusStartTime = Screen('Flip', window);
    % Wait until user presses a key
    [~, ~, ~] = KbWait([], 2); 
    
    %save RT 
    [keyWasPressed, responseTime, key] = recordKeys(stimulusStartTime);
    %sprintf('%s', [responseTime key]) 
    

end

%% close out

finished = 'Press space to continue.';
Screen('TextFont', window, 'Courier');
Screen('TextSize', window, 40);
DrawFormattedText(window, finished, 'center', 'center', 0);
Screen('Flip', window);

KbWait;
sca;
Screen('close')
ListenChar(0);
