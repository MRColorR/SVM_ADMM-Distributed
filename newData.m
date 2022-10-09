function [trainSamples,trainLabels, trainA, testSamples, testLabels] = newData(type)
% setup default behaviour it will be adapted to the specific dataset if needed
n = 2; % Default number of features
m = 250; % Default number of samples
% Generate random values that will be used as dataset if no other dataset is provided
% setup random generator seed to default wich use Mersenne Twister with
% seed 0 this will let us setup what generator rand, randn and randi are using
rng('default');
if (type ~= "load")
    DSOrig = rndDataGen(m);
end
% Load and use a provided dataset if it is available
if type == "load" % if want to use a specific dataset load it, else use default random values
    % import data
    file = uigetfile('*.xls',"Select your dataset",'dataset.xls');
    if(file) % file selected, use it
        DSTable = readtable(file,"VariableNamingRule","preserve");
        DSOrig=table2array(DSTable); % convert dataset table to array
        DSOrig=DSOrig';
        % setup number of samples and features
        n = size(DSOrig,1); % number of features+labels
        m = size(DSOrig,2); % number of samples
    else
        (fprintf("No file selected. Random generated values dataset will be used.\n"));
        DSOrig = rndDataGen(m);
    end
end

% Mixing dataset
r = randperm(size(DSOrig,2)); % permute columns numbers
Dataset = DSOrig(:,r); % Obtain the shuffled version of the original dataset

% Check labels and if needed replace 0 labels with -1 andle leave 1 as 1.
rows = size(Dataset,1); % considering last row of the entire dataset as labels row
label = Dataset(rows,:);
label(label==0) = -1;
Dataset(rows,:) = label;

% Preparing training and test datasets
crossVal = cvpartition(m,'Holdout',0.15);
TrIdxCV = crossVal.test;
trainDS = Dataset(:,~TrIdxCV);
trainSamples = trainDS(1:(size(trainDS,1)-1),:);
trainLabels = trainDS([size(trainDS,1)],:);
trainA = [ -((ones(n-1,1)*trainLabels).*trainSamples)' -trainLabels'];
testDS  = Dataset(:,TrIdxCV);
testSamples = testDS(1:(size(testDS,1)-1),:);
testLabels = testDS([size(testDS,1)],:);

function DSOrig = rndDataGen(m)
% let's generate some data for each class, then we'll assign some of them to
% -1 class some others to +1 class.
Q = floor(m/2);
plus1 = [1.8+0.6*randn(1,0.6*Q), 1.7+0.7*randn(1,0.4*Q);
    1.3*(randn(1,0.6*Q)+1.2), 1.4*(randn(1,0.4*Q)+1.1)];
minus1 = [-1.8+0.6*randn(1,0.6*Q),  -1.7+0.7*randn(1,0.4*Q);
    -1.3*(randn(1,0.6*Q)-1.1), -1.4*(randn(1,0.4*Q)-1)];
x = [minus1, plus1]; % x contains all the generated values
y = [ones(1,Q), -ones(1,Q)]; % labels
DSOrig = [x;y]; % generated dataset
