function [trainSamples,trainLabels, testSamples, testLabels] = newData(type)

% setup default behaviour it will be adapted to the specific dataset if needed
n = 2; % Default number of features
m = 200; % Default number of samples

% Generate random values that will be used as dataset if no other dataset is provided

% setup random generator seed to default wich use Mersenne Twister with
% seed 0 this will let us setup what generator rand, randn and randi are using
rng('default');

% let's generate some data for each class, then we'll assign some of them to
% -1 class some others to +1 class.
N = floor(m/2);
M = floor(m/2);
plus1 = [1.8+0.6*randn(1,0.6*N), 1.7+0.7*randn(1,0.4*N);
    1.3*(randn(1,0.6*N)+1.2), 1.4*(randn(1,0.4*N)+1.1)];

% negative examples
minus1 = [-1.8+0.6*randn(1,0.6*M),  -1.7+0.7*randn(1,0.4*M);
    -1.3*(randn(1,0.6*M)-1.1), -1.4*(randn(1,0.4*M)-1)];

x = [minus1, plus1]; % x contains all the generated values
y = [-ones(1,N), ones(1,M)]; % labels
DSOrig = [ -((ones(n,1)*y).*x)', -y']; % generated dataset

% Load and use a provided dataset if it is available
if type == "load" % if want to use a specific dataset load it, else use default random values
    % import data
    file = uigetfile('*.xls',"Select your dataset",'dataset.xls');
    if(file) % file selected, use it
        DSTable = readtable(file,"VariableNamingRule","preserve");
        DSOrig=table2array(DSTable); % convert dataset table to array

        % setup number of samples and features
        n = size(DSOrig,2); % number of features
        m = size(DSOrig,1); % number of samples


    else
        (fprintf("No file selected. Random generated values dataset will be used.\n"));

    end
end

% Mixing dataset row's order
r = randperm(size(DSOrig,1)); % permute row numbers
Dataset = DSOrig(r,:); % Obtain the shuffled version of the original dataset

% Check labels and if needed replace 0 labels with -1 andle leave 1 as 1.
cols = size(Dataset,2); % considering last colum of the entire dataset as labels col
label = Dataset(:,cols);
label(label==0) = -1;
Dataset(:,cols) = label;

crossVal = cvpartition(m,'Holdout',0.1);
TrIdxCV = crossVal.test;
trainDS = Dataset(~TrIdxCV,:);
trainSamples = trainDS(:,1:(size(trainDS,2)-1));
trainLabels = trainDS(:,[size(trainDS,2)]);
testDS  = Dataset(TrIdxCV,:);
testSamples = testDS(:,1:(size(testDS,2)-1));
testLabels = testDS(:,[size(testDS,2)]);
