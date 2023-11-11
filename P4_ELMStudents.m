clear;

D = load('handwriting.mat');
X = D.X;

[N, ~] = size(X);
J = 10;

Y = zeros(N,J);

% Generate the Y Label
for i =1:10
    Y(1+(500*(i-1)):i*500,i) =1;
end

% Scale the data
Xscaled = (X-min(X))./(max(X)-min(X));

% Remove the NaN elements
Xscaled = Xscaled(:,any(~isnan(Xscaled)));

% Compute again the number of total elements and attributes
[N, ~] = size(Xscaled);

CVHO = cvpartition(N,'HoldOut',0.25);

XscaledTrain = Xscaled(CVHO.training(1),:);
XscaledTest = Xscaled(CVHO.test(1),:);
YTrain = Y(CVHO.training(1),:);
YTest = Y(CVHO.test(1),:);


% Create the validation set
[NTrain, ~] = size(XscaledTrain);
CVHOV = cvpartition(NTrain,'HoldOut',0.25);

% Generate the validation sets
XscaledTrainVal = XscaledTrain(CVHOV.training(1),:);
XscaledVal = XscaledTrain(CVHOV.test(1),:);
YTrainVal = YTrain(CVHOV.training(1),:);
YVal = YTrain(CVHOV.test(1),:);

[NTrainVal, ~] = size(XscaledTrainVal);

[NVal, K] = size(XscaledVal);

% Performance Matrix
Performance = zeros(7,6);

i = 0;
j = 0;
% Estimate the hyper-parameters values
for C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000]
    i = i+1;
    for L = [50 100 500 1000 1500 2000]
        j = j+1;
        
        X = [XscaledTrainVal -ones(NTrainVal, 1)];
        X_Val = [XscaledVal -ones(NVal, 1)];
        t = 2 * rand(L, K+1) - 1;

        t_arg = X * t';
        H = 1./(1 + exp(-t_arg));
        
        h_prod = H' * H;
        [dim1, dim2] = size(h_prod);
        I = eye(dim1, dim2);
        w = ((I/C) + h_prod)\H' * YTrainVal;

        t_arg = X_Val * t';
        H_Val = 1./(1 + exp(-t_arg));

        Yestimated = H_Val * w;

        Label = max(Yestimated, [], 2);
        Acc_matrix = Yestimated == Label;

        cnt = 0;

        for idx = 1:NVal
            if isequal(YVal(idx, :), Acc_matrix(idx, :))
                cnt = cnt + 1;
            end
        end
        Performance(i, j) = cnt/NVal;
        % Implementar el ELM neuronal, calcular el rendimiento asociado a C
        % y L
        
    end
    j=0;
end

C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000];
L = [50 100 500 1000 1500 2000];

[maxValue, linearIndexesOfMaxes] = max(Performance(:));
[rowsOfMaxes, ~] = find(Performance == maxValue);

Copt = C(rowsOfMaxes(1));
Lopt = L(colsOfMaxes(1));   

% Calcular con el conjunto de entrenamiento el ELM neuronal y
% reportar el error cometido en test

[NTest, ~] = size(XscaledTest);
[NTrain, K] = size(XscaledTrain);

X = [XscaledTrain -ones(NTrain, 1)];
X_Test = [XscaledTest -ones(NTest, 1)];
t = 2 * rand(Lopt, K+1) - 1;

t_arg = X * t';
H = 1./(1 + exp(-t_arg));

h_prod = H' * H;
[dim1, dim2] = size(h_prod);
I = eye(dim1, dim2);
w = ((I/Copt) + h_prod)\H' * YTrain;

t_arg = X_Test * t';
H_Test = 1./(1 + exp(-t_arg));

Yestimated = H_Test * w;

Label = max(Yestimated, [], 2);
Acc_matrix = Yestimated == Label;

cnt = 0;

for idx = 1:NTest
    if isequal(YTest(idx, :), Acc_matrix(idx, :))
        cnt = cnt + 1;
    end
end

Accuracy = cnt/NTest;
ECM = sum(sum((YTest - (H_Test * w)).^2)/NTest);

disp("Accuracy: " + Accuracy);
disp("ECM: " + ECM);
