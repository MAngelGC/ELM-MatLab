clear;

D = load('handwriting.mat');
X = D.X;

[N, K] = size(X);
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
[N, K] = size(Xscaled);

CVHO = cvpartition(N,'HoldOut',0.25);

XscaledTrain = Xscaled(CVHO.training(1),:);
XscaledTest = Xscaled(CVHO.test(1),:);
YTrain = Y(CVHO.training(1),:);
YTest = Y(CVHO.test(1),:);


% Create the validation set
[NTrain, K] = size(XscaledTrain);
CVHOV = cvpartition(NTrain,'HoldOut',0.25);

% Generate the validation sets
XscaledTrainVal = XscaledTrain(CVHOV.training(1),:);
XscaledVal = XscaledTrain(CVHOV.test(1),:);
YTrainVal = YTrain(CVHOV.training(1),:);
YVal = YTrain(CVHOV.test(1),:);

[NTrainVal, K] = size(XscaledTrainVal);

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
        
        H = zeros(NTrainVal, L);
        H_Val = zeros(NVal, L);
        for n = 1:NTrainVal
            for l = 1:L
                t_arg = t(l)' * X(n);
                H(n, l) = inv((1 + exp(-t_arg)));
            end 
        end
        
        h_prod = H' * H;
        [dim1, dim2] = size(h_prod);
        I = eye(dim1, dim2);
        w = ((I/C) + h_prod)\H' * YTrainVal;

        for n = 1:NVal
            for l = 1:L
                t_arg = t(l)' * X_Val(n);
                H_Val(n, l) = inv((1 + exp(-t_arg)));
            end 
        end

        Yestimated = H_Val * w;

        Label = max(Yestimated, 1);

        Performance(i, j) = sum(Label == YVal)/NVal;
        % Implementar el ELM neuronal, calcular el rendimiento asociado a C
        % y L
        
    end
    j=0;
end

C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000];
L = [50 100 500 1000 1500 2000];

[maxValue, linearIndexesOfMaxes] = max(Performance(:));
[rowsOfMaxes colsOfMaxes] = find(Performance == maxValue);

Copt = C(rowsOfMaxes(1));
Lopt = L(colsOfMaxes(1));   

% Calcular con el conjunto de entrenamiento el ELM neuronal y
% reportar el error cometido en test

