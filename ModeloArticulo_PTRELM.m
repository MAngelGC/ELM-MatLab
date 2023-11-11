clear;
clc;

D = load('handwriting.mat');
X = D.X;

[N, ~] = size(X);
J = 10;

Y = zeros(N,J);

% Generate the Y Label
for i =1:10
    Y(1+(500*(i-1)):i*500,i)=1;
end

% Escalamos los datos
Xscaled = (X - min(X))./(max(X) - min(X));

% Quitamos los elementos NaN
Xscaled = Xscaled(:, any(~isnan(Xscaled)));

% Recalculamos el número de elementos y atributos
[N, ~] = size(Xscaled);

% Partimos los datos en train y test
CVHO = cvpartition(N, 'HoldOut', .25);

% Creamos los conjuntos de entrenamiento y test para la fase de
% entrenamiento
XscaledTrain = Xscaled(CVHO.training(1), :);
XscaledTest = Xscaled(CVHO.test(1), :);
YTrain = Y(CVHO.training(1), :);
YTest = Y(CVHO.test(1), :);

% Volvemos a aprtir los datos
[NTrain, ~] = size(XscaledTrain);
CVHO = cvpartition(NTrain, 'HoldOut', .25);

% Creamos los conjuntos de entrenamiento y test para la fase de validación
XscaledTrainVal = XscaledTrain(CVHO.training(1), :);
XscaledVal = XscaledTrain(CVHO.test(1), :);
YTrainVal = Y(CVHO.training(1), :);
YVal = Y(CVHO.test(1), :);

[NTrainVal, ~] = size(XscaledTrainVal);

[NVal, K] = size(XscaledVal);

% Inicializamos la matriz de rendimiento
Num_tests_C = 6;
Performance = zeros(1, Num_tests_C);

% Inicializamos los parámetros necesarios
L = 1000; % Número de neuronas en capa oculta, fijo en este caso pero 
% podría estimarse también
M_per_class = cell(1, J); % Matriz M para cada clase
M = zeros(L + 1, L + 1); % Matriz M total
H = cell(1, J); % Matriz H
H_Val = cell(1, J); % Matriz H de validación
B = cell(1, J); % Vector B (contiene Bj de cada clase)
I = eye(L + 1, L + 1); % Matriz identidad
Yestimated = cell(1, J); % Vector de valores Y estimados para cada conjunto
% de patrones pertenecientes a una clase
Label_vector = cell(1, J); % Vector de etiquetas de clase
Acc = cell(1, J); % Vector de comparación de salidas y etiquetas de clase

i = 0;

for C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000]
    i = i + 1;
    cnt = 0;
    for j = 1:J
        X = XscaledTrainVal(YTrainVal(:, j)==1, :); % Seleccionamos los
        % patrones que pertenezcan a la clase 'j'
        
        % Calculamos la matriz Hj
        [Nj, ~] = size(X);
        t = 2 * rand(L, K) - 1;
        t_arg = X * t';
        h = 1./(1 + exp(-t_arg));
        H{j} = h;

        % Calculamos la matriz H_Valj
        t_arg = XscaledVal * t';
        h_val = 1./(1 + exp(-t_arg));
        H_Val{j} = h_val;

        % Asignamos Mj [Nj -sum(h(x))'; -sum(h(x)) Hj' * Hj]
        M_per_class{j} = zeros(L + 1, L + 1);
        M_per_class{j}(1, 1) = Nj;
        M_per_class{j}(1, 2:end) = -sum(h)';
        M_per_class{j}(2:end, 1) = -sum(h);
        M_per_class{j}(2:end, 2:end) = H{j}' * H{j};

        % Sumamos a la matriz M total la Mj resultante
        M = M + M_per_class{j};
    end

    for j = 1:J
        % Calculamos los autovectores de M y Mj + C*I
        [V, D] = eig(M, M_per_class{j} + C*I);

        % Asignamos a Bj el autovector que de mayor autovalor
        B{j} = V(:, max(diag(D)) == diag(D));
        B0j = B{j}(1);
        Bj = B{j}(2:end);

        % Calculamos la salida 
        Yestimated{j} = abs(H_Val{j}*Bj - B0j)/norm(Bj);

        % Asignamos a la etiqueta de la clase j el valor mínimo de las
        % salidas
        Label_vector{j} = min(Yestimated{j}, [], 1);
        Acc{j} = Yestimated{j} == Label_vector{j};

        % Calculamos el número de aciertos en YVal
        for idx = 1:NVal
            if isequal(YVal(idx, j), Acc{j}(idx))
                cnt = cnt + 1;
            end
        end
    end

    % Calculamos la precisión para el valor de C usado y la guardamos
    % en Performance
    Performance(i) = cnt/(NVal * J);
end


% Una vez probados todos los valores de C, seleccionamos el que mayor
% accuracy reporte
C = [10^(-3) 10^(-2) 10^(-1) 1 10 100 1000];

Copt = min(C(Performance == max(Performance)));

% Inicializamos los parámetros necesarios
L = 1000; % Número de neuronas en capa oculta, fijo en este caso pero 
% podría estimarse también
M_per_class = cell(1, J); % Matriz M para cada clase
M = zeros(L + 1, L + 1); % Matriz M total
H = cell(1, J); % Matriz H
H_Test = cell(1, J); % Matriz H de test
B = cell(1, J); % Vector B (contiene Bj de cada clase)
I = eye(L + 1, L + 1); % Matriz identidad
Yestimated = cell(1, J); % Vector de valores Y estimados para cada conjunto
% de patrones pertenecientes a una clase
Label_vector = cell(1, J); % Vector de etiquetas de clase
Acc = cell(1, J); % Vector de comparación de salidas y etiquetas de clase

cnt = 0;

[NTest, K] = size(XscaledTest);
for j = 1:J
        X = XscaledTrain(YTrain(:, j)==1, :); % Seleccionamos los
        % patrones que pertenezcan a la clase 'j'
        [Nj, ~] = size(X);

        % Calculamos la matriz Hj
        t = 2 * rand(L, K) - 1;
        t_arg = X * t';
        h = 1./(1 + exp(-t_arg));
        H{j} = h;

        % Calculamos la matriz H_Testj
        t_arg = XscaledTest * t';
        h_test = 1./(1 + exp(-t_arg));
        H_Test{j} = h_test;

        % Asignamos Mj [Nj -sum(h(x))'; -sum(h(x)) Hj' * Hj]
        M_per_class{j} = zeros(L + 1, L + 1);
        M_per_class{j}(1, 1) = Nj;
        M_per_class{j}(1, 2:end) = -sum(h)';
        M_per_class{j}(2:end, 1) = -sum(h);
        M_per_class{j}(2:end, 2:end) = H{j}' * H{j};

        % Sumamos a la matriz M total la Mj resultante
        M = M + M_per_class{j};
end

for j = 1:J
    % Calculamos los autovectores de M y Mj + Copt*I
    [V, D] = eig(M, M_per_class{j} + Copt*I);

    % Asignamos a Bj el autovector que de mayor autovalor
    B{j} = V(:, max(diag(D)) == diag(D));
    B0j = B{j}(1);
    Bj = B{j}(2:end);

    % Calculamos la salida
    Yestimated{j} = abs(H_Test{j}*Bj - B0j)/norm(Bj);

    % Asignamos a la etiqueta de la clase j el valor mínimo de las
    % salidas
    Label_vector{j} = min(Yestimated{j}, [], 1);
    Acc{j} = Yestimated{j} == Label_vector{j};

    % Calculamos el número de aciertos en el conjunto de test
    for idx = 1:NTest
        if isequal(YTest(idx, j), Acc{j}(idx))
            cnt = cnt + 1;
        end
    end
end

% Calculamos la precisión final
Accuracy = cnt/(NTest * J);

% Calculamos el ECM final
ECM = 0;

for j = 1:J
    ECM = ECM + sum((YTest(:, j) - Acc{j}).^2);
end

ECM = ECM/(NTest * J);

% Mostramos el ECM y la precisión finales
disp("Accuracy: " + Accuracy);
disp("ECM: " + ECM);
