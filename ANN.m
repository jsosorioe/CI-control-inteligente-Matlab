%% Tarea 4: Algoritmo de Retropropagación
clear
clc
%% Inicialización de las variables

N = 2;          %Número de entradas
M = 3;          %Número de neurodos de la capa intermedia
J = 1;          %Número de neurodos en la capa de salida
Q = 4;          %Número de ejemplos
L = 10;          %Número de épocas

eta   = 1;      %Learning Rate
alpha = 1;      %Función de activación

%Función de activación es una sigmoidal y se define con
%los siguientes parámetros:
%[función,derivada] = activation(entrada,bias,alpha)

%% Cargar los vectores de entrada (X) y de verificación (TARG)
    
% %Saturadas
% X = [0,0;
%      0,1;
%      1,0;
%      1,1];
%  
% targ = [0;
%         1;
%         1;
%         0];

% NO Saturadas
 
targ = [0.1 0.9;
        0.9 0.1;
        0.9 0.1;
        0.1 0.9];

%% Inicializar los pesos de la capa oculta y de salida

% Capa oculta
Wh = (0.1)*rand(N,M);
%Wh = [0 0;0.1 0.1]
Bh = ones(1,M)*-1;
%Bh = [0 0.1]

% Capa salida
Wo = (0.1)*rand(M,J);
%Wo = [0.1;0.1]
Bo = ones(1,J)*-1;
%Bo = [0.1]

%% Matriz con datos de validación

for i = 1:L
    valid(i+(i-1)*3,:)   = [rand(1)*(0.15 - 0.05) + 0.05,rand(1)*(0.15 - 0.05) + 0.05,0.1,0.9];
    valid(i+1+(i-1)*3,:) = [rand(1)*(0.15 - 0.05) + 0.05,rand(1)*(0.95 - 0.85) + 0.85,0.9,0.1];
    valid(i+2+(i-1)*3,:) = [rand(1)*(0.95 - 0.85) + 0.85,rand(1)*(0.15 - 0.05) + 0.05,0.9,0.1];
    valid(i+3+(i-1)*3,:) = [rand(1)*(0.95 - 0.85) + 0.85,rand(1)*(0.95 - 0.85) + 0.85,0.1,0.9];
end

sizeV = size(valid);

%% Variables para guardar pesos

saveWh(:,:,1) = [Wh;Bh];
saveWo(:,:,1) = [Wo;Bo];

%% Comienza la actualización de pesos

%Ciclo por número de epocas
for r=1:L
    
     X = [rand(1)*(0.15 - 0.05) + 0.05,rand(1)*(0.15 - 0.05) + 0.05;
          rand(1)*(0.15 - 0.05) + 0.05,rand(1)*(0.95 - 0.85) + 0.85;
          rand(1)*(0.95 - 0.85) + 0.85,rand(1)*(0.15 - 0.05) + 0.05;
          rand(1)*(0.95 - 0.85) + 0.85,rand(1)*(0.95 - 0.85) + 0.85];
    
    E = 0;
       
    %Ciclo por número de ejemplos
    for q=1:Q
        % Calculo de las salidas
        R = [X(q,:),1]*[Wh;Bh];  %SUMATORIA CAPA OCULTA
        for i=1:M
            [Y(i),dery(i)] = activation(R(i),alpha); %SALIDA CAPA OCULTA
        end
        S = [Y,1]*[Wo;Bo];       %SUMATORIA CAPA SALIDA
        for i=1:J
            [Z(i),derz(i)] = activation(S(i),alpha);  %SALIDA ANN
        end
        
        %Actualización de los pesos de la capa de salida
        for i=1:M+1
            for j=1:J
                if i == M+1
                    delta_bo = eta*(targ(q,j)-Z(j))*derz(j)*1;
                    Bo_new(j) = Bo(j) + delta_bo;
                else
                    delta_o = eta*(targ(q,j)-Z(j))*derz(j)*Y(i);
                    Wo_new(i,j) = Wo(i,j) + delta_o;
                end
                
            end
        end
        Wo = Wo_new;
        Bo = Bo_new;
        
        %Actualización de los pesos de la capa oculta
        for i=1:N+1
            for j=1:M
                if i == N+1
                    Suma = 0;
                    for k=1:J
                        Suma = Suma + (targ(q,k)-Z(k))*derz(k)*Wo(j,k);
                    end
                    delta_bh = eta*Suma*dery(j)*1;
                    Bh_new(j)= Bh(j) + delta_bh;
                else
                    Suma = 0;
                    for k=1:J
                        Suma = Suma + (targ(q,k)-Z(k))*derz(k)*Wo(j,k);
                    end
                    delta_h = eta*Suma*dery(j)*X(q,i);
                    Wh_new(i,j) = Wh(i,j) + delta_h;
                end 
            end
        end
        
        Wh = Wh_new;
        Bh = Bh_new;
        
        for i = 1:J
            E = E + (targ(q,i)-Z(i))^2;
        end
        
        out(q,:) = Z;
        
        % saveXTZ(q+4*(r-1),:) = [X(q,:),targ(q,:),Z];
    end
    
   % VAF1(r) = 100*(1-(var(targ(:,1)-out(:,1)))/var(targ(:,1)));
    %VAF2(r) = 100*(1-(var(targ(:,2)-out(:,2)))/var(targ(:,2)));
   % TMSE(r) = sqrt((1/(Q*J))*E);
    
    %VALIDACIÓN
    a = randi(sizeV(1));
    X2 = [valid(a,1),valid(a,2)];
    T2 = [valid(a,3),valid(a,4)];
    
    % Calculo de las salidas
    R2 = [X2,1]*[Wh;Bh];  %SUMATORIA CAPA OCULTA
    for i=1:M
     [Y2(i),dery2(i)] = activation(R2(i),alpha); %SALIDA CAPA OCULTA
    end
    S2 = [Y2,1]*[Wo;Bo];       %SUMATORIA CAPA SALIDA
    for i=1:J
     [Z2(i),derz2(i)] = activation(S2(i),alpha);  %SALIDA ANN
    end
    
    E2 = 0;
    for i = 1:J
        E2 = E2 + (valid(a,2+i)-Z2(i))^2;
    end
    
    TMSE_v(r) = sqrt((1/J)*E2);
    
    %GUARDAR PESOS
    saveWh(:,:,r+1) = [Wh;Bh];
    saveWo(:,:,r+1) = [Wo;Bo];

end

figure(1)
clf()
plot(TMSE)
hold on
grid on
plot(TMSE_v)
legend('Entrenamiento','Validación')
title('Error Total Medio Cuadrático')
xlabel('Epoca')
ylabel('Error')

figure(2)
clf()
plot(VAF1)
hold on
plot(VAF2)
grid on
legend('VAF1','VAF2')
title('Variance Accounted For (VAF)')
xlabel('Epoca')
ylabel('VAF (%)')


figure(3)
clf
hold on
grid on
for j = 1:N+1
    for k = 1:M
        for i = 1:L+1
            Wh11(i) = saveWh(j,k,i);
        end
        plot(Wh11)
    end
end
title('Actualización de los pesos capa Oculta')
xlabel('Epoca')
ylabel('Peso')
legend('11','12','13','21','22','23','B1','B2','B3')

figure(4)
clf
grid on
hold on
for j = 1:M+1
    for k = 1:J
        for i = 1:L+1
            Wh11(i) = saveWo(j,k,i);
        end
        plot(Wh11)
    end
end
title('Actualización de los pesos capa Salida')
xlabel('Epoca')
ylabel('Peso')
legend('11','12','21','22','31','32','B1','B2')

%% VERIFICACIÓN CON NUEVAS ENTRADAS

% X2 = [0,0]
% 
% % Calculo de las salidas
% R = [X2,1]*[Wh;Bh];  %SUMATORIA CAPA OCULTA
% for i=1:M
%     [Y(i),dery(i)] = activation(R(i),alpha); %SALIDA CAPA OCULTA
% end
% S = [Y,1]*[Wo;Bo];       %SUMATORIA CAPA SALIDA
% for i=1:J
%     [Z(i),derz(i)] = activation(S(i),alpha);  %SALIDA ANN
% end
% Z

%% RECTAS SEPARADORAS

X1 = 0:0.025:1;
X2 = 0:0.025:1;

%INICIALES
for i=1:41
    for j=1:41
        X = [X1(i),X2(j)];
        % Calculo de las salidas
        R = [X,1]*saveWh(:,:,1);  %SUMATORIA CAPA OCULTA
        for m=1:M
            [Y(m),dery(m)] = activation(R(m),alpha); %SALIDA CAPA OCULTA
        end
        S = [Y,1]*saveWo(:,:,1);       %SUMATORIA CAPA SALIDA
        for k=1:J
            [Z(k),derz(k)] = activation(S(k),alpha);  %SALIDA ANN
        end
        Zrectas1i(i,j) = Z(1);
        Zrectas2i(i,j) = Z(2);
    end
end

%FINALES
for i=1:41
    for j=1:41
        X = [X1(i),X2(j)];
        % Calculo de las salidas
        R = [X,1]*saveWh(:,:,2001);  %SUMATORIA CAPA OCULTA
        for m=1:M
            [Y(m),dery(m)] = activation(R(m),alpha); %SALIDA CAPA OCULTA
        end
        S = [Y,1]*saveWo(:,:,2001);       %SUMATORIA CAPA SALIDA
        for k=1:J
            [Z(k),derz(k)] = activation(S(k),alpha);  %SALIDA ANN
        end
        Zrectas1f(i,j) = Z(1);
        Zrectas2f(i,j) = Z(2);
    end
end

figure(5)
clf()
subplot(2,2,1)
contourf(X1,X2,Zrectas1i)
subplot(2,2,2)
contourf(X1,X2,Zrectas1f)
subplot(2,2,3)
contourf(X1,X2,Zrectas2i)
subplot(2,2,4)
contourf(X1,X2,Zrectas2f)