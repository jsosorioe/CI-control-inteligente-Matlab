%%

% y(k) = f( y(k-1),y(k-2),u(k-1),u(k-2) )
% y1 = y(k-1)
% y2 = y(k-2)

% u1 = u(k-1)
% u2 = u(k-2)


% y(1) = f( y(0),y(-1),u(0),u(-1) )

% y(3) = f( y(2),y(1),u(2),u(1) )

% targ = b(3:end);
% y1 = b(2:end-1);
% y2 = b(1:end-2);

%% Tarea 4: Algoritmo de Retropropagación
clear
clc

%% Inicialización de las variables

N = 4;          %Número de entradas
M = 7;          %Número de neurodos de la capa intermedia
J = 1;          %Número de neurodos en la capa de salida
L = 100;          %Número de épocas

eta   = 0.001;      %Learning Rate
alpha = 0.25;      %Función de activación

%Función de activación es una sigmoidal y se define con
%los siguientes parámetros:
%[función,derivada] = activation(entrada,bias,alpha)

%% Cargar los vectores de entrada (X) y de verificación (TARG)
    
load('data2.mat')
X = [y1,y2,u1,u2];

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

% Variables para guardar pesos
saveWh(:,:,1) = [Wh;Bh];
saveWo(:,:,1) = [Wo;Bo];

%% Separar datos de ENTRENAMINETO y VALIDACIÓN

Q_t = 6500;       %Número de ejemplos de entrenamiento
Q_v = 1500;       %Número de ejemplos de entrenamiento

X_train = X(1:Q_t,:);
T_train = targ(1:Q_t,:);

X_valid = X(Q_t:end,:);%DUDA
T_valid = targ(Q_t:end,:);%DUDA

%% Comienza la actualización de pesos

%Ciclo por número de epocas
for r=1:L
    
    % ENTRENAMIENTO
    E = 0;
    %Ciclo por número de ejemplos
    for q=1:Q_t
        
        % Calculo de las salidas
        
        R = [X_train(q,:),1]*[Wh;Bh];  %SUMATORIA CAPA OCULTA
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
                    delta_bo = eta*(T_train(q,j)-Z(j))*derz(j)*1;
                    Bo_new(j) = Bo(j) + delta_bo;
                else
                    delta_o = eta*(T_train(q,j)-Z(j))*derz(j)*Y(i);
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
                        Suma = Suma + (T_train(q,k)-Z(k))*derz(k)*Wo(j,k);
                    end
                    delta_bh = eta*Suma*dery(j)*1;
                    Bh_new(j)= Bh(j) + delta_bh;
                else
                    Suma = 0;
                    for k=1:J
                        Suma = Suma + (T_train(q,k)-Z(k))*derz(k)*Wo(j,k);
                    end
                    delta_h = eta*Suma*dery(j)*X_train(q,i);
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
    end
    
    VAF1(r) = 100*(1-(var(T_train(:,1)-out(:,1)))/var(T_train(:,1)));
    
    TMSE(r) = sqrt((1/(Q_t*J))*E);
    
    %GUARDAR PESOS
    saveWh(:,:,r+1) = [Wh;Bh];
    saveWo(:,:,r+1) = [Wo;Bo];

end

figure(1)
clf()
plot(TMSE)
grid on
title('Error Total Medio Cuadrático')
xlabel('Epoca')
ylabel('Error')

figure(2)
clf()
plot(VAF1)
grid on
title('Variance Accounted For (VAF)')
xlabel('Epoca')
ylabel('VAF (%)')

figure(3)
t=0.4:0.2:1600;
clf()
plot(t,targ)
%hold on
%plot(t,Z)
title('Salida')
xlabel('Epoca')
ylabel('VAF (%)')

%%
%salida
figure(4)
t=0.4:0.2:1600;
clf()
plot(t,y1)
%hold on
%plot(t,Z)
title('Salida')
xlabel('Epoca')
ylabel('VAF (%)')

figure(5)
t=0.4:0.2:1600;
clf()
plot(t,y2)
%hold on
%plot(t,Z)
title('Salida')
xlabel('Epoca')
ylabel('VAF (%)')
%%
%entrada
figure(6)
t=0.4:0.2:1600;
clf()
plot(t,u1)
%hold on
%plot(t,Z)
title('Salida')
xlabel('Epoca')
ylabel('VAF (%)')

figure(7)
t=0.4:0.2:1600;
clf()
plot(t,u2)
%hold on
%plot(t,Z)
title('Salida')
xlabel('Epoca')
ylabel('VAF (%)')