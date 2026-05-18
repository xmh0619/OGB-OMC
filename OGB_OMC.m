clc
clear
%%load the dataset
dname = 'emotions-scale';
load([dname '-train.mat']);
load([dname '-test.mat']);
data=[trainData;testData];
labels=[trainLabel;testLabel];

[T, d] = size(data);
L = size(labels, 2);
maxiter=L;
initNum=round(T*0.05);
labels = 2 * labels - 1;

%hyperparameters for OGB-OMC
N =25;
eta=0.4;
scale=2.^-2;
times=20;

sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

%%for storing the performance metrics' values
hammingloss = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
subsetAccuracy=zeros(times,1);
rankingLoss=zeros(times,1);
oneError=zeros(times,1);
macro_F1_score =zeros(times,1);
micro_F1_score =zeros(times,1);
F1=zeros(times,1);

tStrat=tic;
for run=1:times
    SVsIdx = cell(L, N); 
    coff = cell(L, N);  
    SVsNum=cell(L,N);
    Y_bar_t = zeros(T, L);
    index=randperm(T);

    for l = 1:L
        for k = 1:N
            SVsIdx{l,k} = [];
            coff{l,k} = [];
            SVsNum{l,k}=0;
        end
    end

    kernelMatrix=importdata(['kernel_all\' dname '\kernelMatrix_scale_' num2str(scale) '.mat']);


    % initialData=data(index(1:initNum),:);
    % kernelMatrix=rbfkernel(initialData',initialData',scale);
    for j = 1:initNum
        t=index(j);
        x_t = data(t, :)';
        Y_t = labels(t, :);
        Y_t_size = sum(Y_t == 1);
        for l = 1:L
            s_t = zeros(1, N+1);
            for iter = 1:maxiter
                for k = 1:N
                    if SVsNum{l,k}==0
                        h_t_k_l = 0;
                    else
                        % SVsMatrix=data(SVsIdx{l,k},:);
                        % km=rbfkernel(SVsMatrix',x_t,scale);
                        km=kernelMatrix(SVsIdx{l,k},t);
                        h_t_k_l = km'* coff{l,k};
                    end
                    eta_k = 2/(k+1);
                    s_t(k+1) = (1-eta_k)*s_t(k) + eta_k*h_t_k_l;
                end
                Y_bar_t(t, l) = sign(s_t(N+1));
                if Y_t(l) == 1
                    a_t = 1/Y_t_size;
                else
                    a_t = 1/(L - Y_t_size);
                end

                for k = 1:N
                    if Y_t(l) * s_t(k) < 1
                        count = -a_t * Y_t(l);
                    else
                        count = 0;
                    end
                    id = find(SVsIdx{l,k} == t, 1);
                    if isempty(id)
                        SVsNum{l,k} = SVsNum{l,k} + 1;
                        SVsIdx{l,k} = [SVsIdx{l,k}, t];
                        coff{l,k} = [coff{l,k}; -eta * count];
                    else
                        coff{l,k}(id) = coff{l,k}(id) - eta * count;
                    end
                end
            end
        end
    end

    %%online model updating using
    tp = zeros(1,L);
    fp = zeros(1,L);
    fn = zeros(1,L);
    Y_pred_scores=zeros(1,L); 
    for i = initNum+1:T
        w_prev_v=zeros(L,N);
        t=index(i);
        x_t = data(t, :)';  
        Y_t=labels(t,:);
        Y_t_size=sum(Y_t==1);
        s_t = zeros(L, N+1);
        for l = 1:L
            for k = 1:N
                SVsMatrix = data(SVsIdx{l,k},:);
                km = rbfkernel(SVsMatrix',x_t,scale);
                if SVsNum{l,k}==0
                    w_prev_v(l,k)= 0;
                else
                    w_prev_v(l,k)= km' * coff{l,k};
                end
                eta_k = 2/(k+1);
                s_t(l,k+1) = (1-eta_k)*s_t(l,k) + eta_k*w_prev_v(l,k);
            end
            Y_pred_scores(1, l) = s_t(l,N+1);
        end
%%%%%%%%%%%%%%%---online metrics calculation--%%%%%%%%%%%%%%%%%%%%%
        Y_t=(Y_t+1)/2;
        Y_pred=sign(Y_pred_scores);
        Y_pred(Y_pred==-1)=0;
        hammingloss(run) =hammingloss(run) + nnz(Y_pred+Y_t==1)/L;
        if nnz(Y_pred) ~= 0
            precision(run)=precision(run)+(Y_pred*Y_t')/nnz(Y_pred);
        elseif nnz(Y_t)==0
            precision(run)=precision(run)+1;
        end

        if nnz(Y_t) ~= 0
            recall(run) = recall(run)+(Y_pred*Y_t') / nnz(Y_t);
        elseif nnz(Y_pred)==0
            recall(run) = recall(run)+1;
        end

        if Y_pred == Y_t
            subsetAccuracy(run) = subsetAccuracy(run) + 1;
        end
        rele = find(Y_t);
        irrele = find(~Y_t);
        if ~isempty(rele) && ~isempty(irrele)
            misorder = 0;
            for kk = 1:size(rele,2)
                misorder = misorder + nnz(Y_pred_scores(rele(kk)) <= Y_pred_scores(irrele));
            end
            rankingLoss(run)= rankingLoss(run)+ misorder/(size(rele,2)*size(irrele,2));
            [~,id] = max(Y_pred_scores(1:L));
            if Y_t(id) == 0
                oneError(run)= oneError(run)+ 1;
            end
        end
        for kk = 1:L
            if Y_t(kk) == 1 && Y_pred(kk) == 1
                tp(kk) = tp(kk) + 1;
            elseif Y_t(kk) == 1 && Y_pred(kk) == 0
                fn(kk) = fn(kk) + 1;
            elseif Y_t(kk) == 0 && Y_pred(kk) == 1
                fp(kk) = fp(kk) + 1;
            end
        end

%%%%%%%%update model%%%%%%%
        Y_t=2*Y_t-1;
        for l=1:L
            if Y_t(l)==1
                a_t=1/Y_t_size;
            else
                a_t=1/(L-Y_t_size);
            end
            for iter=1:maxiter
                for k=1:N
                    if Y_t(l) * s_t(l,k) < 1
                        count = -a_t * Y_t(l);
                    else
                        count = 0;
                    end
                    id = find(SVsIdx{l,k} == t, 1);
                    if isempty(id)
                        SVsNum{l,k} = SVsNum{l,k} + 1;
                        SVsIdx{l,k} = [SVsIdx{l,k}, t];
                        coff{l,k} = [coff{l,k}; -eta * count];
                    else
                        coff{l,k}(id) = coff{l,k}(id) - eta * count;
                    end
                    w_prev_v(l,k)=w_prev_v(l,k)-eta*count;
                end
                for k=1:N
                    s_t(l,1)=0;
                    eta_k = 2/(k+1);
                    s_t(l,k+1) = (1-eta_k)*s_t(l,k) + eta_k*w_prev_v(l,k);
                end
            end
        end
    end
    t = T - initNum; % the number of examples evaluated
    hammingloss(run)=hammingloss(run) / t;
    precision(run)=precision(run) /t;
    recall(run)=recall(run) /t;
    F1(run)=2*precision(run)*recall(run)/(precision(run)+recall(run));
    subsetAccuracy(run) = subsetAccuracy(run)/t;
    rankingLoss(run) = rankingLoss(run)/t;
    oneError(run) = oneError(run)/t;
    % macro_F1_score = 0;
    for kk = 1:L
        this_F = 0;
        if tp(kk) ~= 0 || fp(kk) ~= 0 || fn(kk) ~= 0
            this_F = (2*tp(kk)) / (2*tp(kk) + fp(kk) + fn(kk));
        end
        macro_F1_score(run) = macro_F1_score(run)+ this_F;
    end
    macro_F1_score(run) = macro_F1_score(run)/ L;
    micro_F1_score(run) = (2*sum(tp)) / (2*sum(tp) + sum(fp) + sum(fn));
end
totalTime=toc(tStrat);
avgTime=totalTime/times;
fprintf('Hamming Loss= %.4f\n', mean(hammingloss));
fprintf('precision=%.4f\n',mean(precision));
fprintf('recall=%.4f\n',mean(recall));
fprintf('F1=%.4f\n',mean(F1));
fprintf('macroF1=%.4f\n',mean(macro_F1_score));
fprintf('microF1=%.4f\n',mean(micro_F1_score));
fprintf('rankingLoss=%.4f\n',mean(rankingLoss));
