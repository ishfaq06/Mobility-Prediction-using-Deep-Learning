%%loading data set
clear all
close all
load('realitymining.mat', 's');
%% creating data set
clear data;
k=1;

i=4;
temp_vect=datevec(s(i).locs(1,1));
start_month=temp_vect(2);
month_change=[];
itr=1;

   
   % data(i,1)=i;
    
    if (isempty(s(i).locs))
        data(k,1)=i;
        data(k,2)=0;
        k=k+1;
    else
        for l=1:1:length(s(i).locs)-1
            g=datevec(s(i).locs(l+1,1)-s(i).locs(l,1));
            temp_vect=datevec(s(i).locs(l,1));
            temp_month=temp_vect(2);
            if temp_month ~= start_month
                start_month=temp_month;
                month_change(itr)=l;
                itr=itr+1;
            end
            
            time=g(3)*24*3600+g(4)*60*60+g(5)*60+g(6);
            data_temp(1,l)=time;
            data_temp(2,l)=temp_vect(4);
        end
        
        for j=6:1:length(s(i).locs)-1
            data(k,1)=i;
            data(k,[2,4,6,8,10,15])=s(i).loc_ids(j-5:j);
            data(k,[3,5,7,9,11,16])=data_temp(1,j-5:j);
            data(k,12)=weekday(s(i).locs(j));
            data(k,13)=data_temp(2,j-1);
            k=k+1
        end
        
        
    end

%save(string(datetime("now"))+".mat", 'data','month_change');
%% mobility data set
clear all
close all
%load('23-Feb-2018 15:51:10.mat')
%load('23-Feb-2018 16:18:09.mat') % cell id 3
%load('23-Feb-2018 16:27:23.mat'); % cell id 13
%load('28-Feb-2018 16:15:11.mat') % cell id 4
load('06-Mar-2018 13:45:16.mat') % cell id 4

%

clear net

A=1:length(data); 
frac=0.3;
mon=6;
len=length(data);
idx=randperm(numel(A));

s1=A(idx(1:frac*length(data)));
%test_data=data(s1,:);
test_data=data(month_change(mon)+1:month_change(mon+1),:);

test_data_input=(test_data(:,[2:1:10,12,13]))';
test_data_output_temp=(test_data(:,15))';
test_data_output=zeros(40,length(test_data_output_temp));

for i=1:length(test_data_output_temp)
    
    s=num2str(test_data_output_temp(i));
    k=3;
    for j=length(s):-1:1
        test_data_output(10*k+str2double(s(j))+1,i)=1;
        k=k-1;
    end
    
  %  test_data_output(test_data_output_temp(i),i)=10;
    
end


subSet2=A(idx(frac*length(data)+1:end));
%s2=sort(subSet2);
s2=subSet2;
%training_data=data(s2,:);
training_data=data(month_change(mon-3)+1:month_change(mon),:);
training_data_input=(training_data(:,[2:1:10,12,13]))';
training_data_target_temp=(training_data(:,15))';
training_data_target=zeros(40,length(training_data_target_temp));

for i=1:length(training_data_target_temp)
    
    s=num2str(training_data_target_temp(i));
    k=3;
    for j=length(s):-1:1
        training_data_target(10*k+str2double(s(j))+1,i)=1;
        k=k-1;
    end    
    
  %  training_data_target(training_data_target_temp(i),i)=10;
    
end
% training the irish data set

%input_data=(data(:,2:2:12))';
%output_data=(data(:,15))';
%%


clear net;
net = feedforwardnet([100, 90, 80, 70, 60, 50, 40],'trainscg');
%net = patternnet([100, 70, 50],'trainscg');
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'tansig';

net.trainParam.epochs=1000;
net.trainParam.max_fail=10;
net = train(net,training_data_input,training_data_target);
%% testing
dt=net(test_data_input);
%dt=round(dt);
%%
count=0;
for i=1:length(test_data_output)
  % [v1 d1]=max(test_data_output(:,i));
   val=0;
    k=0;
    for j=4:-1:1
        [v d]=max(test_data_output((j-1)*10+1:j*10,i));
        val=val+10^(k)*(d-1);
        k=k+1;
    end
    d1=val;
    
    val=0;
    k=0;
    for j=4:-1:1
        [v d]=max(dt((j-1)*10+1:j*10,i));
        val=val+10^(k)*(d-1);
        k=k+1;
    end
    d2=val;
   
   %[v2 d2]=max(dt(:,i));
   if (d1==d2)
       count=count+1;
   end
    
end
corrected_answer=count;
%corrected_answer=sum((dt == test_data_output));
errr=(length(test_data_output)-corrected_answer)/length(test_data_output)*100
%%
clear dd;
dd=zeros(1,40);
a=4;
s=num2str(a);
k=3;
for i=length(s):-1:1
   dd(1,10*k+str2num(s(i))+1)=1;
   k=k-1;
end


val=0;
k=0;
for i=4:-1:1
    [v d]=max(dd(1,(i-1)*10+1:i*10));
    val=val+10^(k)*(d-1);
    k=k+1;
end
val
%%
l=1;
 temp_vect=datevec(s(i).locs(l+1,1));
            temp_month=temp_vect(2);
            if temp_month ~= start_month
                star_month=temp_month
                month_change(itr)=l;
                itr=itr+1;
            end
%% using auto encoder
clear deepnet;
clear autoenc1;
clear autoenc2;
clear softnet;
rng('default')
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(training_data_input,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
%view(autoenc1)
%
feat1 = encode(autoenc1,training_data_input);
%
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
%view(autoenc2)
%
feat2 = encode(autoenc2,feat1);
%
softnet = trainSoftmaxLayer(feat2,training_data_target,'MaxEpochs',500);
%view(softnet)
%
deepnet = stack(autoenc1,autoenc2,softnet);



%
deepnet = train(deepnet,training_data_input,training_data_target);
clear dt;
%%
dt=deepnet(test_data_input);



