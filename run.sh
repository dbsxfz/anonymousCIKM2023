# cd AugDF_github
echo "start......"
# pathname="arrhythmia_1_1_42.log"
# df without any aug
dataset=arrhythmia
model=df
mode=ori
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}

# gcforestcs without any aug
dataset=arrhythmia
model=cs
mode=ori
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}

# df with policy we searched and saved on df, this is for quick use policy
dataset=arrhythmia
model=df
mode=trans
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}

# cs with policy we searched and saved on df
dataset=arrhythmia
model=cs
mode=trans
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}

# search policy on df
dataset=arrhythmia
model=df
mode=search
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}

# search policy on gcforestcs
dataset=arrhythmia
model=cs
mode=search
random_state=42

str="dataset=${dataset},\n model=${model},\n mode=${mode},\n random_state=${random_state}\n"
pathname="${dataset}_${model}_${mode}_${random_state}.log"
echo -e $str
nohup python -u train.py --dataset ${dataset} --model ${model} --mode ${mode}  --random_state ${random_state} >records/${pathname}
echo "end......"