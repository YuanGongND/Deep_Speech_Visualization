Fs=1000;  
[b,a]=butter(9,300/(Fs/2),'high') 

freqz(b,a,128,Fs) 