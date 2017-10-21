clc; clear;
f1=100;f2=200;%���˲������ź�Ƶ��
fs=2000;%����Ƶ��
m=(0.3*f1)/(fs/2);%������ȴ���
M=round(8/m);%���崰�����ĳ���
N=M-1;%�����˲����Ľ���
b=fir1(N,0.5*f2/(fs/2));%ʹ��fir1��������˲���
%����Ĳ����ֱ����˲����Ľ����ͽ�ֹƵ��
figure(1)
[h,f]=freqz(b,1,512);%�˲����ķ�Ƶ����ͼ
%[H,W]=freqz(B,A,N)��N��һ������ʱ��������N���Ƶ�������ͷ�Ƶ��Ӧ����
plot(f*fs/(2*pi),20*log10(abs(h)))%�����ֱ���Ƶ�����ֵ
xlabel('Ƶ��/����');ylabel('����/�ֱ�');title('�˲�����������Ӧ');
figure(2)
subplot(211)
t=0:1/fs:0.5;%����ʱ�䷶Χ�Ͳ���
s=sin(2*pi*f1*t)+sin(2*pi*f2*t);%�˲�ǰ�ź�
plot(t,s);%�˲�ǰ���ź�ͼ��
xlabel('ʱ��/��');ylabel('����');title('�ź��˲�ǰʱ��ͼ');
subplot(212)
Fs=fft(s,512);%���źű任��Ƶ��
AFs=abs(Fs);%�ź�Ƶ��ͼ�ķ�ֵ
f=(0:255)*fs/512;%Ƶ�ʲ���
plot(f,AFs(1:256));%�˲�ǰ���ź�Ƶ��ͼ
xlabel('Ƶ��/����');ylabel('����');title('�ź��˲�ǰƵ��ͼ');
figure(3)
sf=filter(b,1,s);%ʹ��filter�������źŽ����˲�
%�����ֱ�Ϊ�˲���ϵͳ�����ķ��Ӻͷ�ĸ����ʽϵ�������ʹ��˲��ź�����
subplot(211)
plot(t,sf)%�˲�����ź�ͼ��
xlabel('ʱ��/��');ylabel('����');title('�ź��˲���ʱ��ͼ');
axis([0.2 0.5 -2 2]);%�޶�ͼ�����귶Χ
subplot(212)
Fsf=fft(sf,512);%�˲�����ź�Ƶ��ͼ
AFsf=abs(Fsf);%�ź�Ƶ��ͼ�ķ�ֵ
f=(0:255)*fs/512;%Ƶ�ʲ���
plot(f,AFsf(1:256))%�˲�����ź�Ƶ��ͼ
xlabel('Ƶ��/����');ylabel('����');title('�ź��˲���Ƶ��ͼ');