function maxabsdiff=tailscrap()
% test code for circular convolution with the scrapped portion 
% at the tail of the buffer, rather than the front
%
% The idea is to rotate the zero-padded h (impulse response) buffer
% to the left nh-1 samples, rotating the junk samples as well.
% This could be very handy in avoiding buffer copies during fast filtering.
nh=10;
nfft=256;

h=rand(1,nh);
x=rand(1,nfft);

hpad=[ h(nh) zeros(1,nfft-nh) h(1:nh-1) ]; 

% baseline comparison
y1 = filter(h,1,x);
y1_notrans = y1(nh:nfft);

% fast convolution
y2 = ifft( fft(hpad) .* fft(x) );
y2_notrans=y2(1:nfft-nh+1);

maxabsdiff = max(abs(y2_notrans - y1_notrans))

end
