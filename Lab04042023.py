import numpy as np
import matplotlib.pyplot as plt
import math

def RCP(Tsym,alpha,span,L):
    t = np.arange(-0.5 * span,(0.5 * span) + (1/L),(1/L))

    P = np.zeros(len(t))
    print(len(t))
    for i in range(len(t)):
        
        N1 = np.pi * t[i] * (1 - alpha) / Tsym
        N2 = 4 * alpha * t[i] / Tsym
        N3 = np.pi / (4 * alpha)
        N4 = np.pi * t[i] * (1 + alpha) / Tsym

        Nr = np.sin(N1) + (N2 * np.cos(N4))
        Dr = (np.pi * t[i] / Tsym) * (1 - (N2**2))
        if i==(len(t)//2):
            P[i] = (1/Tsym) * ((1 - alpha) + (4*alpha/np.pi))
        elif abs(t[i]) == Tsym/(4*alpha):
            P[i] = (alpha / math.sqrt(2 * Tsym)) * (((1 + 2/np.pi) * np.sin(N3)) + ((1 - 2/np.pi) * np.cos(N3)))
        else:
            P[i] = (1/math.sqrt(Tsym)) * (Nr / Dr)

    return P


Tsym = 1
alpha = 0.3
span = 8*Tsym
L = 16

t = np.arange(-0.5 * span,(0.5 * span) + (1/L),(1/L))

Raised_Cosine_Pulse = RCP(Tsym,alpha,span,L)

values = np.random.randint(0,2,10)#(len(t)))

#print(values)

def upsampling(fun,L):
    y = np.zeros(len(fun)*L)

    for i in range(len(fun)):
        if fun[i]==0:
            y[i*L] = -1
        else:
            y[i*L] = 1
    return y

ups = upsampling(values,16)

filter_output = np.convolve(ups,Raised_Cosine_Pulse)

SNR_in_dB = 20

SNR = 10**(SNR_in_dB/10)

signal_power = np.sum(ups**2)/len(ups)


N0 = signal_power / SNR
print(N0)

noise = ((np.sqrt(N0/2))) * np.random.standard_normal(len(filter_output))
print(max(noise))
channel_output = filter_output + noise
rec_sig = np.convolve(channel_output,Raised_Cosine_Pulse)

plt.subplot(321)
plt.plot(t,Raised_Cosine_Pulse)
plt.title("Raised Cosine Pulse")
plt.subplot(322)
plt.title("Upsampled")
plt.stem(np.arange(len(ups)),ups)
plt.subplot(323)
plt.plot(np.arange(len(filter_output)),filter_output)
plt.title("Filter Output")
plt.subplot(324)
plt.title("Noise Signal")
plt.plot(np.arange(len(channel_output)),noise)
plt.subplot(325)
plt.title("Channel Output")
plt.plot(np.arange(len(channel_output)),channel_output)
plt.subplot(326)
plt.title("recieved signal")
plt.plot(np.arange(len(rec_sig)),rec_sig)
plt.show()